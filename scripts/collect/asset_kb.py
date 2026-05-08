"""Asset knowledge-base lookup + HuggingFace size resolution.

Loads scripts/collect/asset_kb.json and exposes helpers used by
extract_assets.py:

- normalize(name) → canonical KB key (lowercase, alias-folded).
- lookup_dataset(name) / lookup_model(name) → KB entry or None.
- resolve_hf_model_size(hf_id) / resolve_hf_dataset_size(hf_id) →
  total size in bytes for the largest weight files, with on-disk caching.
- enrich(asset, container_type) → fills in canonical name, size_bytes,
  size_human, size_source on an LLM-produced asset dict.
- human_size(n) → "~150 GB" / "440 MB" / "—".

Resolution priority: KB > HF API > LLM-extracted > LLM-estimated > unknown.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("asset_kb")

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent
_KB_PATH = _HERE / "asset_kb.json"
_CACHE_PATH = _REPO_ROOT / "data" / ".asset_size_cache.json"

# HF weight file extensions we count toward total size.
_HF_WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx", ".msgpack", ".h5")

_kb: dict[str, Any] | None = None
_cache: dict[str, Any] | None = None
_cache_lock = threading.Lock()


def _load_kb() -> dict[str, Any]:
    global _kb
    if _kb is None:
        with open(_KB_PATH, encoding="utf-8") as f:
            _kb = json.load(f)
    return _kb


def _load_cache() -> dict[str, Any]:
    global _cache
    if _cache is None:
        if _CACHE_PATH.exists():
            try:
                with open(_CACHE_PATH, encoding="utf-8") as f:
                    _cache = json.load(f)
            except Exception:
                _cache = {}
        else:
            _cache = {}
    return _cache


def _save_cache() -> None:
    if _cache is None:
        return
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_cache, f, indent=2, ensure_ascii=False, sort_keys=True)
    tmp.replace(_CACHE_PATH)


# ---------------------------------------------------------------------------
# Normalization & KB lookup
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def normalize(name: str) -> str:
    """Lowercase, strip punctuation, collapse spaces. Used as KB key."""
    if not name:
        return ""
    s = name.strip().lower()
    # Preserve a few separators that show up in canonical keys ('-' and '/').
    # We fold everything to a single '-' separator so 'CLIP ViT-L/14' →
    # 'clip-vit-l-14'.
    s = s.replace("/", "-")
    s = _PUNCT_RE.sub("-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


_PAREN_RE = re.compile(r"\s*\([^)]*\)")


def _strip_parens(s: str) -> str:
    """Remove parenthetical clarifications: 'ResNet-101 (ImageNet pretrained)' -> 'ResNet-101'."""
    return _PAREN_RE.sub("", s).strip()


def _resolve_alias(name: str, alias_map: dict[str, str]) -> str:
    """Try a few normalization variants against the alias map.

    Order: raw lowercase, normalized, paren-stripped lowercase, paren-stripped+normalized.
    Falls back to plain normalize() so a direct KB key match still works.
    """
    if not name:
        return name
    bare = _strip_parens(name)
    candidates = [
        name.strip().lower(),
        normalize(name),
        bare.strip().lower(),
        normalize(bare),
    ]
    seen = set()
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            if c in alias_map:
                return alias_map[c]
    return normalize(bare or name)


def _direct_key_candidates(name: str) -> list[str]:
    """Normalized keys to try directly against the KB top-level dict."""
    bare = _strip_parens(name)
    return [normalize(name), normalize(bare)]


def lookup_dataset(name: str) -> Optional[dict[str, Any]]:
    if not name:
        return None
    kb = _load_kb()
    aliases = kb.get("_aliases", {}).get("datasets", {})
    key = _resolve_alias(name, aliases)
    entry = kb["datasets"].get(key)
    if entry is not None:
        return {**entry, "_kb_key": key}
    # Fallback: try the normalized form(s) directly as KB keys.
    for k in _direct_key_candidates(name):
        if k in kb["datasets"]:
            return {**kb["datasets"][k], "_kb_key": k}
    return None


def lookup_model(name_or_hf_id: str) -> Optional[dict[str, Any]]:
    if not name_or_hf_id:
        return None
    kb = _load_kb()
    aliases = kb.get("_aliases", {}).get("models", {})
    key = _resolve_alias(name_or_hf_id, aliases)
    entry = kb["models"].get(key)
    if entry is not None:
        return {**entry, "_kb_key": key}
    # Fallback: try normalized variants directly as KB keys.
    for k in _direct_key_candidates(name_or_hf_id):
        if k in kb["models"]:
            return {**kb["models"][k], "_kb_key": k}
    # Try matching by huggingface_id field across all KB entries.
    target_lower = name_or_hf_id.strip().lower()
    bare_lower = _strip_parens(name_or_hf_id).strip().lower()
    for k, v in kb["models"].items():
        hf = (v.get("huggingface_id") or "").lower()
        if hf and (hf == target_lower or hf == bare_lower):
            return {**v, "_kb_key": k}
    return None


# ---------------------------------------------------------------------------
# HuggingFace size resolution (cached on disk)
# ---------------------------------------------------------------------------

def _hf_api():
    """Lazy import so the module works without huggingface_hub installed."""
    try:
        from huggingface_hub import HfApi
        return HfApi()
    except Exception as e:
        log.debug(f"huggingface_hub unavailable: {e}")
        return None


def _sum_weight_files(siblings) -> int:
    total = 0
    counted = 0
    for s in siblings or []:
        name = getattr(s, "rfilename", "") or ""
        size = None
        # Some HF SDK versions expose size on .size or .lfs.size
        if hasattr(s, "size") and isinstance(s.size, int):
            size = s.size
        elif hasattr(s, "lfs") and s.lfs and hasattr(s.lfs, "size"):
            size = s.lfs.size
        if size is None or size <= 0:
            continue
        if name.lower().endswith(_HF_WEIGHT_EXTS):
            total += size
            counted += 1
    if counted == 0:
        return 0
    return total


def resolve_hf_model_size(hf_id: str) -> Optional[int]:
    """Return total bytes of weight files for an HF model id, cached on disk."""
    if not hf_id:
        return None
    cache = _load_cache()
    key = f"model::{hf_id}"
    if key in cache:
        v = cache[key]
        return v.get("size_bytes") if isinstance(v, dict) else None
    api = _hf_api()
    if api is None:
        return None
    try:
        info = api.model_info(hf_id, files_metadata=True)
        size = _sum_weight_files(getattr(info, "siblings", None))
        with _cache_lock:
            cache[key] = {"size_bytes": size or None, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            _save_cache()
        return size or None
    except Exception as e:
        log.debug(f"HF model_info failed for {hf_id}: {e}")
        with _cache_lock:
            cache[key] = {"size_bytes": None, "error": str(e)[:120], "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            _save_cache()
        return None


_HF_SEARCH_URL = "https://huggingface.co/api/{kind}"


def _hf_search(name: str, kind: str, limit: int = 5) -> list[str]:
    """Hit the public HF Hub search API, return up to `limit` repo IDs.

    `kind` ∈ {"datasets", "models"}. No auth required, no rate limit observed
    for low-volume queries.
    """
    if not name:
        return []
    url = _HF_SEARCH_URL.format(kind=kind) + "?" + urllib.parse.urlencode(
        {"search": name, "limit": str(limit), "sort": "downloads", "direction": "-1"}
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "bamboo-asset-kb/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log.debug(f"HF search failed for {kind}/{name!r}: {e}")
        return []
    ids = []
    for item in data:
        rid = item.get("id") if isinstance(item, dict) else None
        if isinstance(rid, str):
            ids.append(rid)
    return ids


def _name_score(query: str, candidate: str) -> float:
    """Crude relevance score for a HF id given a search query.

    Returns 0.0 unless the leaf token-set is essentially the query.
    Short queries (≤ 5 chars after normalization) require an exact
    leaf match; longer queries allow normalized-equality or full
    token-set equality. Substring matches are deliberately rejected
    because HF Hub returns popular but unrelated repos for short
    abbreviations (e.g., 'SAN' → 'tiny-random-sana').
    """
    if not candidate or not query:
        return 0.0
    leaf = candidate.split("/")[-1].lower()
    q_norm = re.sub(r"[^a-z0-9]+", "", query.lower())
    l_norm = re.sub(r"[^a-z0-9]+", "", leaf)
    if not q_norm or not l_norm:
        return 0.0
    # Exact normalized match always wins.
    if l_norm == q_norm:
        return 1.0
    # Short queries: require exact match only.
    if len(q_norm) <= 5:
        return 0.0
    # Longer queries: allow strict prefix or full token-set equality.
    if l_norm.startswith(q_norm) and len(l_norm) <= len(q_norm) + 4:
        return 0.85
    qt = set(t for t in re.split(r"[^a-z0-9]+", query.lower()) if t)
    lt = set(t for t in re.split(r"[^a-z0-9]+", leaf) if t)
    if qt and lt and qt == lt:
        return 0.95
    if qt and lt and qt <= lt and len(lt - qt) <= 1:
        return 0.7
    return 0.0


def hf_search_dataset(name: str) -> Optional[str]:
    """Best-guess HF dataset repo id for a free-text name; cached on disk."""
    if not name:
        return None
    cache = _load_cache()
    key = f"search-dataset::{name.strip().lower()}"
    if key in cache:
        v = cache[key]
        return v.get("hf_id") if isinstance(v, dict) else None
    candidates = _hf_search(name, kind="datasets", limit=10)
    best, best_score = None, 0.0
    for c in candidates:
        s = _name_score(name, c)
        if s > best_score:
            best, best_score = c, s
    # Require a meaningful match so we don't pick the top-downloaded random repo.
    if best_score < 0.5:
        best = None
    with _cache_lock:
        cache[key] = {"hf_id": best, "score": best_score, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        _save_cache()
    return best


def hf_search_model(name: str) -> Optional[str]:
    """Best-guess HF model repo id for a free-text name; cached on disk."""
    if not name:
        return None
    cache = _load_cache()
    key = f"search-model::{name.strip().lower()}"
    if key in cache:
        v = cache[key]
        return v.get("hf_id") if isinstance(v, dict) else None
    candidates = _hf_search(name, kind="models", limit=10)
    best, best_score = None, 0.0
    for c in candidates:
        s = _name_score(name, c)
        if s > best_score:
            best, best_score = c, s
    if best_score < 0.5:
        best = None
    with _cache_lock:
        cache[key] = {"hf_id": best, "score": best_score, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        _save_cache()
    return best


def resolve_hf_dataset_size(hf_id: str) -> Optional[int]:
    """Return approximate total bytes for an HF dataset (LFS data files)."""
    if not hf_id:
        return None
    cache = _load_cache()
    key = f"dataset::{hf_id}"
    if key in cache:
        v = cache[key]
        return v.get("size_bytes") if isinstance(v, dict) else None
    api = _hf_api()
    if api is None:
        return None
    try:
        info = api.dataset_info(hf_id, files_metadata=True)
        total = 0
        for s in getattr(info, "siblings", None) or []:
            size = None
            if hasattr(s, "size") and isinstance(s.size, int):
                size = s.size
            elif hasattr(s, "lfs") and s.lfs and hasattr(s.lfs, "size"):
                size = s.lfs.size
            if isinstance(size, int) and size > 0:
                total += size
        with _cache_lock:
            cache[key] = {"size_bytes": total or None, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            _save_cache()
        return total or None
    except Exception as e:
        log.debug(f"HF dataset_info failed for {hf_id}: {e}")
        with _cache_lock:
            cache[key] = {"size_bytes": None, "error": str(e)[:120], "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            _save_cache()
        return None


# ---------------------------------------------------------------------------
# Human-readable sizes
# ---------------------------------------------------------------------------

def human_size(size_bytes: Optional[int]) -> str:
    """Format as '~150 GB' / '440 MB'. Returns '—' for None or 0."""
    if size_bytes is None or size_bytes <= 0:
        return "—"
    units = [("TB", 1024 ** 4), ("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024), ("B", 1)]
    for unit, factor in units:
        if size_bytes >= factor:
            value = size_bytes / factor
            if value >= 100:
                return f"{value:.0f} {unit}"
            if value >= 10:
                return f"{value:.1f} {unit}"
            return f"{value:.2f} {unit}"
    return f"{size_bytes} B"


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

def enrich(asset: dict[str, Any], *, container: str) -> dict[str, Any]:
    """Fill in size/canonical fields on an LLM-produced asset dict.

    container: "datasets" | "model_weights" | "other_assets"

    Resolution priority for size_bytes (and matching size_source):
        kb       — curated KB hit by name or huggingface_id
        hf_api   — resolved via huggingface_hub
        extracted — paper or README explicitly stated a size (LLM passed it through)
        estimated — LLM size guess
        unknown  — we have nothing
    """
    out = dict(asset)
    name = (out.get("name") or "").strip()
    hf_id = (out.get("huggingface_id") or "").strip() or None

    kb_hit = None
    if container == "datasets":
        kb_hit = lookup_dataset(name)
    elif container == "model_weights":
        kb_hit = lookup_model(hf_id or name)

    if kb_hit:
        # Override or fill from KB
        out["name"] = kb_hit.get("canonical") or out.get("name") or name
        if kb_hit.get("size_bytes") is not None:
            out["size_bytes"] = int(kb_hit["size_bytes"])
            out["size_source"] = "kb"
        if kb_hit.get("url") and not out.get("url"):
            out["url"] = kb_hit["url"]
        if kb_hit.get("license") and not out.get("license"):
            out["license"] = kb_hit["license"]
        if kb_hit.get("huggingface_id") and not out.get("huggingface_id"):
            out["huggingface_id"] = kb_hit["huggingface_id"]
            hf_id = kb_hit["huggingface_id"]

    # If no KB hit and we have an HF id, try HF API.
    if out.get("size_source") != "kb" and hf_id:
        if container == "model_weights":
            sz = resolve_hf_model_size(hf_id)
        elif container == "datasets":
            sz = resolve_hf_dataset_size(hf_id)
        else:
            sz = None
        if sz:
            out["size_bytes"] = sz
            out["size_source"] = "hf_api"

    # Long-tail fallback: search HF Hub for a best-match repo, then resolve.
    if out.get("size_source") not in ("kb", "hf_api") and not hf_id:
        try:
            if container == "datasets":
                guessed = hf_search_dataset(name)
                if guessed:
                    sz = resolve_hf_dataset_size(guessed)
                    if sz:
                        out["huggingface_id"] = guessed
                        out["size_bytes"] = sz
                        out["size_source"] = "hf_search"
            elif container == "model_weights":
                guessed = hf_search_model(name)
                if guessed:
                    sz = resolve_hf_model_size(guessed)
                    if sz:
                        out["huggingface_id"] = guessed
                        out["size_bytes"] = sz
                        out["size_source"] = "hf_search"
        except Exception as e:
            log.debug(f"HF search fallback failed for {name!r}: {e}")

    # If still nothing, keep the LLM value as 'extracted' or 'estimated'.
    if "size_source" not in out:
        if isinstance(out.get("size_bytes"), int) and out["size_bytes"] > 0:
            # Trust LLM-passed-through size as 'estimated' unless it explicitly
            # tagged otherwise (extract_assets.py preserves a 'size_source' from
            # the LLM when set).
            out["size_source"] = "estimated"
        else:
            out["size_bytes"] = None
            out["size_source"] = "unknown"

    out["size_human"] = human_size(out.get("size_bytes"))
    return out
