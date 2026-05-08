"""URL access-pattern classifier + resolvers (UCI, Zenodo) + manual queue.

Loads scripts/collect/access_patterns.json (a hand-curated host -> behavior
map) and exposes:

- classify_url(url) -> Classification dict telling extract_assets.py whether
  a URL is automatable, what method to use, and what auth (if any) is needed.
- probe_url(url) -> follow redirects, return final URL + status code; used
  to refine doi.org links and detect dead URLs.
- resolve_uci(name) / resolve_uci_doi(doi) -> hit UCI's JSON API and HEAD
  the data file to fill size_bytes for tabular assets.
- resolve_zenodo(doi_or_record) -> fetch Zenodo record, sum file sizes.
- enqueue_manual_action(paper_id, asset, classification) -> append to
  data/manual_action_queue.json so a human can resolve later.
- record_unknown_host(url, probe_result) -> append a stub entry to
  access_patterns.json so the next encounter has prior context.
"""
from __future__ import annotations

import fnmatch
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

log = logging.getLogger("access_patterns")

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent
_REGISTRY_PATH = _HERE / "access_patterns.json"
_QUEUE_PATH = _REPO_ROOT / "data" / "manual_action_queue.json"
_PROBE_CACHE_PATH = _REPO_ROOT / "data" / ".url_probe_cache.json"

_registry: dict[str, Any] | None = None
_registry_lock = threading.Lock()
_queue_lock = threading.Lock()
_probe_cache: dict[str, Any] | None = None
_probe_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Registry I/O
# ---------------------------------------------------------------------------

def _load_registry() -> dict[str, Any]:
    global _registry
    if _registry is None:
        with open(_REGISTRY_PATH, encoding="utf-8") as f:
            _registry = json.load(f)
    return _registry


def _save_registry() -> None:
    if _registry is None:
        return
    tmp = _REGISTRY_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_registry, f, indent=2, ensure_ascii=False)
    tmp.replace(_REGISTRY_PATH)


def _load_probe_cache() -> dict[str, Any]:
    global _probe_cache
    if _probe_cache is None:
        if _PROBE_CACHE_PATH.exists():
            try:
                with open(_PROBE_CACHE_PATH, encoding="utf-8") as f:
                    _probe_cache = json.load(f)
            except Exception:
                _probe_cache = {}
        else:
            _probe_cache = {}
    return _probe_cache


def _save_probe_cache() -> None:
    if _probe_cache is None:
        return
    _PROBE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _PROBE_CACHE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_probe_cache, f, indent=2, ensure_ascii=False, sort_keys=True)
    tmp.replace(_PROBE_CACHE_PATH)


# ---------------------------------------------------------------------------
# URL parsing + classification
# ---------------------------------------------------------------------------

def _host_of(url: str) -> str:
    try:
        h = urllib.parse.urlparse(url).hostname or ""
    except Exception:
        return ""
    return h.lower()


def _host_matches(host: str, pattern: str) -> bool:
    """fnmatch-style host comparison: '*.sharepoint.com' matches 'foo.sharepoint.com'.

    Also auto-strips a leading 'www.' from the host so 'www.kaggle.com' matches
    a 'kaggle.com' pattern, since registry entries use the canonical host.
    """
    if not host or not pattern:
        return False
    if pattern == host:
        return True
    if host.startswith("www.") and pattern == host[4:]:
        return True
    if "*" in pattern and (fnmatch.fnmatchcase(host, pattern)
                            or (host.startswith("www.") and fnmatch.fnmatchcase(host[4:], pattern))):
        return True
    return False


def _classify_via_registry(url: str) -> Optional[dict[str, Any]]:
    """Return a classification copied from the registry, or None if no match."""
    host = _host_of(url)
    if not host:
        return None
    reg = _load_registry()
    patterns = reg.get("patterns") or []
    # Score by specificity: exact host > glob host. Within each, regex match
    # beats no regex.
    best = None
    best_score = -1
    for entry in patterns:
        hp = entry.get("host_pattern", "")
        if not _host_matches(host, hp):
            continue
        url_re = entry.get("url_regex")
        regex_match = True
        if url_re:
            try:
                regex_match = bool(re.search(url_re, url))
            except re.error:
                regex_match = False
        if not regex_match:
            continue
        score = 0
        if "*" not in hp:
            score += 100
        if url_re:
            score += 10
        if score > best_score:
            best, best_score = entry, score
    if best is None:
        return None
    return {
        "access_pattern": best.get("access_pattern"),
        "automation_status": best.get("automation_status"),
        "auth_required": best.get("auth_required"),
        "method": best.get("method"),
        "notes": best.get("notes"),
        "matched_host_pattern": best.get("host_pattern"),
        "matched_url_regex": best.get("url_regex"),
    }


def classify_url(url: str) -> dict[str, Any]:
    """Classify a URL. Always returns a dict (never None) — fields will be
    'unknown' if no registry entry matches and probing didn't reveal more.
    """
    if not url or not isinstance(url, str):
        return {
            "access_pattern": "unknown",
            "automation_status": "unknown",
            "auth_required": "unknown",
            "method": None,
            "notes": "no URL provided",
            "url": url,
        }

    hit = _classify_via_registry(url)
    if hit is not None:
        hit["url"] = url
        return hit

    # No registry hit — probe to learn at minimum reachability.
    probe = probe_url(url)
    final = probe.get("final_url") or url
    if probe.get("ok") and final != url:
        hit = _classify_via_registry(final)
        if hit is not None:
            hit["url"] = url
            hit["resolved_url"] = final
            hit["notes"] = (hit.get("notes") or "") + f" | redirected from {url}"
            return hit

    # Still unknown. Stub-register the host so future runs can hand-edit it.
    record_unknown_host(url, probe)

    if not probe.get("ok"):
        return {
            "access_pattern": "unreachable",
            "automation_status": "unreachable",
            "auth_required": "unknown",
            "method": None,
            "notes": f"probe failed: {probe.get('error') or probe.get('status')}",
            "url": url,
            "resolved_url": final,
        }
    return {
        "access_pattern": "unknown",
        "automation_status": "unknown",
        "auth_required": "unknown",
        "method": "host not in access_patterns.json — added stub for human review",
        "notes": f"final url: {final}; status: {probe.get('status')}",
        "url": url,
        "resolved_url": final,
    }


# ---------------------------------------------------------------------------
# HEAD/GET probing
# ---------------------------------------------------------------------------

def probe_url(url: str, *, method: str = "HEAD", timeout: int = 15) -> dict[str, Any]:
    """Issue HEAD (fall back to GET on 405) and return:
        ok, status, final_url, content_length, content_type, error.
    Cached by (method, url) on disk so re-runs are cheap.
    """
    if not url or not isinstance(url, str):
        return {"ok": False, "error": "empty url"}

    cache = _load_probe_cache()
    key = f"{method}::{url}"
    if key in cache:
        return cache[key]

    headers = {"User-Agent": "bamboo-extract-assets/1.0 (+https://github.com/xln3/bamboo-papers)"}
    req = urllib.request.Request(url, method=method, headers=headers)
    out: dict[str, Any] = {"ok": False, "url": url, "method": method}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out["ok"] = True
            out["status"] = resp.status
            out["final_url"] = resp.geturl()
            cl = resp.headers.get("Content-Length")
            if cl and cl.isdigit():
                out["content_length"] = int(cl)
            out["content_type"] = resp.headers.get("Content-Type")
    except urllib.error.HTTPError as e:
        if e.code == 405 and method == "HEAD":
            with _probe_cache_lock:
                cache[key] = {"ok": False, "fallback_to_get": True}
                _save_probe_cache()
            return probe_url(url, method="GET", timeout=timeout)
        out["status"] = e.code
        out["error"] = f"HTTP {e.code} {e.reason}"
        try:
            out["final_url"] = e.geturl()
        except Exception:
            pass
    except urllib.error.URLError as e:
        out["error"] = f"URLError: {getattr(e, 'reason', e)}"
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"

    out["probed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _probe_cache_lock:
        cache[key] = out
        _save_probe_cache()
    return out


def record_unknown_host(url: str, probe_result: dict[str, Any]) -> None:
    """Add a stub registry entry for a host we've never seen before, so the
    next encounter has prior context AND a human can refine it.
    """
    host = _host_of(probe_result.get("final_url") or url)
    if not host:
        return
    with _registry_lock:
        reg = _load_registry()
        patterns = reg.setdefault("patterns", [])
        # If a pattern for this exact host already exists, do nothing.
        for p in patterns:
            if p.get("host_pattern") == host:
                return
        stub = {
            "host_pattern": host,
            "url_regex": None,
            "access_pattern": "unknown",
            "automation_status": "unknown",
            "auth_required": "unknown",
            "method": None,
            "notes": (
                f"AUTO-STUB: first seen {time.strftime('%Y-%m-%d', time.gmtime())}; "
                f"probe status {probe_result.get('status')}; "
                f"sample url: {url[:200]}"
            ),
            "verified_at": None,
        }
        patterns.append(stub)
        _save_registry()
        log.info(f"  [registry] auto-stubbed unknown host {host}")


# ---------------------------------------------------------------------------
# UCI ML Repository resolver
# ---------------------------------------------------------------------------

_UCI_DOI_RE = re.compile(r"^10\.24432/", re.IGNORECASE)
_UCI_DATASET_PAGE_RE = re.compile(r"https?://archive\.ics\.uci\.edu/dataset/(\d+)/", re.IGNORECASE)
_UCI_API_BY_NAME = "https://archive.ics.uci.edu/api/dataset?name={name}"
_UCI_API_BY_ID = "https://archive.ics.uci.edu/api/dataset?id={id}"


def _http_get_json(url: str, timeout: int = 30) -> Optional[dict]:
    cache = _load_probe_cache()
    key = f"GETJSON::{url}"
    if key in cache:
        return cache[key].get("json") if isinstance(cache[key], dict) else None
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "bamboo-extract-assets/1.0", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        with _probe_cache_lock:
            cache[key] = {"json": data, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            _save_probe_cache()
        return data
    except Exception as e:
        log.debug(f"  GET JSON failed for {url}: {e}")
        with _probe_cache_lock:
            cache[key] = {"json": None, "error": str(e)[:200]}
            _save_probe_cache()
        return None


def _http_size(url: str, timeout: int = 30, *, stream_cap_mb: int = 1024) -> Optional[int]:
    """Determine the size of a URL. Tries HEAD's Content-Length first; if the
    server uses chunked encoding (no Content-Length), falls back to a streamed
    GET that reads up to stream_cap_mb (default 1 GB) and counts. Cached.

    The cap protects against streaming a multi-TB resource by accident; data
    files in academic repositories are typically <1 GB.
    """
    cache = _load_probe_cache()
    key = f"SIZE::{url}"
    if key in cache:
        v = cache[key]
        if isinstance(v, dict):
            return v.get("size_bytes")
    p = probe_url(url, method="HEAD", timeout=timeout)
    if p.get("content_length"):
        size = int(p["content_length"])
        with _probe_cache_lock:
            cache[key] = {"size_bytes": size, "method": "head", "probed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            _save_probe_cache()
        return size
    # No Content-Length (chunked) — stream and count. Long timeouts since
    # large-file streams can take minutes.
    req = urllib.request.Request(url, headers={"User-Agent": "bamboo-extract-assets/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=max(timeout, 300)) as resp:
            cap = stream_cap_mb * 1024 * 1024
            total = 0
            while True:
                chunk = resp.read(1024 * 64)
                if not chunk:
                    break
                total += len(chunk)
                if total > cap:
                    total = -1
                    break
        if total < 0:
            with _probe_cache_lock:
                cache[key] = {"size_bytes": None, "method": "stream", "exceeded_cap_mb": stream_cap_mb}
                _save_probe_cache()
            return None
        with _probe_cache_lock:
            cache[key] = {"size_bytes": total, "method": "stream", "probed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            _save_probe_cache()
        return total
    except Exception as e:
        log.debug(f"  size probe failed for {url}: {e}")
        with _probe_cache_lock:
            cache[key] = {"size_bytes": None, "method": "stream", "error": str(e)[:200]}
            _save_probe_cache()
        return None


def resolve_uci(name_or_url_or_doi: str) -> Optional[dict[str, Any]]:
    """Resolve a UCI dataset by name, DOI (10.24432/...), or archive.ics.uci.edu URL.

    Returns dict with: uci_id, canonical_name, data_url, size_bytes, license,
    repository_url, num_instances, num_features. Or None if not resolvable.
    """
    if not name_or_url_or_doi:
        return None
    s = name_or_url_or_doi.strip()
    api_url = None

    m = _UCI_DATASET_PAGE_RE.search(s)
    if m:
        api_url = _UCI_API_BY_ID.format(id=m.group(1))
    elif s.lower().startswith("https://doi.org/10.24432/") or _UCI_DOI_RE.match(s):
        # Resolve the DOI to find the UCI id
        doi_url = s if s.startswith("http") else f"https://doi.org/{s}"
        p = probe_url(doi_url, method="HEAD", timeout=20)
        final = p.get("final_url") or ""
        m = _UCI_DATASET_PAGE_RE.search(final)
        if m:
            api_url = _UCI_API_BY_ID.format(id=m.group(1))
    if api_url is None:
        # Try by name
        encoded = urllib.parse.quote_plus(re.sub(r"\s+", " ", s).strip())
        api_url = _UCI_API_BY_NAME.format(name=encoded)

    body = _http_get_json(api_url, timeout=30)
    if not body:
        return None
    if isinstance(body, dict) and body.get("status") and body["status"] != 200:
        return None
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, dict):
        return None

    out: dict[str, Any] = {
        "uci_id": data.get("uci_id"),
        "canonical_name": data.get("name"),
        "data_url": data.get("data_url"),
        "repository_url": data.get("repository_url"),
        "num_instances": data.get("num_instances"),
        "num_features": data.get("num_features"),
        "license": None,
    }
    # Try to extract license
    addl = data.get("additional_info") or {}
    if isinstance(addl, dict):
        lic = addl.get("citation_requests/acknowledgements")
        if lic and isinstance(lic, str) and len(lic) < 200:
            out["license"] = lic.strip()[:120]

    # Resolve size: prefer data_url HEAD, fallback to canonical zip download.
    size = None
    if out["data_url"]:
        size = _http_size(out["data_url"])
    if size is None and out["uci_id"]:
        # Try canonical zip URL under /static/public/<id>/<slug>.zip. The slug
        # is the path tail of repository_url, which UCI guarantees matches the
        # static asset filename.
        candidates = []
        if out["repository_url"]:
            tail = out["repository_url"].rstrip("/").split("/")[-1]
            if tail:
                candidates.append(f"https://archive.ics.uci.edu/static/public/{out['uci_id']}/{tail}.zip")
        # Fallback: name-based slug (handles older datasets where repo URL slug differs).
        slug = (data.get("name") or "").lower().strip().replace(" ", "+")
        if slug:
            candidates.append(f"https://archive.ics.uci.edu/static/public/{out['uci_id']}/{slug}.zip")
        for candidate in candidates:
            cz_size = _http_size(candidate)
            if cz_size:
                size = cz_size
                out["data_url"] = candidate
                break
    out["size_bytes"] = size
    return out


# ---------------------------------------------------------------------------
# Zenodo resolver
# ---------------------------------------------------------------------------

_ZENODO_DOI_RE = re.compile(r"^10\.5281/zenodo\.(\d+)", re.IGNORECASE)


def resolve_zenodo(doi_or_url: str) -> Optional[dict[str, Any]]:
    """Resolve a Zenodo record by DOI or URL. Returns dict with record_id,
    files (list of {name, url, size_bytes}), license, total_size_bytes.
    """
    if not doi_or_url:
        return None
    s = doi_or_url.strip()
    record_id = None
    m = re.search(r"zenodo\.org/record/(\d+)", s) or re.search(r"zenodo\.org/records/(\d+)", s)
    if m:
        record_id = m.group(1)
    else:
        m = _ZENODO_DOI_RE.search(s)
        if m:
            record_id = m.group(1)
        elif s.lower().startswith("https://doi.org/10.5281/"):
            tail = s.split("zenodo.")[-1].rstrip("/")
            if tail.isdigit():
                record_id = tail
    if not record_id:
        return None
    api_url = f"https://zenodo.org/api/records/{record_id}"
    body = _http_get_json(api_url, timeout=30)
    if not body or not isinstance(body, dict):
        return None
    files = body.get("files") or []
    out_files = []
    total = 0
    for f in files:
        sz = f.get("size") or 0
        if isinstance(sz, int) and sz > 0:
            total += sz
        out_files.append({
            "name": f.get("key") or f.get("filename"),
            "url": (f.get("links") or {}).get("self"),
            "size_bytes": sz if isinstance(sz, int) and sz > 0 else None,
        })
    md = body.get("metadata") or {}
    return {
        "record_id": record_id,
        "files": out_files,
        "license": (md.get("license") or {}).get("id") if isinstance(md.get("license"), dict) else None,
        "total_size_bytes": total or None,
        "title": md.get("title"),
    }


# ---------------------------------------------------------------------------
# Manual action queue
# ---------------------------------------------------------------------------

def _load_queue() -> list[dict]:
    if not _QUEUE_PATH.exists():
        return []
    try:
        with open(_QUEUE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _save_queue(queue: list[dict]) -> None:
    _QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _QUEUE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2, ensure_ascii=False)
    tmp.replace(_QUEUE_PATH)


def enqueue_manual_action(paper_id: str, asset_name: str, classification: dict[str, Any],
                          *, container: str, paper_title: str = "") -> None:
    """Append (or update) a manual-action queue entry. Idempotent per
    (paper_id, asset_name, url)."""
    url = classification.get("url")
    if not url:
        return
    with _queue_lock:
        queue = _load_queue()
        # Dedup
        for item in queue:
            if (item.get("paper_id") == paper_id
                    and item.get("asset_name") == asset_name
                    and item.get("url") == url):
                item["last_seen"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                _save_queue(queue)
                return
        queue.append({
            "paper_id": paper_id,
            "paper_title": paper_title,
            "container": container,
            "asset_name": asset_name,
            "url": url,
            "access_pattern": classification.get("access_pattern"),
            "automation_status": classification.get("automation_status"),
            "auth_required": classification.get("auth_required"),
            "method": classification.get("method"),
            "notes": classification.get("notes"),
            "added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "resolved": False,
        })
        _save_queue(queue)


# ---------------------------------------------------------------------------
# Convenience: classify + maybe-resolve in one shot
# ---------------------------------------------------------------------------

def classify_and_resolve(url: str) -> dict[str, Any]:
    """Classify + perform automatable resolution in one call.

    Returns the classification dict, plus 'resolved' key with extra info
    when we could actually fetch metadata (size, canonical URL, etc.).
    """
    cls = classify_url(url)
    ap = cls.get("access_pattern")
    if ap == "uci_repository":
        info = resolve_uci(url)
        if info:
            cls["resolved"] = info
    elif ap == "zenodo":
        info = resolve_zenodo(url)
        if info:
            cls["resolved"] = info
    return cls
