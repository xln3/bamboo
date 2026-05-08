#!/usr/bin/env python3
"""Extract per-paper required-download manifests via an LLM.

For each paper in bamboo_final.json, read its MinerU-extracted markdown
(or PDF as fallback), fetch the repo README at the pinned commit, send
both to an LLM with a strict-JSON schema, validate, enrich asset sizes
via asset_kb (curated KB > HuggingFace API > LLM estimate), and write
the result to:
  - data/paper_assets/{paper_id}.json   (full per-paper detail, HF-synced)
  - bamboo_final.json[i].required_assets   (inline summary)

Usage:
    python extract_assets.py --paper bamboo-00001 --model deep-deepseek-v4-pro \\
        --api-base https://your-endpoint/v1
    python extract_assets.py --limit 100
    python extract_assets.py --retry-failed

Environment:
    OPENAI_API_KEY  - required for LLM API auth
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import access_patterns
import asset_kb

log = logging.getLogger("extract_assets")

REPO_ROOT = Path(__file__).parent.parent.parent
FINAL_JSON = REPO_ROOT / "data" / "bamboo_final.json"
PAPER_MARKDOWNS = REPO_ROOT / "data" / "paper_markdowns"
PAPER_PDFS = REPO_ROOT / "data" / "paper_pdfs"
PAPER_ASSETS = REPO_ROOT / "data" / "paper_assets"
README_CACHE = REPO_ROOT / "data" / ".readme_cache"

VALID_OTHER_TYPES = {"library", "precomputed_features", "binary", "docker_image", "other"}
VALID_SIZE_SOURCES = {"kb", "hf_api", "hf_search", "extracted", "estimated", "unknown"}
VALID_AVAILABILITY = {"public", "restricted", "paid", "private", "promised", "bundled", "unknown"}
MANUAL_AUTOMATION_STATUSES = {"manual_browser", "manual_only", "unreachable"}

MAX_PAPER_CHARS = 400_000
MAX_README_CHARS = 100_000
MAX_OUTPUT_TOKENS = 131_072


SYSTEM_PROMPT = """You are an ML reproducibility assistant. Given a paper's text and the README of its code repository, you list every heavyweight resource a reader must DOWNLOAD SEPARATELY to reproduce the paper's main experiments.

For each resource, return:
- name: canonical name (e.g., "ImageNet-1K (ILSVRC2012)", "openai/clip-vit-large-patch14", "nuScenes")
- category: one of "dataset", "model_weights", "other"
- type: when category=="other", one of "library", "precomputed_features", "binary", "docker_image", "other"; omit otherwise
- purpose: what it is used for in this paper (≤ 80 chars)
- huggingface_id: HF repo id when known/inferable (e.g., "meta-llama/Llama-2-7b-hf"); omit otherwise
- url: canonical download URL when explicitly given (paper or README). If the only mention is a SharePoint/OneDrive/Baidu Pan/Google Drive link, INCLUDE IT — the pipeline classifies it downstream.
- size_bytes: integer best estimate, or null if you don't know. If the paper or README states a size (e.g., "3.5 GB"), convert and use that.
- size_source: "extracted" if the size came from the paper/README text; "estimated" if you guessed; omit if size_bytes is null.
- license: short license note (only if explicitly stated, ≤ 60 chars)
- required_for_claims: array of claim ids (c1, c2, ...) this resource is needed for. ONLY include claims where this exact asset is the bottleneck. Empty array is fine and often correct.
- notes: brief free-form note (≤ 100 chars total) — e.g., "registration required", "request access", "specific split only", "extraction code: abcd". Skip the field entirely if you have nothing concrete to say.
- availability: REQUIRED. One of:
    "public"      -- anyone with the URL can download (HuggingFace public repo, GitHub release, public Drive share, Zenodo open record, public UCI dataset).
    "restricted"  -- the dataset/model is publicly known but needs a free request/registration (ImageNet, gated HF licenses like LLaMA-2, datasets that require academic affiliation or DUA).
    "paid"        -- requires payment (commercial datasets, paid models).
    "private"     -- not released publicly; only the authors have it (paper says "data not yet available").
    "promised"    -- paper announces a forthcoming release ("will be released", "to be made public", "code & data coming soon").
    "bundled"     -- ships inside the repo itself (skip this and do NOT list).
    "unknown"     -- truly cannot tell from paper or README (use sparingly; you should usually be able to tell).
  When the paper provides a SharePoint/OneDrive/Drive/Baidu Pan link, that is "public" (the link works for anyone) -- access friction is captured by the pipeline's URL classification, NOT by you.

BREVITY:
- Keep purpose ≤ 60 chars, notes ≤ 100 chars. The schema is rejected if any string exceeds these.
- Output valid JSON only — no trailing commas, no comments, no leading text. Total output must fit in 8000 tokens; if you have many assets, prefer short notes over many notes.

DO INCLUDE:
- Datasets the paper trains or evaluates on
- Pretrained model checkpoints loaded by the code (CLIP, LLaMA, BERT, SAM, Stable Diffusion, etc.)
- Other large downloads > ~100 MB that are not bundled with the repo: precomputed features, docker images, binary releases.

DO NOT INCLUDE:
- Standard Python packages (pip-installed deps), CUDA, OS packages
- Small config files, tokenizer vocabs < 50 MB, demo images bundled in the repo
- Code dependencies that ship in the repo
- Hyperparameters, model architectures, dataset statistics

Output a single JSON object on its own, no markdown, no prose, no code fences:
{"datasets": [...], "model_weights": [...], "other_assets": [...]}

If a resource is not used in this paper, do not invent it. If you are uncertain whether a resource is needed, omit it. Prefer precision over recall. Empty arrays are fine."""


USER_PROMPT_TEMPLATE = """Paper title: {title}
Paper venue: {venue}
Code repository: {code_url}{claim_block}

=== PAPER TEXT ===
{paper_text}

=== README (pinned commit {commit_short}) ===
{readme_text}

Return the JSON object now."""


# ---------------------------------------------------------------------------
# Text gathering
# ---------------------------------------------------------------------------

def get_paper_text(paper_id: str, paper: dict) -> str:
    """Read the MinerU markdown for a paper, falling back to PDF extraction."""
    md_path = PAPER_MARKDOWNS / f"{paper_id}.md"
    if md_path.exists():
        try:
            text = md_path.read_text(encoding="utf-8", errors="replace")
            if len(text) >= 500:
                return text
        except Exception as e:
            log.warning(f"  failed to read {md_path}: {e}")

    pdf_path = PAPER_PDFS / f"{paper_id}.pdf"
    if pdf_path.exists():
        try:
            from pdf_extractor import extract_text_mineru
            text = extract_text_mineru(str(pdf_path))
            if text and len(text) >= 500:
                return text
        except Exception as e:
            log.warning(f"  MinerU fallback failed for {paper_id}: {e}")
        try:
            r = subprocess.run(
                ["pdftotext", "-layout", str(pdf_path), "-"],
                capture_output=True, text=True, timeout=60,
            )
            if r.returncode == 0 and len(r.stdout) >= 500:
                return r.stdout
        except Exception as e:
            log.warning(f"  pdftotext fallback failed for {paper_id}: {e}")

    return ""


_GH_REPO_RE = re.compile(r"^https?://(?:www\.)?github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", re.IGNORECASE)


def fetch_readme(code_url: str, commit: str, paper_id: str) -> str:
    """Fetch README at a pinned commit, cached on disk per paper.

    Only GitHub is supported; other hosts return ''.
    """
    cache = README_CACHE / f"{paper_id}.md"
    if cache.exists():
        try:
            return cache.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass

    README_CACHE.mkdir(parents=True, exist_ok=True)

    if not code_url or not commit:
        cache.write_text("", encoding="utf-8")
        return ""
    m = _GH_REPO_RE.match(code_url.strip())
    if not m:
        cache.write_text("", encoding="utf-8")
        return ""
    owner, repo = m.group(1), m.group(2)
    sha = commit.strip()
    candidates = ["README.md", "README.MD", "readme.md", "Readme.md", "README.rst", "README.txt", "README"]
    for fname in candidates:
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{fname}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "bamboo-extract-assets/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status == 200:
                    body = resp.read().decode("utf-8", errors="replace")
                    cache.write_text(body, encoding="utf-8")
                    return body
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            log.debug(f"  README fetch HTTP {e.code} for {url}")
            continue
        except Exception as e:
            log.debug(f"  README fetch error for {url}: {e}")
            continue

    cache.write_text("", encoding="utf-8")
    return ""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(paper: dict, paper_text: str, readme_text: str,
             model: str, api_base: str, api_key: str) -> Optional[dict]:
    """Send paper + README to the LLM, return parsed JSON dict or None."""
    if len(paper_text) > MAX_PAPER_CHARS:
        paper_text = paper_text[:MAX_PAPER_CHARS]
    if len(readme_text) > MAX_README_CHARS:
        readme_text = readme_text[:MAX_README_CHARS]

    # Two shapes seen in bamboo_final.json:
    #   (a) ground_truth_claims = [ {claim_id, description, ...}, ... ]
    #   (b) ground_truth_claims = {"paper_id": ..., "paper_title": ..., "claims": [...]}
    raw_claims = paper.get("ground_truth_claims")
    if isinstance(raw_claims, dict):
        claims = raw_claims.get("claims") or []
    elif isinstance(raw_claims, list):
        claims = raw_claims
    else:
        claims = []
    if claims:
        claim_lines = [f"  - {c.get('claim_id', '?')}: {c.get('description', '')[:80]}" for c in claims[:30]]
        claim_block = "\nClaims (use claim_id values for required_for_claims):\n" + "\n".join(claim_lines)
        if len(claims) > 30:
            claim_block += f"\n  ... ({len(claims) - 30} more)"
    else:
        claim_block = ""

    commit = paper.get("code_commit", "")
    user_content = USER_PROMPT_TEMPLATE.format(
        title=paper.get("title", "Unknown"),
        venue=paper.get("venue", ""),
        code_url=paper.get("code_url", ""),
        commit_short=commit[:8] if commit else "?",
        paper_text=paper_text or "(no paper text available)",
        readme_text=readme_text or "(README not available)",
        claim_block=claim_block,
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }

    url = f"{api_base.rstrip('/')}/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    body = None
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=360) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            # 429 / 5xx are retryable; 4xx (other) usually not
            retryable = (e.code == 429 or 500 <= e.code < 600)
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")[:500]
            except Exception:
                pass
            log.error(f"  LLM API HTTP {e.code} ({e.reason}) attempt {attempt}/{max_attempts}: {err_body}")
            if not retryable or attempt == max_attempts:
                return None
        except urllib.error.URLError as e:
            log.error(f"  LLM API connection error: {e.reason} attempt {attempt}/{max_attempts}")
            if attempt == max_attempts:
                return None
        except Exception as e:
            log.error(f"  LLM API error: {e} attempt {attempt}/{max_attempts}")
            if attempt == max_attempts:
                return None
        # Exponential backoff with jitter: 4s, 8s, 16s, 32s
        backoff = min(2 ** (attempt + 1), 60) + random.uniform(0, 2)
        log.info(f"  retry in {backoff:.1f}s")
        time.sleep(backoff)

    if body is None:
        return None

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        log.error("  Unexpected LLM response structure")
        return None

    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        log.warning(f"  JSON parse failed ({e}); attempting salvage")
        salvaged = _salvage_truncated_json(content)
        if salvaged is not None:
            return salvaged
        log.error("  JSON salvage also failed")
        log.debug(f"  Raw content (first 500 chars): {content[:500]}")
        return None


def _salvage_truncated_json(content: str) -> Optional[dict]:
    """Recover a partial result when the LLM response was cut off mid-string.

    Strategy: walk the brace/bracket stack and find the last position where
    every opened structure is closed and we are NOT inside a string. Trim
    after that, drop any dangling trailing comma, and re-parse.
    """
    last_balanced_pos = -1
    in_string = False
    escape = False
    stack: list[str] = []
    for i, ch in enumerate(content):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            if not stack:
                last_balanced_pos = i
    if last_balanced_pos < 0:
        return None
    trimmed = content[: last_balanced_pos + 1]
    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        pass
    # Fallback: try to recover an outer object by clipping at a clean array
    # boundary and closing it manually. Find last position where we exit a
    # top-level array element cleanly (']' followed by '\n' or '}' or end).
    trimmed = re.sub(r",\s*([\]}])", r"\1", trimmed)
    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Validation + enrichment
# ---------------------------------------------------------------------------

def _coerce_int(v) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if v >= 0 else None
    if isinstance(v, float):
        return int(v) if v >= 0 else None
    return None


def _validate_asset(raw: Any, container: str) -> Optional[dict]:
    """Validate a raw LLM asset dict; return cleaned, un-enriched copy or None.

    container: "datasets" | "model_weights" | "other_assets"
    """
    if not isinstance(raw, dict):
        return None
    name = (raw.get("name") or "").strip()
    if not name:
        return None

    clean: dict[str, Any] = {"name": name}

    if container == "other_assets":
        t = (raw.get("type") or "other").strip().lower()
        clean["type"] = t if t in VALID_OTHER_TYPES else "other"

    purpose = raw.get("purpose")
    if isinstance(purpose, str) and purpose.strip():
        clean["purpose"] = purpose.strip()[:120]

    hf = raw.get("huggingface_id")
    if isinstance(hf, str) and "/" in hf and len(hf) <= 200:
        clean["huggingface_id"] = hf.strip()

    url = raw.get("url")
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        clean["url"] = url.strip()

    size_bytes = _coerce_int(raw.get("size_bytes"))
    if size_bytes is not None:
        clean["size_bytes"] = size_bytes

    src = raw.get("size_source")
    if isinstance(src, str) and src in VALID_SIZE_SOURCES:
        clean["size_source"] = src

    lic = raw.get("license")
    if isinstance(lic, str) and lic.strip():
        clean["license"] = lic.strip()[:80]

    rfc = raw.get("required_for_claims")
    if isinstance(rfc, list):
        valid = [c for c in rfc if isinstance(c, str) and re.match(r"^c\d+$", c)]
        if valid:
            clean["required_for_claims"] = valid

    notes = raw.get("notes")
    if isinstance(notes, str) and notes.strip():
        clean["notes"] = notes.strip()[:240]

    avail = raw.get("availability")
    if isinstance(avail, str) and avail.strip().lower() in VALID_AVAILABILITY:
        clean["availability"] = avail.strip().lower()

    return clean


def apply_url_classification(asset: dict, *, paper_id: str, paper_title: str,
                              container: str) -> dict:
    """For an enriched asset, classify its URL via access_patterns and run any
    automatable resolver (UCI / Zenodo). Mutates and returns the asset dict.

    Sets: access_pattern, automation_status, auth_required, manual_action_required.
    May overwrite size_bytes/size_source/url when a resolver yields better data.
    Enqueues a manual_action_queue.json entry for non-automatable assets.
    """
    url = asset.get("url")
    huggingface_id = asset.get("huggingface_id")

    # Prefer HuggingFace id when both are present — HF mirrors are automatable
    # while paper-side URLs are often unrecognized homepages. Keep the original
    # url field intact so users can still see/reach it.
    if huggingface_id:
        path = "datasets/" if container == "datasets" else ""
        classify_target = f"https://huggingface.co/{path}{huggingface_id}"
    else:
        classify_target = url

    if not classify_target:
        # No URL at all. Build a synthetic classification so the rest of the
        # function (manual flag + queue) handles it uniformly.
        cls = {
            "access_pattern": "unknown",
            "automation_status": "unknown",
            "auth_required": "unknown",
            "method": None,
            "notes": "no URL provided in paper/README and no HF id — needs manual lookup",
            "url": None,
        }
    else:
        cls = access_patterns.classify_and_resolve(classify_target)
    asset["access_pattern"] = cls.get("access_pattern") or "unknown"
    asset["automation_status"] = cls.get("automation_status") or "unknown"
    asset["auth_required"] = cls.get("auth_required") or "unknown"

    # Resolver-supplied data
    resolved = cls.get("resolved") or {}
    if cls.get("access_pattern") == "uci_repository" and resolved:
        if resolved.get("data_url") and not asset.get("url"):
            asset["url"] = resolved["data_url"]
        elif resolved.get("repository_url") and not asset.get("url"):
            asset["url"] = resolved["repository_url"]
        if resolved.get("size_bytes") and (not asset.get("size_bytes") or asset.get("size_source") in (None, "unknown", "estimated")):
            asset["size_bytes"] = int(resolved["size_bytes"])
            asset["size_source"] = "extracted"  # came from upstream metadata, not LLM guess
            asset["size_human"] = asset_kb.human_size(asset["size_bytes"])
        if resolved.get("license") and not asset.get("license"):
            asset["license"] = resolved["license"][:80]
    elif cls.get("access_pattern") == "zenodo" and resolved:
        if resolved.get("total_size_bytes") and (not asset.get("size_bytes") or asset.get("size_source") in (None, "unknown", "estimated")):
            asset["size_bytes"] = int(resolved["total_size_bytes"])
            asset["size_source"] = "extracted"
            asset["size_human"] = asset_kb.human_size(asset["size_bytes"])
        if resolved.get("license") and not asset.get("license"):
            asset["license"] = resolved["license"][:80]

    # Manual-action flag + queue. An asset is a manual blocker when EITHER:
    #   - the URL pattern is non-automatable (SharePoint, Baidu, dead links),
    #   - the auth is human (form, browser, extraction code),
    #   - the LLM marked availability private / promised (paper not released),
    #   - access_pattern is unknown — even with a URL, an unregistered host means
    #     we have no automation path; surface to queue so a human can refine the
    #     registry or download manually ("在线但小众" is not an excuse).
    auto = asset["automation_status"]
    auth = asset["auth_required"]
    avail = asset.get("availability") or "unknown"
    ap = asset["access_pattern"]
    no_url = not asset.get("url")
    is_manual = (
        auto in MANUAL_AUTOMATION_STATUSES
        or auth in ("human_request", "extraction_code", "browser_session")
        or avail in ("private", "promised")
        or auto == "unknown"
        or ap == "unknown"
    )
    asset["manual_action_required"] = bool(is_manual)

    if is_manual:
        # Augment classification with explanatory reason for the queue entry.
        reason = None
        if avail in ("private", "promised"):
            reason = f"availability={avail}: not directly downloadable"
        elif no_url and ap == "unknown":
            # Synthetic cls already says "no URL provided..." — don't duplicate.
            reason = None
        elif ap == "unknown":
            reason = f"unrecognized host — needs registry entry or manual download"
        cls_for_queue = {**cls, "url": classify_target or asset.get("url") or "(no URL)"}
        if reason:
            cls_for_queue["notes"] = (cls_for_queue.get("notes") or "") + f" | {reason}"
        access_patterns.enqueue_manual_action(
            paper_id=paper_id,
            asset_name=asset.get("name", "?"),
            classification=cls_for_queue,
            container=container,
            paper_title=paper_title,
        )

    return asset


def build_required_assets(parsed: dict, *, model: str,
                          paper_id: str = "?", paper_title: str = "") -> dict:
    """Validate, enrich, and assemble the final required_assets dict."""
    datasets_raw = parsed.get("datasets") if isinstance(parsed.get("datasets"), list) else []
    models_raw = parsed.get("model_weights") if isinstance(parsed.get("model_weights"), list) else []
    others_raw = parsed.get("other_assets") if isinstance(parsed.get("other_assets"), list) else []

    datasets = []
    for r in datasets_raw:
        v = _validate_asset(r, "datasets")
        if v is not None:
            v = asset_kb.enrich(v, container="datasets")
            v = apply_url_classification(v, paper_id=paper_id, paper_title=paper_title, container="datasets")
            datasets.append(v)

    models = []
    for r in models_raw:
        v = _validate_asset(r, "model_weights")
        if v is not None:
            v = asset_kb.enrich(v, container="model_weights")
            v = apply_url_classification(v, paper_id=paper_id, paper_title=paper_title, container="model_weights")
            models.append(v)

    others = []
    for r in others_raw:
        v = _validate_asset(r, "other_assets")
        if v is not None:
            # 'other' assets get sizes only via the LLM (no KB or HF lookup).
            sb = v.get("size_bytes")
            if isinstance(sb, int) and sb > 0:
                v.setdefault("size_source", "estimated")
            else:
                v["size_source"] = "unknown"
                v["size_bytes"] = None
            v["size_human"] = asset_kb.human_size(v.get("size_bytes"))
            v = apply_url_classification(v, paper_id=paper_id, paper_title=paper_title, container="other_assets")
            others.append(v)

    total = sum(
        a["size_bytes"] for a in datasets + models + others
        if isinstance(a.get("size_bytes"), int) and a["size_bytes"] > 0
    )

    all_assets = datasets + models + others
    manual_blockers = sum(1 for a in all_assets if a.get("manual_action_required"))
    fully_automatable = bool(all_assets) and all(
        a.get("automation_status") in ("automatable", "automatable_with_auth")
        for a in all_assets
    )

    return {
        "datasets": datasets,
        "model_weights": models,
        "other_assets": others,
        "total_download_size_bytes": total or None,
        "total_download_size_human": asset_kb.human_size(total or None),
        "extraction_method": "llm",
        "extraction_model": model,
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fully_automatable": fully_automatable,
        "manual_blockers": manual_blockers,
    }


# ---------------------------------------------------------------------------
# Per-paper pipeline
# ---------------------------------------------------------------------------

def process_paper(paper: dict, model: str, api_base: str, api_key: str,
                  *, fetch_readme_flag: bool = True) -> Optional[dict]:
    """Extract required_assets for one paper. Returns the dict or None on failure."""
    paper_id = paper.get("paper_id", "?")
    title = paper.get("title", "Unknown")

    paper_text = get_paper_text(paper_id, paper)
    if not paper_text:
        log.warning("  no paper text available")
        return None

    readme_text = ""
    if fetch_readme_flag:
        readme_text = fetch_readme(
            paper.get("code_url", ""),
            paper.get("code_commit", ""),
            paper_id,
        )

    log.info(f"  paper={len(paper_text)} chars, readme={len(readme_text)} chars")

    parsed = call_llm(paper, paper_text, readme_text, model, api_base, api_key)
    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        log.error("  LLM returned non-dict")
        return None

    return build_required_assets(parsed, model=model, paper_id=paper_id, paper_title=title)


def write_paper_assets_file(paper_id: str, assets: dict) -> None:
    """Write data/paper_assets/{paper_id}.json (the source-of-truth detail file)."""
    PAPER_ASSETS.mkdir(parents=True, exist_ok=True)
    path = PAPER_ASSETS / f"{paper_id}.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"paper_id": paper_id, **assets}, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def inline_summary(assets: dict) -> dict:
    """Compact form stored on bamboo_final.json[i].required_assets."""
    return {
        "datasets": assets["datasets"],
        "model_weights": assets["model_weights"],
        "other_assets": assets["other_assets"],
        "total_download_size_bytes": assets["total_download_size_bytes"],
        "total_download_size_human": assets["total_download_size_human"],
        "extraction_method": assets["extraction_method"],
        "extraction_model": assets["extraction_model"],
        "extracted_at": assets["extracted_at"],
        "fully_automatable": assets.get("fully_automatable"),
        "manual_blockers": assets.get("manual_blockers"),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def process_final_json(*, model: str, api_base: str, api_key: str,
                       limit: Optional[int] = None,
                       retry_failed: bool = False,
                       only_paper: Optional[str] = None,
                       fetch_readme_flag: bool = True,
                       workers: int = 1,
                       slice_range: Optional[tuple[int, int]] = None) -> None:
    if not FINAL_JSON.exists():
        log.error(f"bamboo_final.json not found at {FINAL_JSON}")
        sys.exit(1)

    with open(FINAL_JSON, encoding="utf-8") as f:
        papers = json.load(f)

    to_process: list[tuple[int, dict]] = []
    for i, p in enumerate(papers):
        if only_paper:
            if p.get("paper_id") == only_paper:
                to_process.append((i, p))
            continue
        if retry_failed:
            # In retry mode, only process papers with a previous error.
            if p.get("_assets_error"):
                to_process.append((i, p))
            continue
        # Default: skip already-done and already-failed papers.
        if p.get("required_assets") or p.get("_assets_error"):
            continue
        to_process.append((i, p))

    # Apply explicit index slice (used to shard parallel runs).
    if slice_range is not None:
        s, e = slice_range
        to_process = [(i, p) for (i, p) in to_process if s <= i < e]

    if limit is not None:
        to_process = to_process[:limit]

    if retry_failed:
        for i, _ in to_process:
            papers[i].pop("_assets_error", None)

    n_with = sum(1 for p in papers if p.get("required_assets"))
    log.info(f"Total: {len(papers)} papers, {n_with} already have assets, "
             f"processing {len(to_process)} (workers={workers})")
    if not to_process:
        return

    success = 0
    failed = 0
    final_lock = threading.Lock()
    completed = [0]  # counter shared across threads

    def handle_one(idx: int, i: int, paper: dict) -> tuple[int, int, bool]:
        """Run process_paper for one paper. Returns (i, idx, success)."""
        paper_id = paper.get("paper_id", "?")
        title = paper.get("title", "Unknown")
        log.info(f"[{idx + 1}/{len(to_process)}] {paper_id}: {title[:70]}")
        try:
            assets = process_paper(paper, model, api_base, api_key,
                                   fetch_readme_flag=fetch_readme_flag)
            if assets is None:
                with final_lock:
                    papers[i]["_assets_error"] = "extraction_failed"
                return (i, idx, False)
            n_ds = len(assets["datasets"])
            n_mw = len(assets["model_weights"])
            n_ot = len(assets["other_assets"])
            total = assets["total_download_size_human"]
            log.info(f"  [{paper_id}] -> {n_ds} datasets, {n_mw} model_weights, "
                     f"{n_ot} other; total {total}")
            write_paper_assets_file(paper_id, assets)
            with final_lock:
                papers[i]["required_assets"] = inline_summary(assets)
                papers[i].pop("_assets_error", None)
            return (i, idx, True)
        except Exception as e:
            log.error(f"  [{paper_id}] Unexpected error: {e}")
            with final_lock:
                papers[i]["_assets_error"] = f"exception: {str(e)[:100]}"
            return (i, idx, False)

    if workers <= 1:
        # Serial path (original behavior, plus per-paper sleep for rate-limiting).
        for idx, (i, paper) in enumerate(to_process):
            _, _, ok = handle_one(idx, i, paper)
            if ok:
                success += 1
            else:
                failed += 1
            time.sleep(1.0)
            completed[0] += 1
            if completed[0] % 10 == 0:
                with final_lock:
                    with open(FINAL_JSON, "w", encoding="utf-8") as f:
                        json.dump(papers, f, indent=2, ensure_ascii=False)
                log.info(f"  Progress saved. {success} success, {failed} failed "
                         f"({completed[0]}/{len(to_process)})")
    else:
        # Parallel path: each worker thread issues an LLM call concurrently.
        # The LLM HTTP call is the latency bottleneck and releases the GIL.
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(handle_one, idx, i, p)
                       for idx, (i, p) in enumerate(to_process)]
            for fut in concurrent.futures.as_completed(futures):
                _, _, ok = fut.result()
                if ok:
                    success += 1
                else:
                    failed += 1
                completed[0] += 1
                if completed[0] % 10 == 0:
                    with final_lock:
                        with open(FINAL_JSON, "w", encoding="utf-8") as f:
                            json.dump(papers, f, indent=2, ensure_ascii=False)
                    log.info(f"  Progress saved. {success} success, {failed} failed "
                             f"({completed[0]}/{len(to_process)})")

    with final_lock:
        with open(FINAL_JSON, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
    log.info(f"Done: +{success} extracted, {failed} failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-paper required-download manifests.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max papers to process this run.")
    parser.add_argument("--model", type=str, default="deep-deepseek-v4-pro",
                        help="LLM model name.")
    parser.add_argument("--api-base", type=str,
                        default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
                        help="OpenAI-compatible API base URL.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-process papers with _assets_error set.")
    parser.add_argument("--paper", type=str, default=None,
                        help="Process only this single paper_id.")
    parser.add_argument("--no-readme", action="store_true",
                        help="Skip fetching the README (debug).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of concurrent worker threads (default 1). "
                             "Each thread issues one LLM call at a time; the LLM "
                             "HTTP I/O releases the GIL so 4-8 is usually safe.")
    parser.add_argument("--slice", type=str, default=None,
                        help="Process only paper indices in [start, end). "
                             "E.g. --slice 0:1500 for the first quarter. "
                             "Use to shard work across multiple processes.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    slice_range = None
    if args.slice:
        try:
            s, e = (int(x) for x in args.slice.split(":"))
            slice_range = (s, e)
        except ValueError:
            print(f"--slice must be START:END (got {args.slice!r})", file=sys.stderr)
            sys.exit(2)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        log.error("OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    process_final_json(
        model=args.model,
        api_base=args.api_base,
        api_key=api_key,
        limit=args.limit,
        retry_failed=args.retry_failed,
        only_paper=args.paper,
        fetch_readme_flag=not args.no_readme,
        workers=max(1, args.workers),
        slice_range=slice_range,
    )


if __name__ == "__main__":
    main()
