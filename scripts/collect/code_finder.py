"""Find code repository URLs for papers using multiple sources."""

from __future__ import annotations

import logging
import os
import re
import ssl
import time
from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context


class _SSLAdapter(HTTPAdapter):
    """HTTPS adapter with relaxed SSL for Python 3.8 compatibility."""
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.set_ciphers("DEFAULT@SECLEVEL=1")
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


# Shared session with SSL workaround
_session = requests.Session()
_session.mount("https://", _SSLAdapter())
_session.verify = False

# Suppress InsecureRequestWarning
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from config import (
    SEMANTIC_SCHOLAR_API,
    SEMANTIC_SCHOLAR_API_KEY,
    HUGGINGFACE_PAPERS_API,
    RATE_LIMIT,
)

log = logging.getLogger(__name__)

GITHUB_URL_RE = re.compile(
    r"https?://github\.com/[\w\-\.]+/[\w\-\.]+(?:/(?:tree|blob)/[\w\-\.]+)?",
    re.IGNORECASE,
)
GITLAB_URL_RE = re.compile(
    r"https?://gitlab\.com/[\w\-\.]+/[\w\-\.]+",
    re.IGNORECASE,
)
HUGGINGFACE_REPO_RE = re.compile(
    r"https?://huggingface\.co/[\w\-\.]+/[\w\-\.]+",
    re.IGNORECASE,
)

CODE_URL_PATTERNS = [GITHUB_URL_RE, GITLAB_URL_RE, HUGGINGFACE_REPO_RE]


def _normalize_github_url(url: str) -> str:
    """Normalize GitHub URL to repo root (remove /tree/branch etc.)."""
    m = re.match(r"(https?://github\.com/[\w\-\.]+/[\w\-\.]+)", url)
    return m.group(1) if m else url


def find_code_urls(paper: dict) -> list[str]:
    """Find code repository URLs for a paper using multiple strategies.

    Returns a list of unique code URLs found, most confident first.
    """
    urls: list[str] = []

    # 1. Already has a code URL (from venue metadata)
    if paper.get("code_url"):
        urls.append(paper["code_url"])

    # 2. Search abstract for code URLs
    abstract = paper.get("abstract", "")
    if abstract:
        for pattern in CODE_URL_PATTERNS:
            for m in pattern.finditer(abstract):
                urls.append(m.group(0))

    # 3. PapersWithCode API (by title, most reliable for finding code repos)
    title = paper.get("title", "")
    if title and not urls:
        pwc_urls = _search_paperswithcode(title)
        urls.extend(pwc_urls)

    # 4. Semantic Scholar API (may be rate-limited)
    arxiv_id = paper.get("arxiv_id", "")
    if not urls:
        if arxiv_id:
            ss_urls = _search_semantic_scholar(arxiv_id=arxiv_id)
            urls.extend(ss_urls)
        elif title:
            ss_urls = _search_semantic_scholar(title=title)
            urls.extend(ss_urls)

    # 5. HuggingFace Papers API
    if arxiv_id and not urls:
        hf_urls = _search_huggingface_papers(arxiv_id)
        urls.extend(hf_urls)

    # 6. arXiv abstract page (often has GitHub link)
    if arxiv_id and not urls:
        arxiv_urls = _search_arxiv_page(arxiv_id)
        urls.extend(arxiv_urls)

    # Normalize and deduplicate
    normalized = []
    seen = set()
    for url in urls:
        url = url.rstrip("/")
        if "github.com" in url:
            url = _normalize_github_url(url)
        if url not in seen:
            seen.add(url)
            normalized.append(url)

    return normalized


def _search_paperswithcode(title: str) -> list:
    """Search PapersWithCode API for code repositories."""
    urls = []
    try:
        time.sleep(RATE_LIMIT.get("semantic_scholar", 1.0))  # Reuse rate limit
        # Search for paper
        resp = _session.get(
            "https://paperswithcode.com/api/v1/papers/",
            params={"q": title},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results:
                # Find best title match
                title_lower = title.lower().strip()
                for r in results[:3]:
                    r_title = (r.get("title") or "").lower().strip()
                    # Fuzzy match: check if most words overlap
                    title_words = set(title_lower.split())
                    r_words = set(r_title.split())
                    if len(title_words & r_words) >= len(title_words) * 0.7:
                        paper_id = r.get("id")
                        if paper_id:
                            # Get repos for this paper
                            repo_resp = _session.get(
                                f"https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/",
                                timeout=15,
                            )
                            if repo_resp.status_code == 200:
                                repos = repo_resp.json().get("results", [])
                                for repo in repos:
                                    repo_url = repo.get("url")
                                    if repo_url:
                                        urls.append(repo_url)
                            break
        elif resp.status_code == 429:
            log.warning("PapersWithCode rate limited")
    except Exception as e:
        log.debug(f"PapersWithCode search failed: {e}")
    return urls


def _search_semantic_scholar(
    arxiv_id: str = "", title: str = ""
) -> list[str]:
    """Search Semantic Scholar for code URLs."""
    urls = []
    headers = {}
    api_key = SEMANTIC_SCHOLAR_API_KEY or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    try:
        if arxiv_id:
            endpoint = f"{SEMANTIC_SCHOLAR_API}/paper/ARXIV:{arxiv_id}"
        elif title:
            endpoint = f"{SEMANTIC_SCHOLAR_API}/paper/search"
        else:
            return []

        params = {"fields": "externalIds,openAccessPdf,url"}
        if title and not arxiv_id:
            params["query"] = title
            params["limit"] = 1

        time.sleep(RATE_LIMIT.get("semantic_scholar", 1.0))
        resp = _session.get(endpoint, params=params, headers=headers, timeout=15)

        if resp.status_code == 200:
            data = resp.json()
            # Handle search results vs direct lookup
            if "data" in data:
                data = data["data"][0] if data["data"] else {}

            # Check for GitHub external ID
            ext_ids = data.get("externalIds", {})
            github_url = ext_ids.get("GitHub")
            if github_url:
                if not github_url.startswith("http"):
                    github_url = f"https://github.com/{github_url}"
                urls.append(github_url)
        elif resp.status_code == 429:
            log.warning("Semantic Scholar rate limited, skipping")
        else:
            log.debug(f"Semantic Scholar returned {resp.status_code}")

    except Exception as e:
        log.debug(f"Semantic Scholar search failed: {e}")

    return urls


def _search_huggingface_papers(arxiv_id: str) -> list[str]:
    """Search HuggingFace Papers for code URLs."""
    urls = []
    try:
        time.sleep(RATE_LIMIT.get("huggingface", 0.5))
        resp = _session.get(
            f"{HUGGINGFACE_PAPERS_API}/{arxiv_id}",
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            # HuggingFace Papers may have linked repos
            for repo in data.get("repos", []):
                repo_url = repo.get("url", "")
                if repo_url:
                    urls.append(repo_url)
            # Also check for GitHub links in the paper metadata
            github_url = data.get("github", "")
            if github_url:
                urls.append(github_url)
    except Exception as e:
        log.debug(f"HuggingFace Papers search failed: {e}")
    return urls


def _search_arxiv_page(arxiv_id: str) -> list[str]:
    """Check arXiv abstract page for code links."""
    urls = []
    try:
        time.sleep(RATE_LIMIT.get("semantic_scholar", 1.0))
        resp = _session.get(
            f"https://arxiv.org/abs/{arxiv_id}",
            timeout=15,
            headers={"User-Agent": "BAMBOO/1.0 (research benchmark)"},
        )
        if resp.status_code == 200:
            # Search the page for GitHub/GitLab links
            for pattern in CODE_URL_PATTERNS:
                for m in pattern.finditer(resp.text):
                    url = m.group(0)
                    # Filter out common false positives
                    if not any(fp in url.lower() for fp in [
                        "github.com/arxiv",
                        "github.com/login",
                        "github.com/features",
                    ]):
                        urls.append(url)
    except Exception as e:
        log.debug(f"arXiv page fetch failed: {e}")
    return urls


def validate_code_url(url: str) -> dict:
    """Validate a code repository URL. Returns info dict or None if invalid."""
    if "github.com" in url:
        return _validate_github(url)
    # For other platforms, just check if URL is accessible
    try:
        resp = _session.head(url, timeout=10, allow_redirects=True)
        if resp.status_code < 400:
            return {"url": url, "platform": "other", "valid": True}
    except Exception:
        pass
    return {"url": url, "platform": "other", "valid": False}


def _validate_github(url: str) -> dict:
    """Validate a GitHub repository URL using the API."""
    # Extract owner/repo from URL
    m = re.match(r"https?://github\.com/([\w\-\.]+)/([\w\-\.]+)", url)
    if not m:
        return {"url": url, "platform": "github", "valid": False, "reason": "invalid URL format"}

    owner, repo = m.group(1), m.group(2)
    repo = repo.rstrip(".git")

    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Accept": "application/vnd.github.v3+json"}

    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    try:
        time.sleep(RATE_LIMIT.get("github", 1.0))
        resp = _session.get(api_url, headers=headers, timeout=15)

        if resp.status_code == 200:
            data = resp.json()
            return {
                "url": url,
                "platform": "github",
                "valid": True,
                "stars": data.get("stargazers_count", 0),
                "default_branch": data.get("default_branch", "main"),
                "language": data.get("language"),
                "size_kb": data.get("size", 0),
                "archived": data.get("archived", False),
                "updated_at": data.get("updated_at"),
            }
        elif resp.status_code == 404:
            return {"url": url, "platform": "github", "valid": False, "reason": "not found"}
        else:
            return {"url": url, "platform": "github", "valid": False, "reason": f"HTTP {resp.status_code}"}

    except Exception as e:
        return {"url": url, "platform": "github", "valid": False, "reason": str(e)}
