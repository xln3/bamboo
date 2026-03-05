"""Per-venue paper collection adapters."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from config import VenueConfig, RATE_LIMIT

log = logging.getLogger(__name__)


def _rate_limit(source: str):
    time.sleep(RATE_LIMIT.get(source, 1.0))


# Standard browser-like headers to avoid 403s
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _get(url: str, timeout: int = 30, **kwargs) -> requests.Response:
    """HTTP GET with standard headers."""
    headers = dict(_HEADERS)
    headers.update(kwargs.pop("headers", {}))
    return requests.get(url, headers=headers, timeout=timeout, **kwargs)


# ---------- Common paper dict format ----------
def make_paper(
    *,
    venue: str,
    year: int,
    title: str,
    authors: list[str] | None = None,
    abstract: str = "",
    arxiv_id: str = "",
    paper_url: str = "",
    pdf_url: str = "",
    code_url: str = "",
    venue_track: str = "main",
    forum_id: str = "",
) -> dict[str, Any]:
    return {
        "venue": venue,
        "year": year,
        "title": title,
        "authors": authors or [],
        "abstract": abstract,
        "arxiv_id": arxiv_id,
        "paper_url": paper_url,
        "pdf_url": pdf_url,
        "code_url": code_url,
        "venue_track": venue_track,
        "forum_id": forum_id,
    }


# ---------- OpenReview (ICLR, ICML, NeurIPS) ----------
def collect_openreview(cfg: VenueConfig) -> list[dict]:
    """Collect accepted papers from OpenReview venues."""
    try:
        import openreview
    except ImportError:
        log.error("openreview-py not installed. Run: pip install openreview-py")
        return []

    log.info(f"Collecting {cfg.name} {cfg.year} from OpenReview ({cfg.openreview_venue})")
    client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")

    papers = []
    try:
        venue_group = cfg.openreview_venue
        # Accepted venue patterns: "{Name} {Year} Oral/Poster/Spotlight"
        # E.g. "ICLR 2025 Oral", "ICLR 2025 Poster", "ICLR 2025 Spotlight"
        # For NeurIPS/ICML the pattern may differ slightly
        accepted_venue_ids = [
            f"{venue_group}/Oral",
            f"{venue_group}/Poster",
            f"{venue_group}/Spotlight",
        ]

        all_submissions = []
        for vid in accepted_venue_ids:
            try:
                subs = client.get_all_notes(
                    content={"venueid": vid},
                    invitation=f"{venue_group}/-/Submission",
                )
                log.info(f"  {vid}: {len(subs)} papers")
                all_submissions.extend(subs)
            except Exception as e:
                log.debug(f"  {vid}: no results ({e})")

        # Fallback: if no venueid-based results, get all and filter by venue text
        if not all_submissions:
            log.info("  Falling back to full submission scan...")
            all_notes = client.get_all_notes(
                invitation=f"{venue_group}/-/Submission",
            )
            accept_keywords = {"oral", "poster", "spotlight", "accept"}
            reject_keywords = {"reject", "withdrawn", "desk", "submitted to"}
            for note in all_notes:
                content = note.content or {}
                venue_value = _get_content_value(content, "venue") or ""
                venue_lower = venue_value.lower()
                if any(kw in venue_lower for kw in reject_keywords):
                    continue
                if any(kw in venue_lower for kw in accept_keywords):
                    all_submissions.append(note)

        log.info(f"Found {len(all_submissions)} accepted papers for {cfg.name}")

        for note in all_submissions:
            content = note.content or {}

            title = _get_content_value(content, "title") or ""
            abstract = _get_content_value(content, "abstract") or ""
            authors_value = _get_content_value(content, "authors")
            authors = authors_value if isinstance(authors_value, list) else []

            # Extract code URL from content (some venues have a 'code' field)
            code_url_raw = _get_content_value(content, "code") or ""
            # Validate it's actually a URL, not ethics text
            code_url = code_url_raw if code_url_raw.startswith("http") else ""

            # Extract arXiv ID from PDF URL
            arxiv_id = ""
            pdf_url = _get_content_value(content, "pdf") or ""
            if pdf_url and "arxiv.org" in pdf_url:
                arxiv_id = _extract_arxiv_id(pdf_url)

            # Determine track from venue string
            venue_value = _get_content_value(content, "venue") or ""
            venue_lower = venue_value.lower()
            track = "main"
            if "oral" in venue_lower:
                track = "oral"
            elif "spotlight" in venue_lower:
                track = "spotlight"
            elif "poster" in venue_lower:
                track = "poster"
            elif "workshop" in venue_lower:
                track = "workshop"

            paper = make_paper(
                venue=cfg.name,
                year=cfg.year,
                title=title,
                authors=authors,
                abstract=abstract,
                arxiv_id=arxiv_id,
                paper_url=f"https://openreview.net/forum?id={note.id}",
                pdf_url=pdf_url if pdf_url.startswith("http") else f"https://openreview.net{pdf_url}" if pdf_url else "",
                code_url=code_url,
                venue_track=track,
                forum_id=note.id,
            )
            papers.append(paper)

    except Exception as e:
        log.error(f"Error collecting from OpenReview for {cfg.name}: {e}")

    log.info(f"Collected {len(papers)} accepted papers from {cfg.name}")
    return papers


def _get_content_value(content: dict, key: str) -> Any:
    """Extract value from OpenReview content (handles both v1 and v2 formats)."""
    val = content.get(key)
    if val is None:
        return None
    if isinstance(val, dict):
        return val.get("value")
    return val


def _check_acceptance_from_replies(note, client, venue_group: str) -> str | None:
    """Check if a submission was accepted by looking at decision replies."""
    try:
        decisions = client.get_all_notes(
            invitation=f"{venue_group}/-/Decision",
            forum=note.id,
        )
        for d in decisions:
            decision_val = _get_content_value(d.content or {}, "decision")
            if decision_val and "accept" in str(decision_val).lower():
                return decision_val
    except Exception:
        pass
    return None


# ---------- CVF Open Access (CVPR, ICCV) ----------
def collect_cvf(cfg: VenueConfig) -> list[dict]:
    """Collect papers from CVF Open Access."""
    log.info(f"Collecting {cfg.name} {cfg.year} from CVF ({cfg.cvf_conference})")

    base_url = f"https://openaccess.thecvf.com/{cfg.cvf_conference}"
    papers = []

    try:
        # Get the main page which lists all papers or day-wise pages
        resp = _get(base_url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # CVF pages may have day-wise links or list all papers
        # Try to find paper list links (e.g., ?day=all or individual day pages)
        day_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "day=" in href or href.endswith(cfg.cvf_conference):
                full_url = href if href.startswith("http") else f"https://openaccess.thecvf.com/{href}"
                day_links.append(full_url)

        if not day_links:
            day_links = [f"{base_url}?day=all"]

        for page_url in day_links:
            _rate_limit("cvf")
            try:
                resp = _get(page_url)
                resp.raise_for_status()
                page_papers = _parse_cvf_page(resp.text, cfg)
                papers.extend(page_papers)
            except Exception as e:
                log.warning(f"Error fetching CVF page {page_url}: {e}")

    except Exception as e:
        log.error(f"Error collecting from CVF for {cfg.name}: {e}")

    # Deduplicate by title
    seen = set()
    unique = []
    for p in papers:
        key = p["title"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(p)

    log.info(f"Collected {len(unique)} papers from {cfg.name}")
    return unique


def _parse_cvf_page(html: str, cfg: VenueConfig) -> list[dict]:
    """Parse a CVF Open Access page for paper entries."""
    soup = BeautifulSoup(html, "html.parser")
    papers = []

    # CVF uses <dt> for titles and <dd> for details
    for dt in soup.find_all("dt", class_="ptitle"):
        title_tag = dt.find("a")
        if not title_tag:
            continue
        title = title_tag.get_text(strip=True)
        paper_link = title_tag.get("href", "")

        # Find the corresponding <dd> with authors and links
        dd = dt.find_next_sibling("dd")
        authors = []
        pdf_url = ""
        if dd:
            # Authors
            author_tags = dd.find_all("a", href=lambda h: h and "author" in str(h).lower()) if dd else []
            if not author_tags:
                # Try form: <div id="authors">Author1, Author2</div>
                authors_div = dd.find("div", id="authors")
                if authors_div:
                    authors = [a.strip() for a in authors_div.get_text().split(",")]
            else:
                authors = [a.get_text(strip=True) for a in author_tags]

            # PDF link
            for a in dd.find_all("a", href=True):
                href = a["href"]
                if href.endswith(".pdf"):
                    pdf_url = href if href.startswith("http") else f"https://openaccess.thecvf.com{href}"
                    break

        paper = make_paper(
            venue=cfg.name,
            year=cfg.year,
            title=title,
            authors=authors,
            paper_url=f"https://openaccess.thecvf.com/{paper_link}" if paper_link and not paper_link.startswith("http") else paper_link,
            pdf_url=pdf_url,
        )
        papers.append(paper)

    return papers


# ---------- ACL Anthology (ACL, EMNLP) ----------
def collect_acl(cfg: VenueConfig) -> list[dict]:
    """Collect papers from ACL Anthology."""
    log.info(f"Collecting {cfg.name} {cfg.year} from ACL Anthology ({cfg.acl_event})")

    papers = []
    base_url = f"https://aclanthology.org/events/{cfg.acl_event}/"

    try:
        resp = _get(base_url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # ACL Anthology lists papers under volumes
        # Find all paper entries
        for paper_span in soup.find_all("span", class_="d-block"):
            # Paper title is typically in a <strong> > <a>
            title_tag = paper_span.find("a", class_="align-middle")
            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)
            paper_link = title_tag.get("href", "")
            paper_url = paper_link if paper_link.startswith("http") else f"https://aclanthology.org{paper_link}"

            # Get paper ID for PDF URL
            paper_acl_id = paper_link.strip("/").split("/")[-1] if paper_link else ""
            pdf_url = f"https://aclanthology.org/{paper_acl_id}.pdf" if paper_acl_id else ""

            # Authors
            authors = []
            author_span = paper_span.find_next_sibling("span")
            if author_span:
                for a in author_span.find_all("a"):
                    authors.append(a.get_text(strip=True))

            # Determine track from the volume
            parent_section = paper_span.find_parent("section")
            track = "main"
            if parent_section:
                section_title = parent_section.find("h4") or parent_section.find("h3")
                if section_title:
                    section_text = section_title.get_text().lower()
                    if "findings" in section_text:
                        track = "findings"
                    elif "demo" in section_text:
                        track = "demo"
                    elif "workshop" in section_text:
                        track = "workshop"

            paper = make_paper(
                venue=cfg.name,
                year=cfg.year,
                title=title,
                authors=authors,
                paper_url=paper_url,
                pdf_url=pdf_url,
                venue_track=track,
            )
            papers.append(paper)

        _rate_limit("acl")

    except Exception as e:
        log.error(f"Error collecting from ACL Anthology for {cfg.name}: {e}")

    log.info(f"Collected {len(papers)} papers from {cfg.name}")
    return papers


# ---------- AAAI ----------
def collect_aaai(cfg: VenueConfig) -> list[dict]:
    """Collect papers from AAAI proceedings (ojs.aaai.org)."""
    log.info(f"Collecting {cfg.name} {cfg.year} from AAAI proceedings")

    papers = []
    # AAAI 2025 proceedings are at https://ojs.aaai.org/index.php/AAAI/issue/view/XXX
    # The exact issue number varies; we'll try to find it
    base_url = "https://ojs.aaai.org/index.php/AAAI/issue/archive"

    try:
        resp = _get(base_url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Find ALL AAAI-25 issue links (AAAI splits into multiple tracks)
        # Format: "AAAI-25 Technical Tracks 1", "AAAI-25 Technical Tracks 2", etc.
        year_short = str(cfg.year)[-2:]  # "25" for 2025
        issue_urls = []
        for a in soup.find_all("a", href=True):
            text = a.get_text().strip()
            href = a["href"]
            if f"aaai-{year_short}" in text.lower() and "/issue/view/" in href:
                full_url = href if href.startswith("http") else f"https://ojs.aaai.org{href}"
                if full_url not in issue_urls:
                    issue_urls.append(full_url)

        if not issue_urls:
            log.warning(f"Could not find AAAI {cfg.year} issue URLs")

        log.info(f"Found {len(issue_urls)} AAAI-{year_short} issue volumes")
        for issue_url in issue_urls:
            log.info(f"  Parsing {issue_url}")
            issue_papers = _parse_aaai_issue(issue_url, cfg)
            papers.extend(issue_papers)
            _rate_limit("aaai")

    except Exception as e:
        log.error(f"Error collecting from AAAI: {e}")

    log.info(f"Collected {len(papers)} papers from {cfg.name}")
    return papers


def _parse_aaai_issue(issue_url: str, cfg: VenueConfig) -> list[dict]:
    """Parse an AAAI issue page for papers."""
    papers = []
    try:
        resp = _get(issue_url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # OJS lists papers in article summaries
        for article in soup.find_all("div", class_="obj_article_summary"):
            title_tag = article.find("h3", class_="title")
            if not title_tag:
                continue
            a_tag = title_tag.find("a")
            if not a_tag:
                continue

            title = a_tag.get_text(strip=True)
            paper_url = a_tag.get("href", "")

            # Authors
            authors = []
            author_div = article.find("div", class_="authors")
            if author_div:
                authors = [a.strip() for a in author_div.get_text().split(",")]

            paper = make_paper(
                venue=cfg.name,
                year=cfg.year,
                title=title,
                authors=authors,
                paper_url=paper_url,
            )
            papers.append(paper)

        # Check for pagination
        next_link = soup.find("a", class_="next")
        if next_link and next_link.get("href"):
            _rate_limit("aaai")
            papers.extend(_parse_aaai_issue(next_link["href"], cfg))

    except Exception as e:
        log.warning(f"Error parsing AAAI issue page {issue_url}: {e}")

    return papers


# ---------- IEEE Xplore (ICRA) ----------
def collect_ieee(cfg: VenueConfig) -> list[dict]:
    """Collect papers from IEEE Xplore for ICRA.

    Note: IEEE Xplore has strict scraping policies. This collector uses
    the public search API. For full collection, an IEEE API key is recommended.
    """
    log.info(f"Collecting {cfg.name} {cfg.year} from IEEE Xplore")

    papers = []
    # IEEE Xplore API endpoint
    # Without API key, we use the public search
    search_url = "https://ieeexplore.ieee.org/rest/search"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; BAMBOO/1.0; research)",
    }

    # Search in batches
    page_size = 100
    start = 1
    total_collected = 0

    while True:
        payload = {
            "queryText": f"ICRA {cfg.year}",
            "highlight": False,
            "returnFacets": ["ALL"],
            "returnType": "SEARCH",
            "rowsPerPage": page_size,
            "pageNumber": (start - 1) // page_size + 1,
            "searchWithin": [f'"Publication Title":"ICRA"'],
        }

        try:
            resp = requests.post(search_url, json=payload, headers=headers, timeout=30)
            if resp.status_code != 200:
                log.warning(f"IEEE API returned {resp.status_code}, stopping collection")
                break

            data = resp.json()
            records = data.get("records", [])
            if not records:
                break

            for record in records:
                title = record.get("articleTitle", "")
                authors_list = record.get("authors", [])
                authors = [a.get("preferredName", a.get("normalizedName", "")) for a in authors_list]
                doi = record.get("doi", "")
                paper_url = f"https://doi.org/{doi}" if doi else ""

                paper = make_paper(
                    venue=cfg.name,
                    year=cfg.year,
                    title=title,
                    authors=authors,
                    paper_url=paper_url,
                )
                papers.append(paper)

            total_collected += len(records)
            total_results = data.get("totalRecords", 0)

            if total_collected >= total_results:
                break

            start += page_size
            _rate_limit("ieee")

        except Exception as e:
            log.warning(f"Error fetching from IEEE: {e}")
            break

    log.info(f"Collected {len(papers)} papers from {cfg.name}")
    return papers


# ---------- Dispatcher ----------
COLLECTORS = {
    "openreview": collect_openreview,
    "cvf": collect_cvf,
    "acl": collect_acl,
    "aaai": collect_aaai,
    "ieee": collect_ieee,
}


def collect_venue(cfg: VenueConfig) -> list[dict]:
    """Collect papers for a given venue configuration."""
    collector = COLLECTORS.get(cfg.source)
    if not collector:
        log.error(f"No collector for source: {cfg.source}")
        return []
    return collector(cfg)


# ---------- Utility ----------
def _extract_arxiv_id(text: str) -> str:
    """Extract arXiv ID from a URL or string."""
    patterns = [
        r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)",
        r"arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)",
        r"(\d{4}\.\d{4,5}(?:v\d+)?)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            # Remove version suffix for canonical ID
            aid = m.group(1)
            return re.sub(r"v\d+$", "", aid)
    return ""
