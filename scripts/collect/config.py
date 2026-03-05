"""BAMBOO paper collection configuration."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Output directory for collected data
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "papers"

# Per-venue configuration
@dataclass
class VenueConfig:
    venue_id: str
    name: str
    year: int
    area: str
    source: str  # openreview | cvf | acl | aaai | ieee
    # Source-specific config
    openreview_venue: str = ""           # e.g. "ICLR.cc/2025/Conference"
    cvf_conference: str = ""             # e.g. "CVPR2025"
    acl_event: str = ""                  # e.g. "acl-2025"
    proceedings_url: str = ""            # fallback URL


VENUES: List[VenueConfig] = [
    VenueConfig(
        venue_id="iclr2025",
        name="ICLR",
        year=2025,
        area="ml",
        source="openreview",
        openreview_venue="ICLR.cc/2025/Conference",
    ),
    VenueConfig(
        venue_id="icml2025",
        name="ICML",
        year=2025,
        area="ml",
        source="openreview",
        openreview_venue="ICML.cc/2025/Conference",
    ),
    VenueConfig(
        venue_id="neurips2025",
        name="NeurIPS",
        year=2025,
        area="ml",
        source="openreview",
        openreview_venue="NeurIPS.cc/2025/Conference",
    ),
    VenueConfig(
        venue_id="cvpr2025",
        name="CVPR",
        year=2025,
        area="vision",
        source="cvf",
        cvf_conference="CVPR2025",
    ),
    VenueConfig(
        venue_id="iccv2025",
        name="ICCV",
        year=2025,
        area="vision",
        source="cvf",
        cvf_conference="ICCV2025",
    ),
    VenueConfig(
        venue_id="acl2025",
        name="ACL",
        year=2025,
        area="nlp",
        source="acl",
        acl_event="acl-2025",
    ),
    VenueConfig(
        venue_id="emnlp2025",
        name="EMNLP",
        year=2025,
        area="nlp",
        source="acl",
        acl_event="emnlp-2025",
    ),
    VenueConfig(
        venue_id="aaai2025",
        name="AAAI",
        year=2025,
        area="ai",
        source="aaai",
        proceedings_url="https://ojs.aaai.org/index.php/AAAI/issue/archive",
    ),
    VenueConfig(
        venue_id="icra2025",
        name="ICRA",
        year=2025,
        area="robotics",
        source="ieee",
        proceedings_url="https://ieeexplore.ieee.org",
    ),
]

# Semantic Scholar API
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
# Set via env var SEMANTIC_SCHOLAR_API_KEY for higher rate limits
SEMANTIC_SCHOLAR_API_KEY = None

# HuggingFace Papers API (successor to Papers with Code)
HUGGINGFACE_PAPERS_API = "https://huggingface.co/api/papers"

# Rate limiting (seconds between requests)
RATE_LIMIT = {
    "openreview": 1.0,
    "semantic_scholar": 1.0,  # 100 req/5min without key
    "cvf": 0.5,
    "acl": 0.5,
    "aaai": 1.0,
    "ieee": 2.0,
    "huggingface": 0.5,
    "github": 1.0,
}
