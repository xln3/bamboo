#!/usr/bin/env python3
"""MinerU-based PDF text extraction for BAMBOO.

Replaces pdftotext with MinerU for higher-quality text extraction from
academic PDFs. Preserves table structure as markdown, which is critical
for accurate claim extraction.

Usage as module:
    from pdf_extractor import extract_text_mineru, extract_abstract

Usage standalone (test):
    python pdf_extractor.py /path/to/paper.pdf
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger("pdf_extractor")

# ---------------------------------------------------------------------------
# MinerU extraction
# ---------------------------------------------------------------------------

def extract_text_mineru(pdf_path: str, *, method: str = "auto",
                        timeout: int = 120) -> str:
    """Extract text from a PDF using MinerU's pipeline backend.

    Note: MinerU on CPU can be very slow (minutes per paper). Consider
    using extract_text_fast() for batch processing without GPU.

    Args:
        pdf_path: Path to the PDF file.
        method: Parsing method ("auto", "txt", "ocr").
        timeout: Max seconds for the extraction process.

    Returns:
        Extracted text in markdown format. Falls back to pdftotext on failure.
    """
    pdf_path = str(pdf_path)
    if not os.path.isfile(pdf_path):
        log.warning(f"PDF not found: {pdf_path}")
        return ""

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            result = subprocess.run(
                [
                    "mineru",
                    "-p", pdf_path,
                    "-o", tmpdir,
                    "-b", "pipeline",
                    "-m", method,
                ],
                capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "MINERU_LOG_LEVEL": "WARNING"},
            )

            if result.returncode != 0:
                log.warning(f"MinerU failed (exit {result.returncode}): "
                            f"{result.stderr[:200]}")
                return _fallback_pdftotext(pdf_path)

            # MinerU outputs: <output_dir>/<stem>/<method>/<stem>.md
            stem = Path(pdf_path).stem
            md_path = os.path.join(tmpdir, stem, method, f"{stem}.md")

            # If method was "auto", MinerU may pick "txt" or "ocr"
            if not os.path.exists(md_path):
                for m in ("auto", "txt", "ocr"):
                    candidate = os.path.join(tmpdir, stem, m, f"{stem}.md")
                    if os.path.exists(candidate):
                        md_path = candidate
                        break

            if not os.path.exists(md_path):
                # Try to find any .md file
                for root, dirs, files in os.walk(tmpdir):
                    for f in files:
                        if f.endswith(".md"):
                            md_path = os.path.join(root, f)
                            break
                    if os.path.exists(md_path):
                        break

            if os.path.exists(md_path):
                text = Path(md_path).read_text(encoding="utf-8", errors="replace")
                if len(text) > 100:
                    return text
                else:
                    log.warning(f"MinerU output too short ({len(text)} chars)")
                    return _fallback_pdftotext(pdf_path)
            else:
                log.warning("MinerU produced no .md output")
                return _fallback_pdftotext(pdf_path)

        except subprocess.TimeoutExpired:
            log.warning(f"MinerU timed out after {timeout}s")
            return _fallback_pdftotext(pdf_path)
        except Exception as e:
            log.warning(f"MinerU error: {e}")
            return _fallback_pdftotext(pdf_path)


def _fallback_pdftotext(pdf_path: str) -> str:
    """Fallback to pdftotext when MinerU fails."""
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", pdf_path, "-"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and len(result.stdout) > 100:
            log.info("Fell back to pdftotext")
            return result.stdout
    except Exception as e:
        log.warning(f"pdftotext fallback also failed: {e}")
    return ""


def extract_text_fast(pdf_path: str) -> str:
    """Fast text extraction using pdftotext (no ML model overhead).

    Use this for batch processing when MinerU/GPU is unavailable.
    """
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and len(result.stdout) > 100:
            return result.stdout
    except Exception as e:
        log.warning(f"pdftotext failed: {e}")
    return ""


# ---------------------------------------------------------------------------
# Abstract extraction from markdown text
# ---------------------------------------------------------------------------

_ABSTRACT_PATTERNS = [
    # Standard "Abstract" heading followed by text
    re.compile(
        r"(?:^|\n)#+\s*Abstract\s*\n+(.*?)(?=\n#+\s|\n##|\n\*\*1[\.\s]|\n1[\.\s]+Introduction)",
        re.IGNORECASE | re.DOTALL,
    ),
    # Bold "Abstract" marker
    re.compile(
        r"\*\*Abstract\*\*[:\.\s]*(.*?)(?=\n#+\s|\n\*\*1[\.\s]|\n1[\.\s]+Introduction|\n\*\*Introduction)",
        re.IGNORECASE | re.DOTALL,
    ),
    # "Abstract" as a line, followed by text until next section
    re.compile(
        r"(?:^|\n)Abstract\s*\n+(.*?)(?=\n#+\s|\n\d+[\.\s]+[A-Z]|\nIntroduction|\n\*\*)",
        re.IGNORECASE | re.DOTALL,
    ),
    # "ABSTRACT" all caps
    re.compile(
        r"(?:^|\n)ABSTRACT\s*\n+(.*?)(?=\n#+\s|\n\d+[\.\s]+[A-Z]|\nINTRODUCTION|\n1[\.\s])",
        re.IGNORECASE | re.DOTALL,
    ),
]


def extract_abstract(text: str) -> str:
    """Extract the abstract from paper text (markdown or plain).

    Args:
        text: Full paper text (markdown from MinerU or plain from pdftotext).

    Returns:
        Extracted abstract text, or empty string if not found.
    """
    for pattern in _ABSTRACT_PATTERNS:
        m = pattern.search(text)
        if m:
            abstract = m.group(1).strip()
            # Clean up markdown artifacts
            abstract = re.sub(r"\n{2,}", " ", abstract)
            abstract = re.sub(r"\s{2,}", " ", abstract)
            # Remove image references
            abstract = re.sub(r"!\[.*?\]\(.*?\)", "", abstract)
            # Remove leftover markdown formatting
            abstract = abstract.replace("**", "").replace("*", "")
            abstract = abstract.strip()
            if len(abstract) > 50:
                return abstract

    return ""


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"Extracting text from: {pdf_path}")
    text = extract_text_mineru(pdf_path)
    print(f"\n--- Extracted {len(text)} chars ---\n")
    print(text[:3000])

    abstract = extract_abstract(text)
    if abstract:
        print(f"\n--- Abstract ({len(abstract)} chars) ---\n")
        print(abstract)
    else:
        print("\n--- No abstract found ---")
