#!/usr/bin/env python3
"""MinerU worker: processes PDFs in priority order.

Usage:
    python mineru_worker.py --worker-id 0 --total-workers 8 --device cuda
    python mineru_worker.py --worker-id 4 --total-workers 8 --device cpu
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
import pypdfium2 as pdfium

BASE = Path(__file__).parent.parent.parent
PDF_DIR = BASE / "data" / "paper_pdfs"
MD_DIR = BASE / "data" / "paper_markdowns"

# Priority order: AAAI remainder → EMNLP → ACL → CVPR → NeurIPS → ICLR → ICML → ICRA
VENUE_ORDER = ["AAAI", "EMNLP", "ACL", "CVPR", "NeurIPS", "ICLR", "ICML", "ICCV", "ICRA"]


def is_valid_pdf(path):
    if not path.exists() or path.stat().st_size < 10000:
        return False
    try:
        doc = pdfium.PdfDocument(str(path))
        doc.close()
        return True
    except:
        return False


def get_todo(worker_id, total_workers):
    """Get papers to process in priority venue order, partitioned by worker_id."""
    with open(BASE / "data" / "bamboo_final.json") as f:
        papers = json.load(f)

    # Group by venue
    by_venue = {}
    for p in papers:
        v = p.get("venue", "?")
        by_venue.setdefault(v, []).append(p["paper_id"])

    # Build ordered list
    ordered = []
    for v in VENUE_ORDER:
        ordered.extend(sorted(by_venue.get(v, [])))
    # Add any remaining
    seen = set(ordered)
    for p in papers:
        if p["paper_id"] not in seen:
            ordered.append(p["paper_id"])

    # Filter: has valid PDF, no MD yet
    todo = []
    for pid in ordered:
        if (MD_DIR / f"{pid}.md").exists():
            continue
        if (MD_DIR / f"_tmp_{pid}").exists():
            continue  # another worker is on it
        if not is_valid_pdf(PDF_DIR / f"{pid}.pdf"):
            continue
        todo.append(pid)

    # Partition: this worker takes every Nth paper
    return [pid for i, pid in enumerate(todo) if i % total_workers == worker_id]


def process_one(pid, device):
    pdf_path = PDF_DIR / f"{pid}.pdf"
    md_path = MD_DIR / f"{pid}.md"
    tmp_out = MD_DIR / f"_tmp_{pid}"

    if md_path.exists():
        return True

    env = os.environ.copy()
    env["MINERU_MODEL_SOURCE"] = "modelscope"
    env["MINERU_DEVICE_MODE"] = device
    if device == "cpu":
        env["OMP_NUM_THREADS"] = "8"

    try:
        subprocess.run(
            ["mineru", "-p", str(pdf_path), "-o", str(tmp_out),
             "-b", "hybrid-auto-engine"],
            capture_output=True, text=True,
            timeout=1800 if device == "cpu" else 1200,
            env=env,
        )
        md_files = list(tmp_out.rglob("*.md"))
        if md_files:
            shutil.copy2(md_files[0], md_path)
            shutil.rmtree(tmp_out, ignore_errors=True)
            return True
        shutil.rmtree(tmp_out, ignore_errors=True)
        return False
    except Exception as e:
        shutil.rmtree(tmp_out, ignore_errors=True)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--total-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    todo = get_todo(args.worker_id, args.total_workers)
    print(f"W{args.worker_id}({args.device}): {len(todo)} papers to process", flush=True)

    for pid in todo:
        if (MD_DIR / f"{pid}.md").exists():
            continue
        ok = process_one(pid, args.device)
        n_done = len(list(MD_DIR.glob("*.md")))
        status = "OK" if ok else "FAIL"
        print(f"W{args.worker_id}({args.device}) {status}:{pid} (total:{n_done})", flush=True)

    print(f"W{args.worker_id}({args.device}) FINISHED", flush=True)


if __name__ == "__main__":
    main()
