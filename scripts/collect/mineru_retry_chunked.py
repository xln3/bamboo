#!/usr/bin/env python3
"""Retry failed MinerU conversions by splitting PDFs into ~20-page chunks.

For each failed paper:
  1. Split PDF into chunks of CHUNK_PAGES pages using pypdf
  2. Run MinerU on each chunk independently (smaller VRAM, shorter runtime)
  3. Concatenate chunk markdowns into final .md
  4. Clean up temp files

Usage:
    # Local (3 workers, 1 GPU)
    python mineru_retry_chunked.py --worker-id 0 --total-workers 3 --device cuda

    # Remote (24 workers, 8 GPUs — launch with CUDA_VISIBLE_DEVICES)
    CUDA_VISIBLE_DEVICES=0 python mineru_retry_chunked.py --worker-id 0 --total-workers 24 --device cuda
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
from pypdf import PdfReader, PdfWriter

BASE = Path(__file__).parent.parent.parent
PDF_DIR = BASE / "data" / "paper_pdfs"
MD_DIR = BASE / "data" / "paper_markdowns"
CHUNK_PAGES = 20
CHUNK_TIMEOUT = 600  # 10 min per chunk (vs 20 min for whole PDF)


def get_failed_papers(worker_id, total_workers):
    """Get papers that have a valid PDF but no markdown yet."""
    pdfs = {f.replace(".pdf", "") for f in os.listdir(PDF_DIR) if f.endswith(".pdf")}
    mds = {f.replace(".md", "") for f in os.listdir(MD_DIR) if f.endswith(".md")}
    failed = sorted(pdfs - mds)

    # Filter: valid PDF, not currently being processed
    todo = []
    for pid in failed:
        pdf_path = PDF_DIR / f"{pid}.pdf"
        if pdf_path.stat().st_size < 10000:
            continue
        tmp_marker = MD_DIR / f"_tmp_chunk_{pid}"
        if tmp_marker.exists():
            # Check if stale (>30 min old)
            age = time.time() - tmp_marker.stat().st_mtime
            if age < 1800:
                continue
        todo.append(pid)

    # Partition
    return [pid for i, pid in enumerate(todo) if i % total_workers == worker_id]


def split_pdf(pdf_path, chunk_dir, chunk_pages=CHUNK_PAGES):
    """Split a PDF into chunks of chunk_pages pages. Returns list of chunk paths."""
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    chunks = []

    for start in range(0, total_pages, chunk_pages):
        end = min(start + chunk_pages, total_pages)
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])

        chunk_path = chunk_dir / f"chunk_{start:04d}_{end:04d}.pdf"
        with open(chunk_path, "wb") as f:
            writer.write(f)
        chunks.append(chunk_path)

    return chunks


def run_mineru_on_chunk(chunk_path, output_dir, device, env):
    """Run MinerU on a single chunk PDF. Returns markdown text or None."""
    try:
        subprocess.run(
            ["mineru", "-p", str(chunk_path), "-o", str(output_dir),
             "-b", "hybrid-auto-engine"],
            capture_output=True, text=True,
            timeout=CHUNK_TIMEOUT,
            env=env,
        )
        md_files = list(output_dir.rglob("*.md"))
        if md_files:
            text = md_files[0].read_text(encoding="utf-8", errors="replace")
            return text
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def process_one(pid, device):
    """Process a single paper by chunking, converting, and concatenating."""
    pdf_path = PDF_DIR / f"{pid}.pdf"
    md_path = MD_DIR / f"{pid}.md"

    if md_path.exists():
        return True

    tmp_dir = MD_DIR / f"_tmp_chunk_{pid}"
    tmp_dir.mkdir(exist_ok=True)
    chunk_dir = tmp_dir / "chunks"
    chunk_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["MINERU_MODEL_SOURCE"] = "modelscope"
    env["MINERU_DEVICE_MODE"] = device

    try:
        # Step 1: Split PDF
        chunks = split_pdf(pdf_path, chunk_dir)

        # Step 2: Run MinerU on each chunk
        md_parts = []
        failed_chunks = 0
        for i, chunk_path in enumerate(chunks):
            chunk_out = tmp_dir / f"out_{i:03d}"
            chunk_out.mkdir(exist_ok=True)

            text = run_mineru_on_chunk(chunk_path, chunk_out, device, env)
            if text:
                md_parts.append(text)
            else:
                failed_chunks += 1
                md_parts.append(f"\n\n<!-- [chunk {i+1}/{len(chunks)} failed] -->\n\n")

            # Clean up chunk output immediately to save disk
            shutil.rmtree(chunk_out, ignore_errors=True)

        # Step 3: Concatenate — succeed if we got at least 50% of chunks
        if len(md_parts) - failed_chunks >= len(chunks) * 0.5:
            combined = "\n\n".join(md_parts)
            md_path.write_text(combined, encoding="utf-8")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return True
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return False

    except Exception as e:
        print(f"  ERROR {pid}: {e}", flush=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--total-workers", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    todo = get_failed_papers(args.worker_id, args.total_workers)
    print(f"W{args.worker_id}({args.device}): {len(todo)} failed papers to retry (chunked)", flush=True)

    ok_count = 0
    fail_count = 0
    for pid in todo:
        if (MD_DIR / f"{pid}.md").exists():
            continue
        start = time.time()
        success = process_one(pid, args.device)
        elapsed = time.time() - start
        n_done = len(list(MD_DIR.glob("*.md")))

        if success:
            ok_count += 1
            print(f"W{args.worker_id} OK:{pid} ({elapsed:.0f}s, total:{n_done})", flush=True)
        else:
            fail_count += 1
            print(f"W{args.worker_id} FAIL:{pid} ({elapsed:.0f}s, total:{n_done})", flush=True)

    print(f"W{args.worker_id} FINISHED: {ok_count} ok, {fail_count} fail", flush=True)


if __name__ == "__main__":
    main()
