#!/usr/bin/env python3
"""Download OpenReview papers with browser headers (needed to bypass 403)."""
import json, subprocess, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pypdfium2 as pdfium

with open("data/bamboo_final.json") as f:
    papers = json.load(f)
pdf_dir = Path("data/paper_pdfs")

def is_valid(path):
    if not path.exists() or path.stat().st_size < 10000: return False
    try: doc = pdfium.PdfDocument(str(path)); doc.close(); return True
    except: return False

todo = []
for p in papers:
    pid = p["paper_id"]
    if is_valid(pdf_dir / f"{pid}.pdf"): continue
    url = p.get("paper_url", "")
    if "openreview" in url and not p.get("arxiv_id"):
        if "?id=" in url:
            or_id = url.split("?id=")[1].split("&")[0]
            todo.append((pid, f"https://openreview.net/pdf?id={or_id}"))
print(f"[OPENREVIEW] {len(todo)} papers", flush=True)

def dl(args):
    pid, url = args
    pp = pdf_dir / f"{pid}.pdf"
    try:
        subprocess.run(["curl","-sL","--noproxy","*","-o",str(pp),"--max-time","120",
            "-H","User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "-H","Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "-H","Referer: https://openreview.net/",
            url], capture_output=True,timeout=180)
        if is_valid(pp): return True
        pp.unlink(missing_ok=True); return False
    except: pp.unlink(missing_ok=True); return False

ok=0;fail=0
with ThreadPoolExecutor(max_workers=8) as ex:
    futs={ex.submit(dl,a):a[0] for a in todo}
    for i,f in enumerate(as_completed(futs)):
        if f.result(): ok+=1
        else: fail+=1
        if (i+1)%200==0:
            t=sum(1 for p in pdf_dir.glob("*.pdf") if is_valid(p))
            print(f"[OPENREVIEW] [{i+1}/{len(todo)}] +{ok} ok, {fail} fail | total PDFs: {t}",flush=True)
print(f"[OPENREVIEW] Done: +{ok} ok, {fail} fail | total: {sum(1 for p in pdf_dir.glob('*.pdf') if is_valid(p))}",flush=True)
