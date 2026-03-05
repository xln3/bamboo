#!/bin/bash
# Wait for CVPR to finish, then run remaining venues sequentially
cd "$(dirname "$0")"

echo "Waiting for CVPR process (PID 3039778) to complete..."
while kill -0 3039778 2>/dev/null; do
    sleep 30
done
echo "CVPR process complete at $(date)"

# Run remaining venues (skip CVPR since it's done, skip OpenReview venues that already have abstract-based code URLs)
for venue in iccv2025 acl2025 emnlp2025 aaai2025 iclr2025 icml2025 neurips2025; do
    echo ""
    echo "================================================================"
    echo "Processing $venue — $(date)"
    echo "================================================================"
    python3 batch_find_arxiv.py --venue "$venue"
    echo "$venue done at $(date)"
done

echo ""
echo "================================================================"
echo "ALL VENUES COMPLETE — $(date)"
echo "================================================================"

# Print final stats
python3 -c "
import json
from pathlib import Path
DATA = Path('../../data/papers')
total_all = 0
total_code = 0
for f in sorted(DATA.glob('*.json')):
    if f.stem in ('all_papers','papers_with_code','papers_validated'): continue
    papers = json.loads(f.read_text())
    n = sum(1 for p in papers if p.get('code_url'))
    print(f'  {f.stem:>12}: {n:>5} code / {len(papers):>5} papers ({n/max(len(papers),1)*100:.1f}%)')
    total_all += len(papers)
    total_code += n
print(f'  {\"TOTAL\":>12}: {total_code:>5} code / {total_all:>5} papers ({total_code/max(total_all,1)*100:.1f}%)')
"
