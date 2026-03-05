#!/bin/bash
# Run arXiv search for all venues sequentially (to respect arXiv rate limits)
# Safe to interrupt and resume — progress is saved per-venue.

set -e
cd "$(dirname "$0")"

VENUES="iclr2025 icml2025 neurips2025 cvpr2025 iccv2025 acl2025 emnlp2025 aaai2025"

for venue in $VENUES; do
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
