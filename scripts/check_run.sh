#!/bin/bash
# Quick status check for running BAMBOO reproduction
LOG=/home/xln/panda2026/bamboo/data/results/panda-glm-5/run2.log
RESULTS=/home/xln/panda2026/bamboo/data/results/panda-glm-5

echo "=== Runner process ==="
ps aux | grep "[r]unner.py" | head -3
echo

echo "=== PANDA processes ==="
ps aux | grep "[p]anda.*mod.ts" | head -3
echo

echo "=== Last 5 lines of runner log ==="
tail -5 "$LOG" 2>/dev/null
echo

echo "=== Results so far ==="
for f in "$RESULTS"/bamboo-*.json; do
    [ -f "$f" ] || continue
    pid=$(basename "$f" .json)
    ts=$(stat -c '%Y' "$f")
    now=$(date +%s)
    age=$(( (now - ts) / 60 ))
    level=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('pass4',{}).get('overall_level','?'))" 2>/dev/null)
    echo "  $pid: L$level (${age}min ago)"
done
