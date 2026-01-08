#!/bin/bash
set -e

ORIG="Mat Files/com-Friendster.mat.csrbin"
RCM="Mat Files/com-Friendster_RCM.mat.csrbin"

if [ ! -f "$ORIG" ]; then
    echo "Error: Original graph not found at $ORIG"
    exit 1
fi
if [ ! -f "$RCM" ]; then
    echo "Error: RCM graph not found at $RCM"
    exit 1
fi

echo "=========================================================="
echo "          FRIENDSTER AFFOREST BENCHMARK"
echo "=========================================================="

# 1. Baseline Afforest
echo "[1/2] Running Baseline Afforest (Compressed)..."
./bin/bfs_v3 "$ORIG" --algo afforest --compress --benchmark 3 > afforest_baseline.txt
grep "Average Time" afforest_baseline.txt

# 2. RCM Afforest
echo "[2/2] Running RCM Afforest (Compressed)..."
./bin/bfs_v3 "$RCM" --algo afforest --compress --benchmark 3 > afforest_rcm.txt
grep "Average Time" afforest_rcm.txt

echo "=========================================================="
echo "BENCHMARK COMPLETE"
echo "Check afforest_baseline.txt and afforest_rcm.txt"
echo "=========================================================="
