#!/bin/bash
set -e

GRAPH="Mat Files/com-Friendster.mat.csrbin"

if [ ! -f "$GRAPH" ]; then
    echo "Error: $GRAPH not found!"
    exit 1
fi

echo "=========================================================="
echo "          FRIENDSTER RCM BENCHMARK PROTOCOL"
echo "=========================================================="

# 2. RCM Reordering (Streaming Mode)
RCM_GRAPH="Mat Files/com-Friendster_RCM.mat.csrbin"

if [ -f "$RCM_GRAPH" ]; then
    echo "[2/3] Found existing RCM Graph: $RCM_GRAPH"
    echo "Skipping reordering (Delete file to re-run)."
else
    echo "[2/3] Performing Streaming RCM Reordering..."
    echo "      Input: $GRAPH"
    echo "      Output: $RCM_GRAPH"
    ./reorder_graph "$GRAPH" "$RCM_GRAPH" rcm
fi

# 3. RCM Run (After RCM)
echo "[3/3] Running RCM BFS (Compressed Adaptive)..."
./bin/bfs_v3 "$RCM_GRAPH" --algo adaptive --compress --benchmark 3 > rcm_log.txt
grep "Average Time" rcm_log.txt

echo "=========================================================="
echo "BENCHMARK COMPLETE"
echo "Check baseline_log.txt and rcm_log.txt for full details."
echo "=========================================================="
