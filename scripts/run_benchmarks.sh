#!/bin/bash
# Benchmark script for comparing all BFS versions

set -e

GRAPH_FILE=${1:-"tests/graphs/test_graph.txt"}
SOURCE=${2:-0}
NUM_RUNS=${3:-10}

echo "========================================"
echo "BFS Benchmark Suite"
echo "========================================"
echo "Graph: $GRAPH_FILE"
echo "Source: $SOURCE"
echo "Iterations: $NUM_RUNS"
echo "========================================"
echo ""

# Check if graph file exists
if [ ! -f "$GRAPH_FILE" ]; then
    echo "Error: Graph file not found: $GRAPH_FILE"
    exit 1
fi

# Results file
RESULTS_DIR="benchmarks/results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/benchmark_$(date +%Y%m%d_%H%M%S).csv"

echo "version,time_ms,validated" > "$RESULTS_FILE"

# Run each version
for version in v1 v2 v3; do
    echo "----------------------------------------"
    echo "Running Version: $version"
    echo "----------------------------------------"
    
    BIN="./bin/bfs_$version"
    
    if [ ! -f "$BIN" ]; then
        echo "Warning: $BIN not found, skipping..."
        continue
    fi
    
    # Run with benchmark flag
    OUTPUT=$($BIN "$GRAPH_FILE" -s $SOURCE -b $NUM_RUNS 2>&1)
    
    echo "$OUTPUT"
    
    # Extract average time (simple parsing)
    AVG_TIME=$(echo "$OUTPUT" | grep "Average time" | awk '{print $3}')
    VALIDATED=$(echo "$OUTPUT" | grep -q "PASSED" && echo "yes" || echo "no")
    
    echo "$version,$AVG_TIME,$VALIDATED" >> "$RESULTS_FILE"
    echo ""
done

echo "========================================"
echo "Benchmark complete!"
echo "Results saved to: $RESULTS_FILE"
echo "========================================"

# Print summary
echo ""
echo "Summary:"
echo "--------"
cat "$RESULTS_FILE" | column -t -s ','
