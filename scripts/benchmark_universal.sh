#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C

BIN="bin/bfs_v4_1_hybrid"
GRAPH=""
TRIALS=5
SOURCE=0
USE_COMPRESS=1
EXTRA_ARGS=""

usage() {
  cat <<'EOF'
Universal BFS benchmark (portable metrics)

Usage:
  scripts/benchmark_universal.sh --graph <graph.bin> [options]

Options:
  --bin <path>          Binary to run (default: bin/bfs_v4_1_hybrid)
  --graph <path>        Input graph file (required)
  --trials <N>          Number of runs (default: 5)
  --source <id>         BFS source node (default: 0)
  --no-compress         Disable --compress
  --extra-args "..."    Extra args passed to binary
  -h, --help            Show this help

Output metrics:
  - Median Time (ms)
  - Mean Time (ms)
  - MTEPS (Million Traversed Edges Per Second)
  - ns/edge (nanoseconds per traversed edge)
  - MNPS (Million reached nodes per second)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin)
      BIN="$2"
      shift 2
      ;;
    --graph)
      GRAPH="$2"
      shift 2
      ;;
    --trials)
      TRIALS="$2"
      shift 2
      ;;
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --no-compress)
      USE_COMPRESS=0
      shift
      ;;
    --extra-args)
      EXTRA_ARGS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$GRAPH" ]]; then
  echo "Error: --graph is required" >&2
  usage
  exit 1
fi

if [[ ! -x "$BIN" ]]; then
  echo "Error: binary not found/executable: $BIN" >&2
  exit 1
fi

if [[ ! -f "$GRAPH" ]]; then
  echo "Error: graph file not found: $GRAPH" >&2
  exit 1
fi

if ! [[ "$TRIALS" =~ ^[0-9]+$ ]] || [[ "$TRIALS" -lt 1 ]]; then
  echo "Error: --trials must be a positive integer" >&2
  exit 1
fi

declare -a times
declare -a mteps_values
declare -a ns_edge_values
declare -a mnps_values

edges=""
reachable=""

run_cmd=("$BIN" "$GRAPH" "-s" "$SOURCE")
if [[ "$USE_COMPRESS" -eq 1 ]]; then
  run_cmd+=("--compress")
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_split=($EXTRA_ARGS)
  run_cmd+=("${extra_split[@]}")
fi

for ((i=1; i<=TRIALS; i++)); do
  echo "[trial $i/$TRIALS] running..."

  set +e
  out="$("${run_cmd[@]}" 2>&1)"
  status=$?
  set -e

  if [[ $status -ne 0 ]]; then
    echo "Run failed with exit code $status" >&2
    echo "$out" >&2
    exit $status
  fi

  t_ms="$(echo "$out" | grep -m1 "Time:" | awk '{print $2}')"
  if [[ -z "$t_ms" ]]; then
    echo "Could not parse Time from output" >&2
    echo "$out" >&2
    exit 1
  fi

  if [[ -z "$edges" ]]; then
    edges="$(echo "$out" | grep -m1 '^Edges:' | awk '{print $2}')"
    if [[ -z "$edges" ]]; then
      edges="$(echo "$out" | grep -m1 'Loading Graph Binary:' | sed -E 's/.*Edges=([0-9]+).*/\1/')"
    fi
  fi

  if [[ -z "$reachable" ]]; then
    reachable="$(echo "$out" | grep -m1 'Reachable Nodes:' | sed -E 's/.*Reachable Nodes: ([0-9]+) \/.*/\1/')"
  fi

  if [[ -z "$edges" ]]; then
    echo "Could not parse edge count from output" >&2
    echo "$out" >&2
    exit 1
  fi

  if [[ -z "$reachable" ]]; then
    reachable=0
  fi

  mteps="$(awk -v e="$edges" -v ms="$t_ms" 'BEGIN { printf "%.3f", ((e * 1000.0) / ms) / 1e6 }')"
  ns_edge="$(awk -v e="$edges" -v ms="$t_ms" 'BEGIN { printf "%.3f", (ms * 1e6) / e }')"
  mnps="$(awk -v n="$reachable" -v ms="$t_ms" 'BEGIN { if (n > 0) printf "%.3f", ((n * 1000.0) / ms) / 1e6; else printf "0.000" }')"

  times+=("$t_ms")
  mteps_values+=("$mteps")
  ns_edge_values+=("$ns_edge")
  mnps_values+=("$mnps")

  echo "  time=${t_ms} ms | MTEPS=${mteps} | ns/edge=${ns_edge} | MNPS=${mnps}"
done

mapfile -t sorted_times < <(printf '%s\n' "${times[@]}" | sort -n)
count=${#sorted_times[@]}

if (( count % 2 == 1 )); then
  median_time="${sorted_times[$((count/2))]}"
else
  low="${sorted_times[$((count/2 - 1))]}"
  high="${sorted_times[$((count/2))]}"
  median_time="$(awk -v a="$low" -v b="$high" 'BEGIN { printf "%.3f", (a+b)/2.0 }')"
fi

mean_time="$(printf '%s\n' "${times[@]}" | awk '{sum+=$1} END {printf "%.3f", sum/NR}')"
median_mteps="$(awk -v e="$edges" -v ms="$median_time" 'BEGIN { printf "%.3f", ((e * 1000.0) / ms) / 1e6 }')"
median_ns_edge="$(awk -v e="$edges" -v ms="$median_time" 'BEGIN { printf "%.3f", (ms * 1e6) / e }')"
median_mnps="$(awk -v n="$reachable" -v ms="$median_time" 'BEGIN { if (n > 0) printf "%.3f", ((n * 1000.0) / ms) / 1e6; else printf "0.000" }')"

echo
echo "===== Universal Benchmark Summary ====="
echo "binary:        $BIN"
echo "graph:         $GRAPH"
echo "mode:          $([[ "$USE_COMPRESS" -eq 1 ]] && echo compressed || echo uncompressed)"
echo "trials:        $TRIALS"
echo "edges:         $edges"
echo "reachable:     $reachable"
echo "median time:   ${median_time} ms"
echo "mean time:     ${mean_time} ms"
echo "median MTEPS:  ${median_mteps}"
echo "median ns/edge:${median_ns_edge}"
echo "median MNPS:   ${median_mnps}"
