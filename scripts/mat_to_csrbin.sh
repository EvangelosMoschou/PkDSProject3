#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MAT_FILE="${1:-$ROOT_DIR/Mat Files/road_usa.mat}"
BIN_FILE="${2:-$ROOT_DIR/Mat Files/road_usa.bin}"

if [ ! -f "$MAT_FILE" ]; then
  echo "Error: input file not found: $MAT_FILE"
  exit 1
fi

if [ ! -x "$ROOT_DIR/bin/mat_to_csrbin" ]; then
  echo "Tool not found: $ROOT_DIR/bin/mat_to_csrbin"
  echo "Build it first: make mat_to_csrbin_tool"
  exit 1
fi

"$ROOT_DIR/bin/mat_to_csrbin" "$MAT_FILE" "$BIN_FILE"
