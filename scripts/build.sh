#!/bin/bash
# Build script for Project 3: CUDA BFS

set -e

echo "Building CUDA BFS implementations..."

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA toolkit."
    exit 1
fi

# Print CUDA version
echo "CUDA version: $(nvcc --version | grep release)"

HOST_NATIVE="${HOST_NATIVE:-1}"
ENABLE_LTO="${ENABLE_LTO:-0}"
PGO="${PGO:-off}"

echo "Build options: HOST_NATIVE=${HOST_NATIVE} ENABLE_LTO=${ENABLE_LTO} PGO=${PGO}"

# Build all versions
make clean
make all HOST_NATIVE="${HOST_NATIVE}" ENABLE_LTO="${ENABLE_LTO}" PGO="${PGO}"
make v41 HOST_NATIVE="${HOST_NATIVE}" ENABLE_LTO="${ENABLE_LTO}" PGO="${PGO}"

echo ""
echo "Build complete! Executables:"
ls -la bin/

echo ""
echo "Usage:"
echo "  ./bin/bfs_v5_multi_gpu <graph_file>"
echo "  ./bin/bfs_v4_1_hybrid <graph_file> --compress -s 0"
