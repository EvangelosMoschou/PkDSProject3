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

# Build all versions
make clean
make all

echo ""
echo "Build complete! Executables:"
ls -la bin/

echo ""
echo "Usage:"
echo "  ./bin/bfs_v1 <graph_file> -s <source>   # Dynamic Thread Assignment"
echo "  ./bin/bfs_v2 <graph_file> -s <source>   # Chunk-Based Processing"
echo "  ./bin/bfs_v3 <graph_file> -s <source>   # Shared Memory + Warp Cooperation"
