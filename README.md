# Project 3: Parallel BFS using CUDA (V5.3 - World-Class Optimized)

Implementation of Breadth-First Search (BFS) and Connected Components (Afforest) on graphs using NVIDIA CUDA. Features adaptive parallelization, direction-optimization (Top-Down/Bottom-Up), delta-varint compression, and zero-copy memory streaming for billion-scale graphs.

## 🏆 Performance Highlights

| Algorithm | Graph | Time | Notes |
| :--- | :--- | :--- | :--- |
| **Compressed BFS** | Friendster (65M nodes, 3.6B edges) | **831 ms** | V5.3 Record |
| **Afforest** | Friendster | 21.5 s | Single-pass |

## Project Structure

```
Project3/
├── src/
│   ├── common/              # Common utilities (graph I/O, compression)
│   ├── legacy/v3_shared/    # Legacy kernels (Thread/Warp/Block)
│   └── v4_1_hybrid/         # V5.3 Hybrid Optimized Solvers
│       ├── bfs_adaptive.cu            # Uncompressed BFS (Hierarchical Atomics, Direct Emission)
│       ├── bfs_compressed_adaptive.cu # Compressed BFS (Zero-Copy, Warp Aggregation)
│       └── afforest.cu                # Union-Find (GCC Pruning)
├── include/
│   ├── cuda_common.h        # Core CUDA types
│   └── bfs_kernels.cuh      # Shared BFS kernel definitions
├── Julia/                   # Julia reference implementation
├── Report/                  # LaTeX report
├── Makefile
└── README.md
```

## Key Algorithms

### V5.3 Optimizations (Current)
1.  **Direct Queue Emission:** Bottom-Up kernels build the next frontier queue directly using warp-aggregated atomics, eliminating O(N) distance scans.
2.  **Warp Aggregation:** Top-Down kernels use `__ballot_sync`/`__popc` to reduce global atomics from O(edges) to O(warps).
3.  **Hybrid Direction-Optimization:** Switches between Top-Down (small frontiers) and Bottom-Up (large frontiers) based on threshold (N/26).
4.  **Delta-Varint Compression:** Reduces memory bandwidth by 37% (14.4GB → 9.1GB for Friendster).
5.  **Zero-Copy Streaming:** Enables processing of graphs larger than VRAM via pinned host memory.

### Adaptive BFS
Classifies frontier nodes by degree:
-   **Small (deg < 32):** Per-thread processing
-   **Medium (deg < 1024):** Per-warp processing
-   **Large (deg >= 1024):** Per-block processing

### Afforest (Connected Components)
Union-Find with optional neighbor sampling for fast convergence on social graphs.

## Requirements

-   NVIDIA GPU (Compute Capability 6.0+, Ampere recommended)
-   CUDA Toolkit 11.0+
-   GCC/G++ compiler
-   Make
-   HDF5 library (for `.mat` file support)

## Building

```bash
make all          # Build all versions
make v3           # Build V5.3 Hybrid Solver (recommended)
make clean        # Clean build files
```

## Usage

```bash
# Standard BFS (Uncompressed, for graphs < VRAM)
./bin/bfs_v3 --algo adaptive -s 0 <graph_file>

# Compressed BFS (Zero-Copy, for graphs > VRAM)
./bin/bfs_v3 --algo adaptive --compress -s 0 <graph_file>

# Afforest (Connected Components)
./bin/bfs_v3 --algo afforest --compress <graph_file>
```

### Supported Graph Formats
-   `.mat` (MATLAB sparse matrix, HDF5)
-   `.csrbin` (Binary CSR)
-   `.txt` (Edge list)

## Technical Walkthrough

For a detailed explanation of the optimizations, see the [Project Report](Report/report.pdf) and the [Julia Reference Implementation](Julia/bfs_module.jl).

## License

Educational use - AUTH Parallel Processing Course

## Source Code

**Repository:** [https://github.com/EvangelosMoschou/PkDSProject3.git](https://github.com/EvangelosMoschou/PkDSProject3.git)
