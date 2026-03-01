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
make all          # Build v5 executable: bin/bfs_v5_multi_gpu
make v41          # Build v4.1 hybrid executable: bin/bfs_v4_1_hybrid
make clean        # Clean build files
```

### Compiler Optimization Modes

`Makefile` now supports native tuning, optional LTO, and optional PGO:

```bash
# Native host tuning (default enabled)
make v41 HOST_NATIVE=1

# Native + LTO
make v41 HOST_NATIVE=1 ENABLE_LTO=1

# Portable build (disable native)
make v41 HOST_NATIVE=0
```

PGO workflow (2-phase):

```bash
# 1) Build with instrumentation
make pgo-gen

# 2) Run representative workload(s)
bin/bfs_v4_1_hybrid "Mat Files/road_usa.bin" --compress

# 3) Rebuild using collected profiles
make pgo-use
```

## Usage

```bash
# Standard BFS (Uncompressed, for graphs < VRAM)
./bin/bfs_v4_1_hybrid <graph_file> -s 0

# Compressed BFS (Zero-Copy, for graphs > VRAM)
./bin/bfs_v4_1_hybrid <graph_file> --compress -s 0

# Afforest (Connected Components)
./bin/bfs_v4_1_hybrid <graph_file> --algo afforest --compress
```

## Universal Benchmarking (Cross-Machine)

Raw time in ms is useful, but a more portable comparison metric is:

- `MTEPS` = million traversed edges per second
- `ns/edge` = nanoseconds spent per traversed edge

Use the helper script:

```bash
bash scripts/benchmark_universal.sh --graph "Mat Files/road_usa.bin" --trials 5
```

### Latest Universal Results (Road USA, Compressed, 3 Trials)

| Binary | Median Time (ms) | Median MTEPS | Median ns/edge | Median MNPS |
| :--- | ---: | ---: | ---: | ---: |
| `bin/bfs_v4_1_hybrid` | 1132.285 | 50.967 | 19.621 | 21.150 |
| `bin/bfs_v5_multi_gpu` | 1188.301 | 48.564 | 20.591 | 20.153 |

Reproduce with:

```bash
bash scripts/benchmark_universal.sh --bin bin/bfs_v4_1_hybrid --graph "Mat Files/road_usa.bin" --trials 3
bash scripts/benchmark_universal.sh --bin bin/bfs_v5_multi_gpu --graph "Mat Files/road_usa.bin" --trials 3
```

Optional examples:

```bash
# Uncompressed
bash scripts/benchmark_universal.sh --graph "Mat Files/road_usa.bin" --no-compress --trials 5

# Different source and extra args
bash scripts/benchmark_universal.sh --graph "Mat Files/road_usa.bin" --source 42 --extra-args "--validate"
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
