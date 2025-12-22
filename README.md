# Project 3: Parallel BFS using CUDA

Implementation of Breadth-First Search (BFS) on graphs using NVIDIA CUDA, featuring three different parallelization strategies.

## Project Structure

```
Project3/
├── assignment/          # Assignment description
│   └── README_ASSIGNMENT.md
├── src/                 # Source files
│   ├── common/          # Common utilities
│   │   ├── graph.h
│   │   ├── graph.cu
│   │   ├── utils.h
│   │   └── utils.cu
│   ├── v1_dynamic/      # Version 1: Dynamic thread assignment
│   │   ├── bfs_dynamic.h
│   │   └── bfs_dynamic.cu
│   ├── v2_chunked/      # Version 2: Chunk-based processing
│   │   ├── bfs_chunked.h
│   │   └── bfs_chunked.cu
│   └── v3_shared/       # Version 3: Shared memory + warp cooperation
│       ├── bfs_shared.h
│       └── bfs_shared.cu
├── include/             # Header files
│   └── cuda_common.h
├── tests/               # Test graphs and validation
│   ├── graphs/
│   └── validation/
├── benchmarks/          # Performance benchmarks
│   └── results/
├── scripts/             # Build and run scripts
│   ├── build.sh
│   └── run_benchmarks.sh
├── Report/              # Project report
│   └── report.tex
├── Makefile
└── README.md
```

## Technical Walkthrough

For a detailed explanation of the SOTA Async Streaming, Zero-Copy optimizations, and the Strategic Dispatcher, see [walkthrough.txt](./walkthrough.txt).

## Versions

### Version 1: Dynamic Thread Assignment
One thread explores one node at a time. When finished, it dynamically picks up another unprocessed node from the work queue.

### Version 2: Chunk-Based Processing
Each thread processes a fixed chunk of nodes using an internal for-loop. Similar to the pthreads approach.

### Version 3: Shared Memory with Warp Cooperation
Threads within the same block collaborate using shared memory. Each warp cooperates to load neighbors of the same node.

## Requirements

- NVIDIA GPU with CUDA Compute Capability 3.0+
- CUDA Toolkit 11.0+
- GCC/G++ compiler
- Make

## Building

```bash
make all          # Build all versions
make v1           # Build Version 1 only
make v2           # Build Version 2 only
make v3           # Build Version 3 only
make clean        # Clean build files
```

## Usage

```bash
./bin/bfs_v1 <graph_file> <source_node>
./bin/bfs_v2 <graph_file> <source_node>
./bin/bfs_v3 <graph_file> <source_node>
```

## Performance Testing

```bash
./scripts/run_benchmarks.sh
```

## License

Educational use - AUTH Parallel Processing Course
