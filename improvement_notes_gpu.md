# Project 3 Improvements (GPU/CUDA)

Based on the analysis of `src/v3_shared/bfs_shared.cu` and findings from the Exercise (MPI) project, here are the targeted improvements for Project 3.

## A. Infrastructure & Reporting (Missing)
The current `bfs_shared.cu` prints hardcoded text to stdout. We need:
1.  **JSON Output**: Create `src/common/json_gpu.cu` to print structured results (Time, Traversed Edges, GTEPS, Device Info).
2.  **Multi-Trial Benchmarking**: Loop the kernel execution 5-10 times to get stable mean/median times, filtering out "warmup" noise.

## B. 64-bit Edge Indices (Confirmed)
- **Status**: [SAFE]
- `include/cuda_common.h` defines `typedef long long edge_t;`.
- `bfs_shared.cu` uses `edge_t` for `row_ptr` and loops.
- **Action**: No changes needed, but ensure any new kernels also use `edge_t`.

## C. Active-Set / Frontier Logic (Partially Present)
- **Status**: [GOOD]
- `bfs_shared.cu` implements `queueToBitmapKernel` and `bfsBottomUpKernel` which logic is identical to the "Active Set" idea.
- **Improvement**: Ensure the "Hybrid Heuristic" (switching between Top-Down and Bottom-Up) is tuned correctly. Currently it switches at `num_nodes / 20`.

## D. SOTA Async Streaming (Present)
- **Status**: [EXCELLENT]
- The code already contains `bfsNodeAlignedStreamedKernel` which implements the advanced "Zero-Copy + Streaming" optimization to handle graphs larger than VRAM.

## E. Proposed Plan
1.  **Implement `json_gpu.cu`**: Match the format of Project 2's new JSON output.
2.  **Update `main`**: Add `--json` and `--benchmark` flags.
3.  **Verify Header Dependencies**: Ensure `json_gpu.h` is included in `bfs_shared.h`.
