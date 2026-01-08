/**
 * @file bfs_adaptive.cu
 * @brief Version 4.1: Hybrid-Optimized Adaptive BFS Solver
 *
 * Features:
 * - Hierarchical Atomics: Warp-level primitives + shared memory aggregation
 * - Hybrid Direction-Optimization: Top-Down/Bottom-Up switching
 * - Configurable threshold via --bu-threshold CLI option
 */
#define CUDA_ATOMICS_IMPL
#include "bfs_adaptive.h"
#include "bfs_kernels.cuh"
#include <stdio.h>

#define THREAD_QUEUE_LIMIT 32
#define WARP_QUEUE_LIMIT 1024

// =============================================================================
// Kernels
// =============================================================================

/**
 * Frontier Classification Kernel with Hierarchical Atomics
 *
 * Optimization Strategy:
 * 1. Warp-level: Use __ballot_sync + __popc for intra-warp offset calculation
 * 2. Block-level: Shared memory aggregation across warps
 * 3. Global-level: Single thread (tid=0) performs 3 global atomicAdds
 *
 * This reduces global atomics from O(frontier_size) to O(3 * num_blocks)
 */
__global__ void classifyFrontierKernel(
    const node_t *__restrict__ frontier, int frontier_size,
    const edge_t *__restrict__ row_ptr, node_t *__restrict__ q_small,
    int *__restrict__ count_small, node_t *__restrict__ q_medium,
    int *__restrict__ count_medium, node_t *__restrict__ q_large,
    int *__restrict__ count_large) {

  // Shared memory for block-level aggregation
  __shared__ int s_counts[3];  // [small, medium, large] per-block counts
  __shared__ int s_offsets[3]; // Global base offsets after atomicAdd
  __shared__ int
      s_warp_bases[3][8]; // Per-warp base within block (max 8 warps/block)

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  int warps_per_block = blockDim.x / WARP_SIZE;

  // Phase 1: Initialize shared counters
  if (tid < 3)
    s_counts[tid] = 0;
  __syncthreads();

  // Phase 2: Classify nodes and compute warp-level offsets
  int my_category = -1;
  node_t my_node = 0;
  bool has_work = (gid < frontier_size);

  if (has_work) {
    my_node = frontier[gid];
    edge_t deg = row_ptr[my_node + 1] - row_ptr[my_node];
    if (deg < THREAD_QUEUE_LIMIT)
      my_category = 0; // Small
    else if (deg < WARP_QUEUE_LIMIT)
      my_category = 1; // Medium
    else
      my_category = 2; // Large
  }

  // Warp-level ballot for each category
  unsigned int mask_small = __ballot_sync(0xFFFFFFFF, my_category == 0);
  unsigned int mask_medium = __ballot_sync(0xFFFFFFFF, my_category == 1);
  unsigned int mask_large = __ballot_sync(0xFFFFFFFF, my_category == 2);

  // Count per warp using __popc
  int warp_count_small = __popc(mask_small);
  int warp_count_medium = __popc(mask_medium);
  int warp_count_large = __popc(mask_large);

  // Calculate intra-warp offset using __popc on lower lanes mask
  unsigned int lower_mask = (1u << lane_id) - 1;
  int intra_warp_offset = -1;
  if (my_category == 0)
    intra_warp_offset = __popc(mask_small & lower_mask);
  else if (my_category == 1)
    intra_warp_offset = __popc(mask_medium & lower_mask);
  else if (my_category == 2)
    intra_warp_offset = __popc(mask_large & lower_mask);

  // Phase 3: First lane of each warp atomicAdds to shared to get warp's base
  // within block
  if (lane_id == 0) {
    if (warp_count_small > 0)
      s_warp_bases[0][warp_id] = atomicAdd(&s_counts[0], warp_count_small);
    if (warp_count_medium > 0)
      s_warp_bases[1][warp_id] = atomicAdd(&s_counts[1], warp_count_medium);
    if (warp_count_large > 0)
      s_warp_bases[2][warp_id] = atomicAdd(&s_counts[2], warp_count_large);
  }
  __syncthreads();

  // Phase 4: First thread performs global atomicAdds to get block's global base
  if (tid == 0) {
    if (s_counts[0] > 0)
      s_offsets[0] = atomicAdd(count_small, s_counts[0]);
    else
      s_offsets[0] = 0;
    if (s_counts[1] > 0)
      s_offsets[1] = atomicAdd(count_medium, s_counts[1]);
    else
      s_offsets[1] = 0;
    if (s_counts[2] > 0)
      s_offsets[2] = atomicAdd(count_large, s_counts[2]);
    else
      s_offsets[2] = 0;
  }
  __syncthreads();

  // Phase 5: Write nodes to output queues
  if (has_work && my_category >= 0) {
    int global_idx = s_offsets[my_category] +
                     s_warp_bases[my_category][warp_id] + intra_warp_offset;
    if (my_category == 0)
      q_small[global_idx] = my_node;
    else if (my_category == 1)
      q_medium[global_idx] = my_node;
    else
      q_large[global_idx] = my_node;
  }
}

// 2. Thread Kernel (Small Nodes: deg < 32)
__global__ void bfsThreadKernel(const node_t *__restrict__ q, int q_size,
                                const edge_t *__restrict__ row_ptr,
                                const node_t *__restrict__ col_idx,
                                level_t *__restrict__ distances,
                                node_t *__restrict__ next_frontier,
                                int *__restrict__ next_frontier_size,
                                level_t current_level) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < q_size) {
    node_t u = q[tid];
    edge_t start = row_ptr[u];
    edge_t end = row_ptr[u + 1];

    for (edge_t e = start; e < end; e++) {
      node_t v = col_idx[e];
      level_t old = atomicCAS(&distances[v], UNVISITED, current_level + 1);
      if (old == UNVISITED) {
        int pos = atomicAdd(next_frontier_size, 1);
        next_frontier[pos] = v;
      }
    }
  }
}

// 3. Warp Kernel (Medium Nodes: 32 <= deg < 1024)
__global__ void bfsWarpKernel(const node_t *__restrict__ q, int q_size,
                              const edge_t *__restrict__ row_ptr,
                              const node_t *__restrict__ col_idx,
                              level_t *__restrict__ distances,
                              node_t *__restrict__ next_frontier,
                              int *__restrict__ next_frontier_size,
                              level_t current_level) {
  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  int global_warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id;

  if (global_warp_id < q_size) {
    node_t u = q[global_warp_id];
    edge_t start = row_ptr[u];
    edge_t end = row_ptr[u + 1];

    for (edge_t e = start + lane_id; e < end; e += WARP_SIZE) {
      node_t v = col_idx[e];
      level_t old = atomicCAS(&distances[v], UNVISITED, current_level + 1);
      if (old == UNVISITED) {
        int pos = atomicAdd(next_frontier_size, 1);
        next_frontier[pos] = v;
      }
    }
  }
}

// 4. Block Kernel (Large Nodes: deg >= 1024)
__global__ void bfsBlockKernel(const node_t *__restrict__ q, int q_size,
                               const edge_t *__restrict__ row_ptr,
                               const node_t *__restrict__ col_idx,
                               level_t *__restrict__ distances,
                               node_t *__restrict__ next_frontier,
                               int *__restrict__ next_frontier_size,
                               level_t current_level) {
  node_t u_idx = blockIdx.x;
  if (u_idx < q_size) {
    node_t u = q[u_idx];
    edge_t start = row_ptr[u];
    edge_t end = row_ptr[u + 1];

    int tid = threadIdx.x;
    for (edge_t e = start + tid; e < end; e += blockDim.x) {
      node_t v = col_idx[e];
      level_t old = atomicCAS(&distances[v], UNVISITED, current_level + 1);
      if (old == UNVISITED) {
        int pos = atomicAdd(next_frontier_size, 1);
        next_frontier[pos] = v;
      }
    }
  }
}

// =============================================================================
// Hybrid-Optimized Solver
// =============================================================================

BFSResult *solveBFSAdaptive(CSRGraph *graph, node_t source) {
  return solveBFSAdaptiveWithThreshold(graph, source, 20);
}

BFSResult *solveBFSAdaptiveWithThreshold(CSRGraph *graph, node_t source,
                                         int bu_threshold_divisor) {
  node_t num_nodes = graph->num_nodes;

  // Allocations
  level_t *d_distances;
  node_t *d_frontier, *d_next_frontier;
  int *d_next_frontier_size;

  // Queues for classification
  node_t *d_q_small, *d_q_medium, *d_q_large;
  int *d_counts; // [3] : small, medium, large

  // Bitmaps for Bottom-Up traversal
  int bitmap_ints = (num_nodes + 31) / 32;
  unsigned int *d_frontier_bitmap, *d_visited_bitmap;

  CUDA_CHECK(cudaMalloc(&d_distances, num_nodes * sizeof(level_t)));
  CUDA_CHECK(cudaMemset(d_distances, UNVISITED, num_nodes * sizeof(level_t)));

  CUDA_CHECK(cudaMalloc(&d_frontier, num_nodes * sizeof(node_t)));
  CUDA_CHECK(cudaMalloc(&d_next_frontier, num_nodes * sizeof(node_t)));

  CUDA_CHECK(cudaMalloc(&d_q_small, num_nodes * sizeof(node_t)));
  CUDA_CHECK(cudaMalloc(&d_q_medium, num_nodes * sizeof(node_t)));
  CUDA_CHECK(cudaMalloc(&d_q_large, num_nodes * sizeof(node_t)));

  int *h_counts;
  CUDA_CHECK(cudaMallocHost(&h_counts, 3 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_counts, 3 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_next_frontier_size, sizeof(int)));

  // Bitmap allocations
  CUDA_CHECK(
      cudaMalloc(&d_frontier_bitmap, bitmap_ints * sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&d_visited_bitmap, bitmap_ints * sizeof(unsigned int)));

  // Init Source
  level_t zero = 0;
  CUDA_CHECK(cudaMemcpy(d_distances + source, &zero, sizeof(level_t),
                        cudaMemcpyHostToDevice));

  node_t h_frontier[1] = {source};
  CUDA_CHECK(cudaMemcpy(d_frontier, h_frontier, sizeof(node_t),
                        cudaMemcpyHostToDevice));

  int frontier_size = 1;
  int level = 0;

  // Direction-switching threshold: switch to Bottom-Up when frontier >
  // N/divisor
  int bu_threshold = num_nodes / bu_threshold_divisor;

  CudaTimer timer = createTimer();
  startTimer(&timer);

  while (frontier_size > 0) {
    CUDA_CHECK(cudaMemset(d_next_frontier_size, 0, sizeof(int)));

    if (frontier_size > bu_threshold) {
      // =========================================================
      // BOTTOM-UP PATH (Large Frontier)
      // =========================================================
      printf("Level %d: [BOTTOM-UP] frontier=%d (threshold=%d)\n", level,
             frontier_size, bu_threshold);

      // 1. Convert Queue -> Bitmap
      int grid_bitmap = (bitmap_ints + 255) / 256;
      clearBitmapKernel<<<grid_bitmap, 256>>>(d_frontier_bitmap, bitmap_ints);

      int grid_queue = (frontier_size + 255) / 256;
      queueToBitmapKernel<<<grid_queue, 256>>>(d_frontier, frontier_size,
                                               d_frontier_bitmap);
      CUDA_CHECK(cudaDeviceSynchronize());

      // 2. Build Visited Bitmap from distances
      clearBitmapKernel<<<grid_bitmap, 256>>>(d_visited_bitmap, bitmap_ints);
      int grid_nodes = (num_nodes + 255) / 256;
      generateVisitedBitmapKernel<<<grid_nodes, 256>>>(
          d_distances, d_visited_bitmap, num_nodes);
      CUDA_CHECK(cudaDeviceSynchronize());

      // 3. Bottom-Up Traversal
      int block_size = WARPS_PER_BLOCK * WARP_SIZE;
      int grid_warps = (num_nodes + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
      bfsBottomUpWarpKernel<<<grid_warps, block_size>>>(
          graph->d_row_ptr, graph->d_col_idx, d_distances, d_frontier_bitmap,
          d_visited_bitmap, d_next_frontier_size, level, num_nodes);
      CUDA_CHECK(cudaDeviceSynchronize());

      // 4. Regenerate Queue from distances (for next iteration)
      CUDA_CHECK(cudaMemset(d_next_frontier_size, 0, sizeof(int)));
      distancesToQueueKernel<<<grid_nodes, 256>>>(
          d_distances, num_nodes, d_next_frontier, d_next_frontier_size,
          level + 1);
      CUDA_CHECK(cudaDeviceSynchronize());

    } else {
      // =========================================================
      // TOP-DOWN ADAPTIVE PATH (Small Frontier)
      // =========================================================

      // 1. Reset Counters
      CUDA_CHECK(cudaMemset(d_counts, 0, 3 * sizeof(int)));

      // 2. Classify Frontier -> Queues
      int numBlocks = (frontier_size + 255) / 256;
      classifyFrontierKernel<<<numBlocks, 256>>>(
          d_frontier, frontier_size, graph->d_row_ptr, d_q_small, d_counts + 0,
          d_q_medium, d_counts + 1, d_q_large, d_counts + 2);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      // 3. Read back counters
      CUDA_CHECK(cudaMemcpy(h_counts, d_counts, 3 * sizeof(int),
                            cudaMemcpyDeviceToHost));
      int c_small = h_counts[0];
      int c_medium = h_counts[1];
      int c_large = h_counts[2];

      printf("Level %d: [TOP-DOWN] Sm=%d, Med=%d, Lrg=%d\n", level, c_small,
             c_medium, c_large);

      // 4. Dispatch Kernels
      if (c_small > 0) {
        int grid = (c_small + 255) / 256;
        bfsThreadKernel<<<grid, 256>>>(
            d_q_small, c_small, graph->d_row_ptr, graph->d_col_idx, d_distances,
            d_next_frontier, d_next_frontier_size, level);
      }

      if (c_medium > 0) {
        int grid = (c_medium + 7) / 8;
        bfsWarpKernel<<<grid, 256>>>(
            d_q_medium, c_medium, graph->d_row_ptr, graph->d_col_idx,
            d_distances, d_next_frontier, d_next_frontier_size, level);
      }

      if (c_large > 0) {
        bfsBlockKernel<<<c_large, 256>>>(
            d_q_large, c_large, graph->d_row_ptr, graph->d_col_idx, d_distances,
            d_next_frontier, d_next_frontier_size, level);
      }

      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Update Frontier
    CUDA_CHECK(cudaMemcpy(&frontier_size, d_next_frontier_size, sizeof(int),
                          cudaMemcpyDeviceToHost));

    node_t *temp = d_frontier;
    d_frontier = d_next_frontier;
    d_next_frontier = temp;

    level++;
  }

  float elapsed = stopTimer(&timer);

  BFSResult *res = new BFSResult;
  res->elapsed_ms = elapsed;
  res->num_nodes = num_nodes;
  res->source = source;
  res->distances = new level_t[num_nodes];
  res->parents = nullptr;

  CUDA_CHECK(cudaMemcpy(res->distances, d_distances,
                        num_nodes * sizeof(level_t), cudaMemcpyDeviceToHost));

  // Cleanup
  CUDA_CHECK(cudaFree(d_distances));
  CUDA_CHECK(cudaFree(d_frontier));
  CUDA_CHECK(cudaFree(d_next_frontier));
  CUDA_CHECK(cudaFree(d_q_small));
  CUDA_CHECK(cudaFree(d_q_medium));
  CUDA_CHECK(cudaFree(d_q_large));
  CUDA_CHECK(cudaFree(d_counts));
  CUDA_CHECK(cudaFree(d_next_frontier_size));
  CUDA_CHECK(cudaFree(d_frontier_bitmap));
  CUDA_CHECK(cudaFree(d_visited_bitmap));
  CUDA_CHECK(cudaFreeHost(h_counts));

  return res;
}
