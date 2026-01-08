#define CUDA_ATOMICS_IMPL
// Restoration of Legacy Kernels (Best Performance: 996ms)
// Restoration of Legacy Kernels (Best Performance: 996ms)
#include "../legacy/v3_shared/bfs_compressed_kernel.cuh" // Legacy Prototypes
#define BFS_KERNELS_SKIP_DEFINITIONS
#include "bfs_kernels.cuh"
#include "cuda_common.h"
#include "graph.h"
#include "utils.h"

#define THREAD_QUEUE_LIMIT 32

// Classifier Kernel for Compressed Graph
// Classifier Kernel for Compressed Graph with Hierarchical Atomics (V4.2)
// Optimization: Warp-level aggregation + Shared atomics -> Reduces global
// contention
// Deep-Inline Kernels Removed (Performance Regression)
// Using Legacy Kernels from bfs_compressed_kernels.cu

// Classifier Kernel for Compressed Graph
// Classifier Kernel for Compressed Graph with Hierarchical Atomics (V4.2)
// Optimization: Warp-level aggregation + Shared atomics -> Reduces global
// contention
__global__ void classifyCompressedFrontierKernel(
    const node_t *__restrict__ frontier, int frontier_size,
    const edge_t *__restrict__ row_ptr, node_t *__restrict__ q_small,
    int *__restrict__ count_small, node_t *__restrict__ q_large,
    int *__restrict__ count_large) {

  // Shared memory for block-level aggregation
  __shared__ int s_counts[2];  // [0]=small, [1]=large
  __shared__ int s_offsets[2]; // Global base offsets
  // V5.3: Hardening - Support up to 32 warps (1024 threads)
  __shared__ int s_warp_bases[2][32];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  int warp_id = tid / 32;
  int lane_id = tid % 32;

  // Phase 1: Initialize shared counters
  if (tid < 2)
    s_counts[tid] = 0;
  __syncthreads();

  // Phase 2: Classify nodes
  int my_category = -1;
  node_t my_node = 0;

  if (gid < frontier_size) {
    my_node = frontier[gid];
    edge_t start = row_ptr[my_node];
    edge_t end = row_ptr[my_node + 1];
    edge_t bytes = end - start;

    if (bytes < THREAD_QUEUE_LIMIT)
      my_category = 0; // Small
    else
      my_category = 1; // Large
  }

  // Phase 3: Warp-level aggregation
  unsigned int mask_small = __ballot_sync(0xFFFFFFFF, my_category == 0);
  unsigned int mask_large = __ballot_sync(0xFFFFFFFF, my_category == 1);

  int warp_count_small = __popc(mask_small);
  int warp_count_large = __popc(mask_large);

  // Intra-warp offset
  unsigned int lower_mask = (1u << lane_id) - 1;
  int intra_warp_offset = -1;
  if (my_category == 0)
    intra_warp_offset = __popc(mask_small & lower_mask);
  else if (my_category == 1)
    intra_warp_offset = __popc(mask_large & lower_mask);

  // Phase 4: Warp leaders add to shared memory
  if (lane_id == 0) {
    if (warp_count_small > 0)
      s_warp_bases[0][warp_id] = atomicAdd(&s_counts[0], warp_count_small);
    if (warp_count_large > 0)
      s_warp_bases[1][warp_id] = atomicAdd(&s_counts[1], warp_count_large);
  }
  __syncthreads();

  // Phase 5: Block leader adds to global memory
  if (tid == 0) {
    if (s_counts[0] > 0)
      s_offsets[0] = atomicAdd(count_small, s_counts[0]);
    else
      s_offsets[0] = 0;

    if (s_counts[1] > 0)
      s_offsets[1] = atomicAdd(count_large, s_counts[1]);
    else
      s_offsets[1] = 0;
  }
  __syncthreads();

  // Phase 6: Write to global queues
  if (gid < frontier_size && my_category >= 0) {
    int global_idx = s_offsets[my_category] +
                     s_warp_bases[my_category][warp_id] + intra_warp_offset;
    if (my_category == 0)
      q_small[global_idx] = my_node;
    else
      q_large[global_idx] = my_node;
  }
}

// Bottom-Up Kernel for Compressed Graph
// Optimized V5.2: Direct Queue Emission (Warp-Aggregated)
// Removes O(N) scan after this kernel.
__global__ void
bfsCompressedBottomUpKernel(const edge_t *__restrict__ row_ptr,
                            const uint8_t *__restrict__ compressed_col,
                            level_t *__restrict__ distances,
                            const unsigned int *__restrict__ frontier_bitmap,
                            int *__restrict__ next_frontier_size,
                            node_t *__restrict__ next_frontier, // Added
                            level_t current_level, node_t num_nodes) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int lane_id = tid % 32;

  // Stride over nodes
  for (int u = tid; u < num_nodes; u += gridDim.x * blockDim.x) {
    bool found = false;

    if (distances[u] == UNVISITED) {
      // Check neighbors
      edge_t curr = row_ptr[u];
      edge_t end = row_ptr[u + 1];
      node_t prev_neighbor = 0;

      while (curr < end) {
        // Decode Varint (Serial)
        node_t val = 0;
        int shift = 0;
        uint8_t byte;
        do {
          byte = compressed_col[curr++];
          val |= (node_t)(byte & 127) << shift;
          shift += 7;
        } while (byte & 128);

        node_t v = prev_neighbor + val;
        prev_neighbor = v;

        // Check frontier bitmap
        if (frontier_bitmap[v / 32] & (1 << (v % 32))) {
          distances[u] = current_level + 1;
          found = true;
          break; // Found parent, stop
        }
      }
    }

    // Warp-Aggregated Queue Append
    // 1. Ballot valid threads
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, found);

    // 2. Compute popcount
    int pop_count = __popc(ballot);

    // 3. Leader reserves space
    if (pop_count > 0) {
      int base_idx = 0;
      if (lane_id == 0) {
        base_idx = atomicAdd(next_frontier_size, pop_count);
      }
      base_idx = __shfl_sync(0xFFFFFFFF, base_idx, 0);

      // 4. Compute local offset and store
      if (found) {
        unsigned int mask = (1u << lane_id) - 1;
        int local_offset = __popc(ballot & mask);
        next_frontier[base_idx + local_offset] = u;
      }
    }
  }
}

BFSResult *solveBFSCompressedAdaptive(CompressedCSRGraph *graph,
                                      node_t source) {
  node_t num_nodes = graph->num_nodes;

  // Allocations
  level_t *d_distances;
  node_t *d_frontier, *d_next_frontier;
  int *d_frontier_size,
      *d_next_frontier_size; /* d_frontier_size unused in hybrid? */
  int *d_next_frontier_cnt;  // Pointer alias for clarity

  // Adaptive Queues
  node_t *d_q_small, *d_q_large;
  int *d_count_small, *d_count_large;

  // Bitmaps for Bottom-Up
  int bitmap_ints = (num_nodes + 31) / 32;
  unsigned int *d_frontier_bitmap; // d_visited_bitmap removed (V5.3)

  // Host vars
  int h_frontier_size;
  int h_count_small, h_count_large;

  // 1. Setup Memory
  // Distances
  CUDA_CHECK(cudaMalloc((void **)&d_distances, num_nodes * sizeof(level_t)));
  CUDA_CHECK(cudaMemset(d_distances, UNVISITED, num_nodes * sizeof(level_t)));

  // Frontiers
  CUDA_CHECK(cudaMalloc((void **)&d_frontier, num_nodes * sizeof(node_t)));
  CUDA_CHECK(cudaMalloc((void **)&d_next_frontier, num_nodes * sizeof(node_t)));
  CUDA_CHECK(cudaMalloc((void **)&d_next_frontier_size, sizeof(int)));
  d_next_frontier_cnt = d_next_frontier_size;

  // Queues
  CUDA_CHECK(cudaMalloc((void **)&d_q_small, num_nodes * sizeof(node_t)));
  CUDA_CHECK(cudaMalloc((void **)&d_q_large, num_nodes * sizeof(node_t)));
  CUDA_CHECK(cudaMalloc((void **)&d_count_small, sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&d_count_large, sizeof(int)));

  // Bitmaps
  CUDA_CHECK(
      cudaMalloc(&d_frontier_bitmap, bitmap_ints * sizeof(unsigned int)));
  // d_visited_bitmap removed

  // 2. Initialize Source
  node_t start_node = source;
  CUDA_CHECK(cudaMemcpy(d_frontier, &start_node, sizeof(node_t),
                        cudaMemcpyHostToDevice));
  h_frontier_size = 1;
  // Initialize distances
  level_t source_dist = 0;
  CUDA_CHECK(cudaMemcpy(&d_distances[source], &source_dist, sizeof(level_t),
                        cudaMemcpyHostToDevice));

  // Timer
  CudaTimer timer = createTimer();
  startTimer(&timer); // Record TOTAL time including setup overhead if desired,
                      // matching adaptive

  int level = 0;

  // Tuning: Aggressive Bottom-Up for Compressed (V4.3 Best Speed: 996ms)
  // Bu_Threshold = N/26 (2.5M) forces L5 (3M) and above to Bottom-Up.
  // This avoids the slow Top-Down decoding for large frontiers.
  int bu_threshold = num_nodes / 26;

  printf("Starting Compressed BFS (Hybrid V5.1). Threshold: %d\n",
         bu_threshold);

  while (h_frontier_size > 0) {
    CUDA_CHECK(cudaMemset(d_next_frontier_size, 0, sizeof(int)));

    if (h_frontier_size > bu_threshold) {
      // === BOTTOM-UP COMPRESSED ===
      printf("Level %d: [BOTTOM-UP] frontier=%d\n", level, h_frontier_size);

      // 1. Convert Queue to Bitmap
      clearBitmapKernel<<<(bitmap_ints + 255) / 256, 256>>>(d_frontier_bitmap,
                                                            bitmap_ints);
      queueToBitmapKernel<<<(h_frontier_size + 255) / 256, 256>>>(
          d_frontier, h_frontier_size, d_frontier_bitmap);
      CUDA_CHECK(cudaDeviceSynchronize());

      // 2. Visited Bitmap (Skipped: Direct check is faster for compressed)
      // clearBitmapKernel... (Removed)
      // generateVisitedBitmapKernel... (Removed)

      // 3. Kernel
      // 3. Kernel (Now emits queue directly)
      int threads = 256;
      int blocks = (num_nodes + threads - 1) / threads;
      bfsCompressedBottomUpKernel<<<blocks, threads>>>(
          graph->d_row_Ptr, graph->d_compressed_col, d_distances,
          d_frontier_bitmap, d_next_frontier_size, d_next_frontier, level,
          num_nodes);

      // Removed: atomicAdd contention reduced via Warp Aggregation
      // Removed: distancesToQueueKernel (O(N) Scan eliminated)

      // Sync moved to end of loop or removed if not needed?
      // We need to sync before swapping if we read frontier size on host.
      // But we can check for errors.
      // cudaDeviceSynchronize(); // Optimized: Let it overlap or sync later.

    } else {
      // === TOP-DOWN COMPRESSED ===
      // Reset Queue Counts
      CUDA_CHECK(cudaMemset(d_count_small, 0, sizeof(int)));
      CUDA_CHECK(cudaMemset(d_count_large, 0, sizeof(int)));

      // 1. Classify
      int threads = 256;
      int blocks = (h_frontier_size + threads - 1) / threads;
      classifyCompressedFrontierKernel<<<blocks, threads>>>(
          d_frontier, h_frontier_size, graph->d_row_Ptr, d_q_small,
          d_count_small, d_q_large, d_count_large);

      // Read back
      CUDA_CHECK(cudaMemcpy(&h_count_small, d_count_small, sizeof(int),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(&h_count_large, d_count_large, sizeof(int),
                            cudaMemcpyDeviceToHost));
      printf("Level %d: [TOP-DOWN] Size %d. Sm: %d, Lg: %d\n", level,
             h_frontier_size, h_count_small, h_count_large);

      // 2. Dispatch
      if (h_count_small > 0) {
        int sm_blocks = (h_count_small + threads - 1) / threads;
        bfsCompressedThreadKernel<<<sm_blocks, threads>>>(
            graph->d_row_Ptr, graph->d_compressed_col, d_distances, d_q_small,
            h_count_small, d_next_frontier, d_next_frontier_size, level);
      }
      if (h_count_large > 0) {
        int warps_per_block = 32;
        int lg_blocks = (h_count_large + warps_per_block - 1) / warps_per_block;
        bfsCompressedWarpKernel<<<lg_blocks, 1024>>>(
            graph->d_row_Ptr, graph->d_compressed_col, d_distances, d_q_large,
            h_count_large, d_next_frontier, d_next_frontier_size, level);
      }
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Update size
    CUDA_CHECK(cudaMemcpy(&h_frontier_size, d_next_frontier_cnt, sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Swap
    node_t *temp = d_frontier;
    d_frontier = d_next_frontier;
    d_next_frontier = temp;

    level++;
  }

  // Result
  float elapsed = stopTimer(&timer);
  BFSResult *res = allocBFSResult(num_nodes, source);
  res->elapsed_ms = elapsed;
  CUDA_CHECK(cudaMemcpy(res->distances, d_distances,
                        num_nodes * sizeof(level_t), cudaMemcpyDeviceToHost));

  // Cleanup
  CUDA_CHECK(cudaFree(d_distances));
  CUDA_CHECK(cudaFree(d_frontier));
  CUDA_CHECK(cudaFree(d_next_frontier));
  CUDA_CHECK(cudaFree(d_next_frontier_size));
  CUDA_CHECK(cudaFree(d_q_small));
  CUDA_CHECK(cudaFree(d_q_large));
  CUDA_CHECK(cudaFree(d_count_small));
  CUDA_CHECK(cudaFree(d_count_large));
  CUDA_CHECK(cudaFree(d_frontier_bitmap));
  // CUDA_CHECK(cudaFree(d_visited_bitmap)); removed

  return res;
}
