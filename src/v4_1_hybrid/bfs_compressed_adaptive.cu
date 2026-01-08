#define CUDA_ATOMICS_IMPL
#include "../legacy/v3_shared/bfs_compressed_kernel.cuh"
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
__global__ void classifyCompressedFrontierKernel(
    const node_t *__restrict__ frontier, int frontier_size,
    const edge_t *__restrict__ row_ptr, node_t *__restrict__ q_small,
    int *__restrict__ count_small, node_t *__restrict__ q_large,
    int *__restrict__ count_large) {

  // Shared memory for block-level aggregation
  __shared__ int s_counts[2];        // [0]=small, [1]=large
  __shared__ int s_offsets[2];       // Global base offsets
  __shared__ int s_warp_bases[2][8]; // Per-warp base within block (max 8 warps)

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
// Decodes neighbor lists on the fly to find *any* parent in the frontier.
__global__ void
bfsCompressedBottomUpKernel(const edge_t *__restrict__ row_ptr,
                            const uint8_t *__restrict__ compressed_col,
                            level_t *__restrict__ distances,
                            const unsigned int *__restrict__ frontier_bitmap,
                            const unsigned int *__restrict__ visited_bitmap,
                            int *__restrict__ next_frontier_size,
                            level_t current_level, node_t num_nodes) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;

  // Warp-strided loop over nodes to ensure coalescing if possible,
  // though bottom-up usually maps 1 thread/node or 1 warp/node.
  // Given the decoding cost, 1 thread per node is simple but divergent.
  // 1 Warp per node is better for bandwidth but complex to decode in parallel.
  // We stick to the standard V3 Bottom-Up pattern: 1 Thread per Node (strided).

  // Actually, V3 bfsBottomUpWarpKernel from bfs_kernels.cuh uses 1 Warp per
  // Chunk of nodes? Let's stick to a simple 1-thread-per-node approach for the
  // compressed version first, utilizing the L2 cache for the byte stream.

  for (int u = tid; u < num_nodes; u += gridDim.x * blockDim.x) {
    // Optimization: Use bitmap cache instead of global distances array (32x B/W
    // savings)
    bool is_visited = (visited_bitmap[u / 32] & (1 << (u % 32)));
    if (!is_visited) {
      // Check neighbors
      edge_t curr = row_ptr[u];
      edge_t end = row_ptr[u + 1];
      node_t prev_neighbor = 0;

      bool found_parent = false;

      while (curr < end) {
        // Decode Varint
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

        // Check overlap with frontier bitmap
        // Optimize: check visited bitmap first? No, we checking if we ARE
        // unvisited (line 35). We check if neighbor v is in frontier.

        bool is_frontier = (frontier_bitmap[v / 32] & (1 << (v % 32)));
        if (is_frontier) {
          distances[u] = current_level + 1;
          // atomicAdd(next_frontier_size, 1); // REMOVED: Redundant bottleneck.
          // Count is recalculated in distancesToQueueKernel.
          found_parent = true;
          break;
        }
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
  unsigned int *d_frontier_bitmap, *d_visited_bitmap;

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
  CUDA_CHECK(cudaMalloc(&d_visited_bitmap, bitmap_ints * sizeof(unsigned int)));

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

  // Tuning: Aggressive Bottom-Up for Compressed
  // Decode overhead suggests we want to avoid Edge-Scanning large frontiers
  // even more.
  // V4.3 Tuning: N/26 (~2.5M). Strategy:
  // - Level 2 (2.1M): Run Top-Down (Fast expansion, high degree)
  // - Level 5 (3.0M): Run Bottom-Up (Sparsae/Dense transition, save work)
  int bu_threshold = num_nodes / 26;

  printf("Starting Compressed BFS (Hybrid V4.3). Threshold: %d\n",
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

      // 2. Visited Bitmap (Required for Optimized Kernel)
      clearBitmapKernel<<<(bitmap_ints + 255) / 256, 256>>>(d_visited_bitmap,
                                                            bitmap_ints);
      generateVisitedBitmapKernel<<<(num_nodes + 255) / 256, 256>>>(
          d_distances, d_visited_bitmap, num_nodes);
      CUDA_CHECK(cudaDeviceSynchronize());

      // 3. Kernel
      int threads = 256;
      int blocks = (num_nodes + threads - 1) / threads;
      bfsCompressedBottomUpKernel<<<blocks, threads>>>(
          graph->d_row_Ptr, graph->d_compressed_col, d_distances,
          d_frontier_bitmap, d_visited_bitmap, d_next_frontier_size, level,
          num_nodes);
      CUDA_CHECK(cudaDeviceSynchronize());

      // 4. Regenerate Queue
      // We reuse the standard distancesToQueueKernel
      CUDA_CHECK(cudaMemset(d_next_frontier_size, 0,
                            sizeof(int))); // clear again to count for queue
      distancesToQueueKernel<<<(num_nodes + 255) / 256, 256>>>(
          d_distances, num_nodes, d_next_frontier, d_next_frontier_size,
          level + 1);
      CUDA_CHECK(cudaDeviceSynchronize());

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
  CUDA_CHECK(cudaFree(d_visited_bitmap));

  return res;
}
