#define CUDA_ATOMICS_IMPL
#include "bfs_adaptive.h"
#include <stdio.h>

#define THREAD_QUEUE_LIMIT 32
#define WARP_QUEUE_LIMIT 1024

// Device globals for queue counters to avoid atomicAdd arguments passing every
// time? Better to pass pointer.

// =============================================================================
// Kernels
// =============================================================================

// 1. Frontier Classification
__global__ void classifyFrontierKernel(
    const node_t *__restrict__ frontier, int frontier_size,
    const edge_t *__restrict__ row_ptr, node_t *__restrict__ q_small,
    int *__restrict__ count_small, node_t *__restrict__ q_medium,
    int *__restrict__ count_medium, node_t *__restrict__ q_large,
    int *__restrict__ count_large) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < frontier_size) {
    node_t u = frontier[tid];
    edge_t start = row_ptr[u];
    edge_t end = row_ptr[u + 1];
    edge_t deg = end - start;

    if (deg < THREAD_QUEUE_LIMIT) {
      int pos = atomicAdd(count_small, 1);
      q_small[pos] = u;
    } else if (deg < WARP_QUEUE_LIMIT) {
      int pos = atomicAdd(count_medium, 1);
      q_medium[pos] = u;
    } else {
      int pos = atomicAdd(count_large, 1);
      q_large[pos] = u;
    }
  }
}

// 2. Thread Kernel (Small Nodes: deg < 32)
// One Thread Per Node
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

    // Serial Neighbor Check
    for (edge_t e = start; e < end; e++) {
      node_t v = col_idx[e];
      level_t old =
          atomicCAS_uint8(&distances[v], UNVISITED, current_level + 1);
      if (old == UNVISITED) {
        int pos = atomicAdd(next_frontier_size, 1);
        next_frontier[pos] = v;
      }
    }
  }
}

// 3. Warp Kernel (Medium Nodes: 32 <= deg < 1024)
// One Warp Per Node
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

    // Coalesced access?
    // Threads in warp iterate from start to end
    for (edge_t e = start + lane_id; e < end; e += WARP_SIZE) {
      node_t v = col_idx[e];
      level_t old =
          atomicCAS_uint8(&distances[v], UNVISITED, current_level + 1);
      if (old == UNVISITED) {
        int pos = atomicAdd(next_frontier_size, 1);
        next_frontier[pos] = v;
      }
    }
  }
}

// 4. Block Kernel (Large Nodes: deg >= 1024)
// One Block Per Node (Persistent-ish or simple grid mapping)
// Launch 'q_size' blocks.
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
    // Block Stride Loop
    for (edge_t e = start + tid; e < end; e += blockDim.x) {
      node_t v = col_idx[e];
      level_t old =
          atomicCAS_uint8(&distances[v], UNVISITED, current_level + 1);
      if (old == UNVISITED) {
        int pos = atomicAdd(next_frontier_size, 1);
        next_frontier[pos] = v;
      }
    }
  }
}

// =============================================================================
// Solver
// =============================================================================

BFSResult *solveBFSAdaptive(CSRGraph *graph, node_t source) {
  node_t num_nodes = graph->num_nodes;

  // Allocations
  level_t *d_distances;
  node_t *d_frontier, *d_next_frontier;
  int *d_frontier_size, *d_next_frontier_size;

  // Queues
  node_t *d_q_small, *d_q_medium, *d_q_large;
  int *d_counts; // [3] : small, medium, large

  CUDA_CHECK(cudaMalloc(&d_distances, num_nodes * sizeof(level_t)));
  CUDA_CHECK(cudaMemset(d_distances, UNVISITED, num_nodes * sizeof(level_t)));

  // Frontier buffers (Ping-Pong)
  CUDA_CHECK(cudaMalloc(&d_frontier, num_nodes * sizeof(node_t)));
  CUDA_CHECK(cudaMalloc(&d_next_frontier, num_nodes * sizeof(node_t)));

  // Queues (Partitioned frontier)
  // Worst case: all nodes in one queue. So we need num_nodes size for each?
  // Memory heavy! 3 * num_nodes * 4B = 12 * N. Friendster N=65M. 780MB. OK.
  CUDA_CHECK(cudaMalloc(&d_q_small, num_nodes * sizeof(node_t)));
  CUDA_CHECK(cudaMalloc(&d_q_medium, num_nodes * sizeof(node_t)));
  CUDA_CHECK(cudaMalloc(&d_q_large, num_nodes * sizeof(node_t)));

  // Counters (Pinned or Device?)
  // Need Host access every iter? Yes, to launch kernels.
  // Use mapped memory or separate implementation.
  int *h_counts; // [3]
  CUDA_CHECK(cudaMallocHost(&h_counts, 3 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_counts, 3 * sizeof(int)));

  CUDA_CHECK(cudaMalloc(&d_next_frontier_size, sizeof(int)));

  // Init Source
  level_t zero = 0;
  CUDA_CHECK(cudaMemcpy(d_distances + source, &zero, sizeof(level_t),
                        cudaMemcpyHostToDevice));

  node_t h_frontier[1] = {source};
  CUDA_CHECK(cudaMemcpy(d_frontier, h_frontier, sizeof(node_t),
                        cudaMemcpyHostToDevice));

  int frontier_size = 1;
  int level = 0;

  CudaTimer timer = createTimer();
  startTimer(&timer);

  while (frontier_size > 0) {
    // 1. Reset Counters
    h_counts[0] = h_counts[1] = h_counts[2] = 0;
    // CUDA_CHECK(cudaMemcpy(d_counts, h_counts, 3 * sizeof(int),
    // cudaMemcpyHostToDevice)); Better: Memset d_counts
    CUDA_CHECK(cudaMemset(d_counts, 0, 3 * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_next_frontier_size, 0, sizeof(int)));

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

    printf("Level %d: Sm=%d, Med=%d, Lrg=%d\n", level, c_small, c_medium,
           c_large);

    // 4. Dispatch Kernels
    if (c_small > 0) {
      int grid = (c_small + 255) / 256;
      bfsThreadKernel<<<grid, 256>>>(
          d_q_small, c_small, graph->d_row_ptr, graph->d_col_idx, d_distances,
          d_next_frontier, d_next_frontier_size, level);
    }

    if (c_medium > 0) {
      // 1 Warp per Node. Block has 8 Warps (256 threads).
      // Grid size needs to cover c_medium warps.
      // Total Warps Needed = c_medium.
      // Warps Per Block = 8.
      int grid = (c_medium + 7) / 8;
      bfsWarpKernel<<<grid, 256>>>(
          d_q_medium, c_medium, graph->d_row_ptr, graph->d_col_idx, d_distances,
          d_next_frontier, d_next_frontier_size, level);
    }

    if (c_large > 0) {
      // One Block Per Node
      bfsBlockKernel<<<c_large, 256>>>(
          d_q_large, c_large, graph->d_row_ptr, graph->d_col_idx, d_distances,
          d_next_frontier, d_next_frontier_size, level);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Update Frontier
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

  // Copy results back
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
  CUDA_CHECK(cudaFreeHost(h_counts));

  return res;
}
