#define CUDA_ATOMICS_IMPL
#include "../legacy/v3_shared/bfs_compressed_kernel.cuh"
#include "cuda_common.h"
#include "graph.h"
#include "utils.h"

#define THREAD_QUEUE_LIMIT 32

// Classifier Kernel for Compressed Graph
// Uses byte-length of neighbor list as proxy for degree/workload.
__global__ void classifyCompressedFrontierKernel(
    const node_t *__restrict__ frontier, int frontier_size,
    const edge_t *__restrict__ row_ptr, node_t *__restrict__ q_small,
    int *__restrict__ count_small, node_t *__restrict__ q_large,
    int *__restrict__ count_large) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < frontier_size) {
    node_t u = frontier[tid];
    edge_t start = row_ptr[u];
    edge_t end = row_ptr[u + 1];
    edge_t bytes = end - start;

    if (bytes < THREAD_QUEUE_LIMIT) {
      int pos = atomicAdd(count_small, 1);
      q_small[pos] = u;
    } else {
      int pos = atomicAdd(count_large, 1);
      q_large[pos] = u;
    }
  }
}

BFSResult *solveBFSCompressedAdaptive(CompressedCSRGraph *graph,
                                      node_t source) {
  node_t num_nodes = graph->num_nodes;

  // Allocations
  level_t *d_distances;
  node_t *d_frontier, *d_next_frontier;
  int *d_frontier_size, *d_next_frontier_size;

  // Adaptive Queues
  node_t *d_q_small, *d_q_large;
  int *d_count_small, *d_count_large;

  // Host vars
  int h_frontier_size;
  int h_count_small, h_count_large;

  // 1. Setup Memory
  // Distances
  CUDA_CHECK(cudaMalloc((void **)&d_distances, num_nodes * sizeof(level_t)));
  CUDA_CHECK(cudaMemset(d_distances, UNVISITED, num_nodes * sizeof(level_t)));

  // Frontiers (Double Buffer)
  CUDA_CHECK(cudaMalloc((void **)&d_frontier, num_nodes * sizeof(node_t)));
  CUDA_CHECK(cudaMalloc((void **)&d_next_frontier, num_nodes * sizeof(node_t)));
  CUDA_CHECK(cudaMalloc((void **)&d_frontier_size, sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&d_next_frontier_size, sizeof(int)));

  // Queues (Worst case: all nodes in one queue)
  CUDA_CHECK(cudaMalloc((void **)&d_q_small, num_nodes * sizeof(node_t)));
  CUDA_CHECK(cudaMalloc((void **)&d_q_large, num_nodes * sizeof(node_t)));
  CUDA_CHECK(cudaMalloc((void **)&d_count_small, sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&d_count_large, sizeof(int)));

  // 2. Initialize Source
  node_t start_node = source;
  CUDA_CHECK(cudaMemcpy(d_frontier, &start_node, sizeof(node_t),
                        cudaMemcpyHostToDevice));
  h_frontier_size = 1;
  CUDA_CHECK(cudaMemcpy(d_frontier_size, &h_frontier_size, sizeof(int),
                        cudaMemcpyHostToDevice));

  level_t source_dist = 0;
  CUDA_CHECK(cudaMemcpy(&d_distances[source], &source_dist, sizeof(level_t),
                        cudaMemcpyHostToDevice));

  // Timer
  CudaTimer timer = createTimer();
  float total_ms = 0.0f;

  int level = 0;

  // BFS Loop
  printf("Starting BFS Loop. Frontier Size: %d\n", h_frontier_size);
  while (h_frontier_size > 0) {
    startTimer(&timer);

    // Reset Queue Counts
    CUDA_CHECK(cudaMemset(d_count_small, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count_large, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_next_frontier_size, 0, sizeof(int)));

    // 1. Classify Frontier
    int threads = 256;
    int blocks = (h_frontier_size + threads - 1) / threads;
    classifyCompressedFrontierKernel<<<blocks, threads>>>(
        d_frontier, h_frontier_size, graph->d_row_Ptr, d_q_small, d_count_small,
        d_q_large, d_count_large);

    // Read back counts
    CUDA_CHECK(cudaMemcpy(&h_count_small, d_count_small, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_count_large, d_count_large, sizeof(int),
                          cudaMemcpyDeviceToHost));
    printf("Level %d: Size %d. Small: %d, Large: %d\n", level, h_frontier_size,
           h_count_small, h_count_large);

    // 2. Dispatch Kernels
    if (h_count_small > 0) {
      int sm_blocks = (h_count_small + threads - 1) / threads;
      bfsCompressedThreadKernel<<<sm_blocks, threads>>>(
          graph->d_row_Ptr, graph->d_compressed_col, d_distances, d_q_small,
          h_count_small, d_next_frontier, d_next_frontier_size, level);
    }

    if (h_count_large > 0) {
      int warps_per_block =
          32; // 1024 threads / 32 = 32 warps (matches kernel expectation)
      int lg_blocks = (h_count_large + warps_per_block - 1) / warps_per_block;
      // Note: bfsCompressedWarpKernel assumes 1 Warp per Node in Frontier
      bfsCompressedWarpKernel<<<lg_blocks, 1024>>>(
          graph->d_row_Ptr, graph->d_compressed_col, d_distances, d_q_large,
          h_count_large, d_next_frontier, d_next_frontier_size, level);
    }

    total_ms += stopTimer(&timer);

    // Swap Frontiers
    node_t *temp = d_frontier;
    d_frontier = d_next_frontier;
    d_next_frontier = temp;

    // Update Size
    CUDA_CHECK(cudaMemcpy(&h_frontier_size, d_next_frontier_size, sizeof(int),
                          cudaMemcpyDeviceToHost));
    level++;
  }

  // Result
  BFSResult *res = allocBFSResult(num_nodes, source);
  res->elapsed_ms = total_ms;
  CUDA_CHECK(cudaMemcpy(res->distances, d_distances,
                        num_nodes * sizeof(level_t), cudaMemcpyDeviceToHost));

  // Cleanup
  CUDA_CHECK(cudaFree(d_distances));
  CUDA_CHECK(cudaFree(d_frontier));
  CUDA_CHECK(cudaFree(d_next_frontier));
  CUDA_CHECK(cudaFree(d_frontier_size));
  CUDA_CHECK(cudaFree(d_next_frontier_size));
  CUDA_CHECK(cudaFree(d_q_small));
  CUDA_CHECK(cudaFree(d_q_large));
  CUDA_CHECK(cudaFree(d_count_small));
  CUDA_CHECK(cudaFree(d_count_large));

  return res;
}
