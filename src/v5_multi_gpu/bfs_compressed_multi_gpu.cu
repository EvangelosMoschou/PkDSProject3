#define BFS_KERNELS_SKIP_DEFINITIONS
#include "bfs_compressed_kernel.cuh"
#include "bfs_kernels.cuh"
#include "bfs_multi_gpu.h"
#include "cuda_common.h"
#include "graph.h"
#include "utils.h"
#include <algorithm>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
// Optimized V5.4: Warp-Cooperative Scanning + Direct Queue Emission
__global__ void bfsCompressedBottomUpKernel(
    const edge_t *__restrict__ row_ptr,
    const uint8_t *__restrict__ compressed_col, level_t *__restrict__ distances,
    const unsigned int *__restrict__ frontier_bitmap,
    int *__restrict__ next_frontier_size, node_t *__restrict__ next_frontier,
    level_t current_level, node_t num_nodes) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;

  // Let each warp handle one node to better utilize memory bandwidth and decode
  // efficiently
  for (int u = warp_id; u < num_nodes; u += (gridDim.x * blockDim.x) / 32) {
    if (distances[u] == UNVISITED) {
      edge_t curr = row_ptr[u];
      edge_t end = row_ptr[u + 1];
      node_t prev_neighbor = 0;
      bool found = false;

      // Thread 0 decodes, everyone checks (or just thread 0 since variables are
      // local)
      if (lane_id == 0) {
        while (curr < end) {
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

          if (frontier_bitmap[v / 32] & (1u << (v % 32))) {
            found = true;
            break;
          }
        }
      }

      // Broadcast found status
      found = __shfl_sync(0xFFFFFFFF, found, 0);

      // Only thread 0 updates distance and queue
      if (found && lane_id == 0) {
        distances[u] = current_level + 1;
        int pos = atomicAdd(next_frontier_size, 1);
        next_frontier[pos] = u;
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
  int level = 0;

  // 1.5 Setup Device Pointers for Compressed Graph (MANDATORY)
  setupCompressedGraphDevice(graph);
  printf("DEBUG: Graph Device Pointers: row_Ptr=%p, col=%p\n",
         (void *)graph->d_row_Ptr, (void *)graph->d_compressed_col);
  if (!graph->d_row_Ptr || !graph->d_compressed_col) {
    fprintf(stderr, "FATAL: Compressed graph device pointers are NULL!\n");
    exit(1);
  }

  // Timer
  startTimer(&timer);

  // Since graph was symmetrized during .csrbin generation, Bottom-Up is now
  // safe.
  int bu_threshold = num_nodes / 20;

  printf("Starting Compressed BFS (Adaptive). Threshold: %d\n", bu_threshold);

  while (h_frontier_size > 0) {
    CUDA_CHECK(cudaMemset(d_next_frontier_size, 0, sizeof(int)));

    if (h_frontier_size > bu_threshold) {
      // BOTTOM-UP
      int grid_bitmap = (bitmap_ints + 255) / 256;
      clearBitmapKernel<<<grid_bitmap, 256>>>(d_frontier_bitmap, bitmap_ints);
      int grid_queue = (h_frontier_size + 255) / 256;
      queueToBitmapKernel<<<grid_queue, 256>>>(d_frontier, h_frontier_size,
                                               d_frontier_bitmap);
      CUDA_CHECK(cudaDeviceSynchronize());

      dim3 blocks = (num_nodes + 255) / 256;
      bfsCompressedBottomUpKernel<<<blocks, 256>>>(
          graph->d_row_Ptr, graph->d_compressed_col, d_distances,
          d_frontier_bitmap, d_next_frontier_size, d_next_frontier, level,
          num_nodes);
      CUDA_CHECK(cudaDeviceSynchronize());
    } else {
      // TOP-DOWN Adaptive
      CUDA_CHECK(cudaMemset(d_count_small, 0, sizeof(int)));
      CUDA_CHECK(cudaMemset(d_count_large, 0, sizeof(int)));

      dim3 threads(1024);
      dim3 blocks((h_frontier_size + threads.x - 1) / threads.x);

      classifyCompressedFrontierKernel<<<blocks, threads>>>(
          d_frontier, h_frontier_size, graph->d_row_Ptr, d_q_small,
          d_count_small, d_q_large, d_count_large);

      CUDA_CHECK(cudaMemcpy(&h_count_small, d_count_small, sizeof(int),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(&h_count_large, d_count_large, sizeof(int),
                            cudaMemcpyDeviceToHost));

      if (h_count_small > 0) {
        int g_small = (h_count_small + 255) / 256;
        bfsCompressedThreadKernel<<<g_small, 256>>>(
            graph->d_row_Ptr, graph->d_compressed_col, d_distances, d_q_small,
            h_count_small, d_next_frontier, d_next_frontier_size, level);
      }
      if (h_count_large > 0) {
        bfsCompressedWarpKernel<<<h_count_large, 256>>>(
            graph->d_row_Ptr, graph->d_compressed_col, d_distances, d_q_large,
            h_count_large, d_next_frontier, d_next_frontier_size, level);
      }
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Update size and swap
    CUDA_CHECK(cudaMemcpy(&h_frontier_size, d_next_frontier_size, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(d_next_frontier_size, 0, sizeof(int)));

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
// =============================================================================
// Simulated Multi-GPU Kernels (Compressed)
// =============================================================================

__global__ void bfsCompressedThreadKernel_MultiGPU(
    const edge_t *__restrict__ row_ptr,
    const uint8_t *__restrict__ compressed_col, level_t *__restrict__ distances,
    const node_t *__restrict__ q, int q_size,
    node_t *__restrict__ next_frontier, int *__restrict__ next_frontier_size,
    level_t current_level, node_t **out_queues, int **out_queue_sizes,
    int my_gpu_id, int nodes_per_gpu, node_t num_nodes, int num_gpus) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < q_size) {
    node_t u = q[tid];
    edge_t start = row_ptr[u];
    edge_t end = row_ptr[u + 1];
    node_t prev_neighbor = 0;

    while (start < end) {
      node_t val = 0;
      int shift = 0;
      uint8_t byte;
      do {
        byte = compressed_col[start++];
        val |= (node_t)(byte & 127) << shift;
        shift += 7;
      } while (byte & 128);

      node_t v = prev_neighbor + val;
      prev_neighbor = v;

      int target_gpu = v / nodes_per_gpu;
      if (target_gpu >= num_gpus)
        target_gpu = num_gpus - 1;

      if (target_gpu == my_gpu_id) {
        level_t old = atomicCAS(&distances[v], UNVISITED, current_level + 1);
        if (old == UNVISITED) {
          int idx = atomicAdd(next_frontier_size, 1);
          next_frontier[idx] = v;
        }
      } else {
        int idx = atomicAdd(out_queue_sizes[target_gpu], 1);
        out_queues[target_gpu][idx] = v;
      }
    }
  }
}

// =============================================================================
// Simulated Multi-GPU Solver (Compressed)
// =============================================================================

BFSResult *solveBFSCompressedMultiGPUSimulated(CompressedCSRGraph *graph,
                                               node_t source, int num_gpus) {
  node_t num_nodes = graph->num_nodes;
  int nodes_per_gpu = (num_nodes + num_gpus - 1) / num_gpus;

  level_t *distances;
  CUDA_CHECK(cudaMallocManaged(&distances, num_nodes * sizeof(level_t)));
  for (node_t i = 0; i < num_nodes; i++)
    distances[i] = UNVISITED;

  int source_gpu = source / nodes_per_gpu;
  if (source_gpu >= num_gpus)
    source_gpu = num_gpus - 1;
  distances[source] = 0;

  node_t **frontiers = (node_t **)malloc((size_t)num_gpus * sizeof(node_t *));
  node_t **next_frontiers = (node_t **)malloc((size_t)num_gpus * sizeof(node_t *));
  int **d_next_frontier_sizes = (int **)malloc((size_t)num_gpus * sizeof(int *));
  int *h_frontier_sizes = (int *)malloc((size_t)num_gpus * sizeof(int));
  for (int i = 0; i < num_gpus; i++)
    h_frontier_sizes[i] = 0;
  h_frontier_sizes[source_gpu] = 1;

  node_t ***msg_queues = (node_t ***)malloc((size_t)num_gpus * sizeof(node_t **));
  int ***d_msg_queue_sizes = (int ***)malloc((size_t)num_gpus * sizeof(int **));
  node_t ***d_ptr_msg_queues = (node_t ***)malloc((size_t)num_gpus * sizeof(node_t **));
  int ***d_ptr_msg_queue_sizes = (int ***)malloc((size_t)num_gpus * sizeof(int **));

  for (int i = 0; i < num_gpus; i++) {
    CUDA_CHECK(cudaMallocManaged(&frontiers[i], num_nodes * sizeof(node_t)));
    CUDA_CHECK(
        cudaMallocManaged(&next_frontiers[i], num_nodes * sizeof(node_t)));
    CUDA_CHECK(cudaMallocManaged(&d_next_frontier_sizes[i], sizeof(int)));

    msg_queues[i] = (node_t **)malloc((size_t)num_gpus * sizeof(node_t *));
    d_msg_queue_sizes[i] = (int **)malloc((size_t)num_gpus * sizeof(int *));
    for (int j = 0; j < num_gpus; j++) {
      CUDA_CHECK(
          cudaMallocManaged(&msg_queues[i][j], num_nodes * sizeof(node_t)));
      CUDA_CHECK(cudaMallocManaged(&d_msg_queue_sizes[i][j], sizeof(int)));
    }

    CUDA_CHECK(cudaMalloc(&d_ptr_msg_queues[i], num_gpus * sizeof(node_t *)));
    CUDA_CHECK(cudaMemcpy(d_ptr_msg_queues[i], msg_queues[i],
                          num_gpus * sizeof(node_t *), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_ptr_msg_queue_sizes[i], num_gpus * sizeof(int *)));
    CUDA_CHECK(cudaMemcpy(d_ptr_msg_queue_sizes[i], d_msg_queue_sizes[i],
                          num_gpus * sizeof(int *), cudaMemcpyHostToDevice));
  }

  frontiers[source_gpu][0] = source;

  int level = 0;
  double start_time = omp_get_wtime();

  while (true) {
    int total_frontier = 0;
    for (int i = 0; i < num_gpus; i++)
      total_frontier += h_frontier_sizes[i];
    if (total_frontier == 0)
      break;

    printf("Level %d: Global Frontier=%d\n", level, total_frontier);

#pragma omp parallel num_threads(num_gpus)
    {
      int gpu_id = omp_get_thread_num();
      cudaSetDevice(0);
      cudaStream_t stream;
      cudaStreamCreate(&stream);

      cudaMemsetAsync(d_next_frontier_sizes[gpu_id], 0, sizeof(int), stream);
      for (int j = 0; j < num_gpus; j++)
        cudaMemsetAsync(d_msg_queue_sizes[gpu_id][j], 0, sizeof(int), stream);

      if (h_frontier_sizes[gpu_id] > 0) {
        int grid = (h_frontier_sizes[gpu_id] + 255) / 256;
        bfsCompressedThreadKernel_MultiGPU<<<grid, 256, 0, stream>>>(
            graph->d_row_Ptr, graph->d_compressed_col, distances,
            frontiers[gpu_id], h_frontier_sizes[gpu_id], next_frontiers[gpu_id],
            d_next_frontier_sizes[gpu_id], level, d_ptr_msg_queues[gpu_id],
            d_ptr_msg_queue_sizes[gpu_id], gpu_id, nodes_per_gpu, num_nodes,
            num_gpus);
      }
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }

#pragma omp parallel num_threads(num_gpus)
    {
      int gpu_id = omp_get_thread_num();
      cudaSetDevice(0);
      cudaStream_t stream;
      cudaStreamCreate(&stream);

      for (int sender = 0; sender < num_gpus; sender++) {
        if (sender == gpu_id)
          continue;
        int size = *d_msg_queue_sizes[sender][gpu_id];
        if (size > 0) {
          int grid = (size + 255) / 256;
          mergeIncomingQueues<<<grid, 256, 0, stream>>>(
              msg_queues[sender][gpu_id], size, distances,
              next_frontiers[gpu_id], d_next_frontier_sizes[gpu_id], level);
        }
      }
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }

    for (int i = 0; i < num_gpus; i++) {
      h_frontier_sizes[i] = *d_next_frontier_sizes[i];
      node_t *tmp = frontiers[i];
      frontiers[i] = next_frontiers[i];
      next_frontiers[i] = tmp;
    }
    level++;
  }

  double end_time = omp_get_wtime();
  BFSResult *res = (BFSResult *)malloc(sizeof(BFSResult));
  res->elapsed_ms = (float)((end_time - start_time) * 1000.0);
  res->num_nodes = num_nodes;
  res->source = source;
  res->distances = (level_t *)malloc((size_t)num_nodes * sizeof(level_t));
  res->parents = NULL;
  memcpy(res->distances, distances, (size_t)num_nodes * sizeof(level_t));

  CUDA_CHECK(cudaFree(distances));
  for (int i = 0; i < num_gpus; i++) {
    CUDA_CHECK(cudaFree(frontiers[i]));
    CUDA_CHECK(cudaFree(next_frontiers[i]));
    CUDA_CHECK(cudaFree(d_next_frontier_sizes[i]));
    CUDA_CHECK(cudaFree(d_ptr_msg_queues[i]));
    CUDA_CHECK(cudaFree(d_ptr_msg_queue_sizes[i]));
    for (int j = 0; j < num_gpus; j++) {
      CUDA_CHECK(cudaFree(msg_queues[i][j]));
      CUDA_CHECK(cudaFree(d_msg_queue_sizes[i][j]));
    }
    free(msg_queues[i]);
    free(d_msg_queue_sizes[i]);
  }
  free(frontiers);
  free(next_frontiers);
  free(d_next_frontier_sizes);
  free(h_frontier_sizes);
  free(msg_queues);
  free(d_msg_queue_sizes);
  free(d_ptr_msg_queues);
  free(d_ptr_msg_queue_sizes);

  return res;
}
