#include "bfs_dynamic.h"
#include <cstring>

// =============================================================================
// Version 1: Dynamic Thread Assignment BFS
// =============================================================================
//
// Each thread dynamically picks up work from a shared frontier queue.
// When a thread finishes processing a node, it atomically fetches the
// next available node from the frontier.
//
// Pros:
//   - Good load balancing for irregular graphs
//   - Handles high-degree nodes well
//
// Cons:
//   - Atomic contention on frontier queue
//   - Less predictable memory access patterns
// =============================================================================

/**
 * Kernel: Each thread processes one node at a time from the frontier
 * Uses atomic operations to dynamically fetch work
 */
__global__ void bfsDynamicKernel(
    const edge_t *__restrict__ row_ptr, const node_t *__restrict__ col_idx,
    level_t *__restrict__ distances, const node_t *__restrict__ frontier,
    const int frontier_size, node_t *__restrict__ next_frontier,
    int *__restrict__ next_frontier_size, const level_t current_level) {
  // Global work index - each thread picks up work dynamically
  __shared__ int shared_work_idx;

  if (threadIdx.x == 0) {
    shared_work_idx = blockIdx.x * blockDim.x;
  }
  __syncthreads();

  while (true) {
    // Atomically get next work item
    int my_work_idx;
    if (threadIdx.x == 0) {
      my_work_idx = atomicAdd(&shared_work_idx, blockDim.x);
    }
    // Broadcast to all threads in block
    my_work_idx = __shfl_sync(0xFFFFFFFF, my_work_idx, 0) + threadIdx.x;

    // Check if there's still work to do
    if (my_work_idx >= frontier_size)
      break;

    node_t current = frontier[my_work_idx];
    edge_t start = row_ptr[current];
    edge_t end = row_ptr[current + 1];

    // Process all neighbors
    for (edge_t e = start; e < end; e++) {
      node_t neighbor = col_idx[e];

      // Try to mark as visited (atomic compare-and-swap)
      if (atomicCAS(&distances[neighbor], UNVISITED, current_level + 1) ==
          UNVISITED) {
        // Successfully visited - add to next frontier
        int idx = atomicAdd(next_frontier_size, 1);
        next_frontier[idx] = neighbor;
      }
    }
  }
}

/**
 * Alternative simpler kernel for comparison
 * Each thread gets one index from frontier
 */
__global__ void bfsDynamicKernelSimple(
    const edge_t *__restrict__ row_ptr, const node_t *__restrict__ col_idx,
    level_t *__restrict__ distances, const node_t *__restrict__ frontier,
    const int frontier_size, node_t *__restrict__ next_frontier,
    int *__restrict__ next_frontier_size, const level_t current_level) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < frontier_size) {
    node_t current = frontier[tid];
    edge_t start = row_ptr[current];
    edge_t end = row_ptr[current + 1];

    // Process all neighbors of this node
    for (edge_t e = start; e < end; e++) {
      node_t neighbor = col_idx[e];

      // Atomically try to visit
      if (atomicCAS(&distances[neighbor], UNVISITED, current_level + 1) ==
          UNVISITED) {
        int idx = atomicAdd(next_frontier_size, 1);
        next_frontier[idx] = neighbor;
      }
    }
  }
}

BFSResult *bfsDynamic(CSRGraph *graph, node_t source) {
  // Ensure graph is on device
  if (!graph->d_row_ptr) {
    copyGraphToDevice(graph);
  }

  node_t num_nodes = graph->num_nodes;

  // Allocate device memory for BFS
  level_t *d_distances;
  node_t *d_frontier;
  node_t *d_next_frontier;
  int *d_frontier_size;
  int *d_next_frontier_size;

  CUDA_CHECK(cudaMallocManaged(&d_distances, num_nodes * sizeof(level_t)));
  CUDA_CHECK(cudaMallocManaged(&d_frontier, num_nodes * sizeof(node_t)));
  CUDA_CHECK(cudaMallocManaged(&d_next_frontier, num_nodes * sizeof(node_t)));
  CUDA_CHECK(cudaMallocManaged(&d_frontier_size, sizeof(int)));
  CUDA_CHECK(cudaMallocManaged(&d_next_frontier_size, sizeof(int)));

  // Initialize distances to UNVISITED
  CUDA_CHECK(cudaMemset(d_distances, 0xFF, num_nodes * sizeof(level_t)));

  // Set source distance to 0
  level_t zero = 0;
  CUDA_CHECK(cudaMemcpy(d_distances + source, &zero, sizeof(level_t),
                        cudaMemcpyHostToDevice));

  // Initialize frontier with source
  CUDA_CHECK(
      cudaMemcpy(d_frontier, &source, sizeof(node_t), cudaMemcpyHostToDevice));
  int h_frontier_size = 1;
  CUDA_CHECK(cudaMemcpy(d_frontier_size, &h_frontier_size, sizeof(int),
                        cudaMemcpyHostToDevice));

  // Create timer
  CudaTimer timer = createTimer();
  startTimer(&timer);

  // BFS iterations
  level_t current_level = 0;

  while (h_frontier_size > 0) {
    // Reset next frontier size
    CUDA_CHECK(cudaMemset(d_next_frontier_size, 0, sizeof(int)));

    // Launch kernel
    int num_blocks = (h_frontier_size + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

    bfsDynamicKernelSimple<<<num_blocks, BLOCK_SIZE_1D>>>(
        graph->d_row_ptr, graph->d_col_idx, d_distances, d_frontier,
        h_frontier_size, d_next_frontier, d_next_frontier_size, current_level);
    CUDA_CHECK_LAST();

    // Get next frontier size
    CUDA_CHECK(cudaMemcpy(&h_frontier_size, d_next_frontier_size, sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Swap frontiers
    node_t *temp = d_frontier;
    d_frontier = d_next_frontier;
    d_next_frontier = temp;

    current_level++;
  }

  float elapsed = stopTimer(&timer);
  destroyTimer(&timer);

  // Allocate and fill result
  BFSResult *result = new BFSResult;
  result->num_nodes = num_nodes;
  result->source = source;
  result->elapsed_ms = elapsed;
  result->distances = new level_t[num_nodes];
  result->parents = nullptr;

  // Copy distances back
  CUDA_CHECK(cudaMemcpy(result->distances, d_distances,
                        num_nodes * sizeof(level_t), cudaMemcpyDeviceToHost));

  // Cleanup
  CUDA_CHECK(cudaFree(d_distances));
  CUDA_CHECK(cudaFree(d_frontier));
  CUDA_CHECK(cudaFree(d_next_frontier));
  CUDA_CHECK(cudaFree(d_frontier_size));
  CUDA_CHECK(cudaFree(d_next_frontier_size));

  return result;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv) {
  printf("=== BFS Version 1: Dynamic Thread Assignment ===\n\n");

  // Print GPU info
  printDeviceInfo();

  // Parse arguments
  BFSOptions opts = parseArgs(argc, argv);

  // Load graph
  printf("Loading graph: %s\n", opts.graph_file);
  CSRGraph *graph = nullptr;

  // Check file extension
  const char *ext = strrchr(opts.graph_file, '.');
  if (ext && strcmp(ext, ".csrbin") == 0) {
    graph = loadGraphCSRBin(opts.graph_file);
  } else {
    graph = loadGraph(opts.graph_file);
  }

  if (!graph) {
    fprintf(stderr, "Failed to load graph\n");
    return 1;
  }

  printGraphStats(graph);

  // Copy to device
  copyGraphToDevice(graph);

  // Run BFS
  printf("Running BFS from source %d...\n", opts.source);
  BFSResult *result = bfsDynamic(graph, opts.source);

  printBFSResult(result);

  // Validate if requested
  if (opts.validate) {
    printf("Validating against CPU BFS...\n");
    validateBFSResult(result, graph);
  }

  // Benchmark mode
  if (opts.benchmark) {
    printf("\nRunning benchmark (%d iterations)...\n", opts.num_runs);
    float total_time = 0;

    for (int i = 0; i < opts.num_runs; i++) {
      BFSResult *r = bfsDynamic(graph, opts.source);
      total_time += r->elapsed_ms;
      freeBFSResult(r);
    }

    printf("Average time: %.3f ms\n", total_time / opts.num_runs);
  }

  // Cleanup
  freeBFSResult(result);
  freeGraph(graph);

  return 0;
}
