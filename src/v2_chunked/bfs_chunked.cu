#include "bfs_chunked.h"
#include <cstring>

// =============================================================================
// Version 2: Chunk-Based Processing BFS
// =============================================================================
//
// Each thread is assigned a fixed chunk of the frontier to process.
// The kernel contains a for-loop that iterates over the thread's
// assigned range of nodes.
//
// This is similar to the static work distribution approach used in
// pthreads implementations.
//
// Pros:
//   - Simple implementation
//   - Less atomic contention
//   - Predictable memory access patterns
//
// Cons:
//   - Potential load imbalance for irregular graphs
//   - High-degree nodes can create bottlenecks
// =============================================================================

/**
 * Kernel: Each thread processes a chunk of nodes from the frontier
 * Contains internal for-loop over assigned range
 */
__global__ void bfsChunkedKernel(
    const edge_t *__restrict__ row_ptr, const node_t *__restrict__ col_idx,
    level_t *__restrict__ distances, const node_t *__restrict__ frontier,
    const int frontier_size, node_t *__restrict__ next_frontier,
    int *__restrict__ next_frontier_size, const level_t current_level,
    const int chunk_size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate this thread's chunk range
  int chunk_start = tid * chunk_size;
  int chunk_end = min(chunk_start + chunk_size, frontier_size);

  // Process all nodes in this thread's chunk
  for (int i = chunk_start; i < chunk_end; i++) {
    node_t current = frontier[i];
    edge_t start = row_ptr[current];
    edge_t end = row_ptr[current + 1];

    // Process all neighbors
    for (edge_t e = start; e < end; e++) {
      node_t neighbor = col_idx[e];

      // Atomically try to visit neighbor
      if (atomicCAS(&distances[neighbor], UNVISITED, current_level + 1) ==
          UNVISITED) {
        int idx = atomicAdd(next_frontier_size, 1);
        next_frontier[idx] = neighbor;
      }
    }
  }
}

/**
 * Alternative: Level-synchronous kernel
 * All threads scan all nodes, checking if they're at current level
 */
__global__ void bfsLevelSyncKernel(
    const edge_t *__restrict__ row_ptr, const node_t *__restrict__ col_idx,
    level_t *__restrict__ distances, int *__restrict__ changed,
    const node_t num_nodes, const level_t current_level, const int chunk_size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate chunk range for this thread
  int start_node = tid * chunk_size;
  int end_node = min(start_node + chunk_size, (int)num_nodes);

  // For-loop over assigned node chunk
  for (node_t node = start_node; node < end_node; node++) {
    // Only process nodes at current level
    if (distances[node] == current_level) {
      edge_t start = row_ptr[node];
      edge_t end = row_ptr[node + 1];

      // Check all neighbors
      for (edge_t e = start; e < end; e++) {
        node_t neighbor = col_idx[e];

        // Try to update unvisited neighbor
        if (distances[neighbor] == UNVISITED) {
          distances[neighbor] = current_level + 1;
          *changed = 1;
        }
      }
    }
  }
}

BFSResult *bfsChunked(CSRGraph *graph, node_t source) {
  // Ensure graph is on device
  if (!graph->d_row_ptr) {
    copyGraphToDevice(graph);
  }

  node_t num_nodes = graph->num_nodes;

  // Allocate device memory
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

  // Calculate total threads and chunk size
  int total_threads = 1024; // Fixed number of threads

  while (h_frontier_size > 0) {
    // Reset next frontier size
    CUDA_CHECK(cudaMemset(d_next_frontier_size, 0, sizeof(int)));

    // Calculate chunk size for this iteration
    int chunk_size = (h_frontier_size + total_threads - 1) / total_threads;
    chunk_size = max(chunk_size, 1);

    // Calculate grid dimensions
    int num_blocks = (total_threads + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

    bfsChunkedKernel<<<num_blocks, BLOCK_SIZE_1D>>>(
        graph->d_row_ptr, graph->d_col_idx, d_distances, d_frontier,
        h_frontier_size, d_next_frontier, d_next_frontier_size, current_level,
        chunk_size);
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
  printf("=== BFS Version 2: Chunk-Based Processing ===\n\n");

  // Print GPU info
  printDeviceInfo();

  // Parse arguments
  BFSOptions opts = parseArgs(argc, argv);

  // Load graph
  printf("Loading graph: %s\n", opts.graph_file);
  CSRGraph *graph = nullptr;

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
  BFSResult *result = bfsChunked(graph, opts.source);

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
      BFSResult *r = bfsChunked(graph, opts.source);
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
