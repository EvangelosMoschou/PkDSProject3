#include "bfs_shared.h"
#include <cstring>

// =============================================================================
// Hybrid Strategic Solver
// =============================================================================
// This module selects between high-speed BFS and memory-efficient Union Find
// based on the available VRAM and graph dimensions.
// =============================================================================

#define WARPS_PER_BLOCK 8
// FIX: SHARED_NEIGHBORS_PER_WARP was 64 but warps only have 32 threads!
// This caused half the neighbors to be skipped, resulting in ~50% reachability.
#define SHARED_NEIGHBORS_PER_WARP WARP_SIZE // Must equal warp size (32)

// =============================================================================
// Kernels: BFS (Breadth-First Search)
// =============================================================================


__global__ void bfsWarpKernel(
    const edge_t *__restrict__ row_ptr, const node_t *__restrict__ col_idx,
    level_t *__restrict__ distances, const node_t *__restrict__ frontier,
    const int frontier_size, node_t *__restrict__ next_frontier,
    int *__restrict__ next_frontier_size, const level_t current_level) {

  __shared__ node_t s_neighbors[WARPS_PER_BLOCK][SHARED_NEIGHBORS_PER_WARP];

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;

  if (global_warp_id < frontier_size) {
    node_t current = frontier[global_warp_id];
    edge_t start = row_ptr[current];
    edge_t end = row_ptr[current + 1];
    edge_t degree = end - start;

    for (edge_t chunk_start = 0; chunk_start < degree;
         chunk_start += SHARED_NEIGHBORS_PER_WARP) {
      int chunk_size =
          min((int)(degree - chunk_start), SHARED_NEIGHBORS_PER_WARP);

      if (lane_id < chunk_size) {
        s_neighbors[warp_id][lane_id] = col_idx[start + chunk_start + lane_id];
      }
      __syncwarp();

      bool found = false;
      node_t neighbor = 0;

      if (lane_id < chunk_size) {
        neighbor = s_neighbors[warp_id][lane_id];

        // Optimization: Use native 32-bit atomics (Hardware Supported)
        // Returns old value. If old value was UNVISITED, we successfully
        // claimed it.
        level_t old_val = atomicCAS(&distances[neighbor], (level_t)UNVISITED,
                                    (level_t)(current_level + 1));

        if (old_val == UNVISITED) {
          found = true;
        }
      }

      // Warp Aggregation for Frontier Writes
      unsigned int ballot = __ballot_sync(0xFFFFFFFF, found);
      int pop_count = __popc(ballot);

      if (pop_count > 0) {
        int base_idx = 0;
        // Leader elects space
        if (lane_id == 0) {
          base_idx = atomicAdd(next_frontier_size, pop_count);
        }
        // Broadcast base index to all lanes
        base_idx = __shfl_sync(0xFFFFFFFF, base_idx, 0);

        // Calculate local offset using population count of lower lanes
        unsigned int lower_mask = (1u << lane_id) - 1;
        int local_offset = __popc(ballot & lower_mask);

        if (found) {
          next_frontier[base_idx + local_offset] = neighbor;
        }
      }
      __syncwarp();
    }
  }
}

// =============================================================================
// Kernels: Union Find (Connected Components)
// =============================================================================

__global__ void initParentKernel(node_t *parent, node_t num_nodes) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_nodes)
    parent[tid] = tid;
}

__global__ void hookKernel(const edge_t *__restrict__ row_ptr,
                           const node_t *__restrict__ col_idx,
                           node_t *__restrict__ parent,
                           int *__restrict__ changed, node_t num_nodes) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_nodes) {
    node_t u = tid;
    node_t p_u = parent[u];
    edge_t start = row_ptr[u];
    edge_t end = row_ptr[u + 1];
    if (start == end)
      return;
    for (edge_t e = start; e < end; e++) {
      node_t v = col_idx[e];
      node_t p_v = parent[v];
      if (p_v < p_u) {
        if (atomicMin(&parent[p_u], p_v) > p_v) {
          *changed = 1;
          p_u = p_v;
        }
      }
    }
  }
}

__global__ void compressKernel(node_t *parent, node_t num_nodes) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_nodes) {
    node_t p = parent[tid];
    node_t pp = parent[p];
    if (p != pp)
      parent[tid] = pp;
  }
}

__global__ void connectivityToDistanceKernel(const node_t *__restrict__ parent,
                                             level_t *__restrict__ distances,
                                             node_t source_root,
                                             node_t num_nodes) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_nodes) {
    node_t curr = tid;
    while (curr != parent[curr])
      curr = parent[curr];
    distances[tid] = (curr == source_root) ? 1 : UNVISITED;
  }
}

// =============================================================================
// Internal Solver Implementations
// =============================================================================

// Optimization B: Warp-Cooperative Bottom-Up Kernel
// - Warps load row_ptr cooperatively and scan adjacency lists together.
__global__ void
bfsBottomUpWarpKernel(const edge_t *__restrict__ row_ptr,
                      const node_t *__restrict__ col_idx,
                      level_t *__restrict__ distances,
                      const unsigned int *__restrict__ frontier_bitmap,
                      const unsigned int *__restrict__ visited_bitmap,
                      int *__restrict__ next_frontier_size,
                      const level_t current_level, node_t num_nodes) {
  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;

  // Each warp processes WARP_SIZE nodes, but ideally needs ONE node per warp if
  // doing cooperative scan? Actually, Bottom-Up is best when one Thread = One
  // Node? No. If high-degree unvisited node, one thread = slow. If we map 1
  // Warp -> 1 Node:

  node_t u = global_warp_id;
  if (u < num_nodes) {
    bool visited = (visited_bitmap[u / 32] >> (u % 32)) & 1;
    if (!visited) {
      edge_t start = row_ptr[u];
      edge_t end = row_ptr[u + 1];
      bool found = false;

      // Cooperative Scan
      for (edge_t e = start + lane_id; e < end; e += WARP_SIZE) {
        node_t neighbor = col_idx[e];
        if ((frontier_bitmap[neighbor / 32] >> (neighbor % 32)) & 1) {
          found = true;
        }
        if (__any_sync(0xFFFFFFFF, found)) {
          found = true;
          break; // Fast exit if any thread finds it
        }
      }

      if (found) {
        if (lane_id == 0)
          distances[u] = current_level + 1;
      }
    }
  }
}

// Kernel to convert Queue to Bitmap
__global__ void queueToBitmapKernel(const node_t *__restrict__ queue, int size,
                                    unsigned int *__restrict__ bitmap) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    node_t node = queue[tid];
    atomicOr(&bitmap[node / 32], (1 << (node % 32)));
  }
}

// Generate Visited Bitmap from Distances
__global__ void
generateVisitedBitmapKernel(const level_t *__restrict__ distances,
                            unsigned int *__restrict__ visited_bitmap,
                            node_t num_nodes) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_nodes) {
    if (distances[tid] != UNVISITED) {
      atomicOr(&visited_bitmap[tid / 32], (1 << (tid % 32)));
    }
  }
}

// Kernel to clear Bitmap
__global__ void clearBitmapKernel(unsigned int *__restrict__ bitmap,
                                  int size_ints) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size_ints)
    bitmap[tid] = 0;
}

// Kernel to generate Queue from Distances (switch back to Top-Down)
__global__ void distancesToQueueKernel(const level_t *__restrict__ distances,
                                       node_t num_nodes,
                                       node_t *__restrict__ queue,
                                       int *__restrict__ queue_size,
                                       level_t level) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_nodes) {
    if (distances[tid] == level) {
      int idx = atomicAdd(queue_size, 1);
      queue[idx] = tid;
    }
  }
}

BFSResult *solveBFS(CSRGraph *graph, node_t source) {
  node_t num_nodes = graph->num_nodes;
  level_t *d_distances;
  node_t *d_frontier, *d_next_frontier;
  int *d_next_frontier_size;
  unsigned int *d_frontier_bitmap; // For Bottom-Up
  unsigned int *d_visited_bitmap;  // NEW: For Bottom-Up efficient scanning

  CUDA_CHECK(cudaMalloc(&d_distances, num_nodes * sizeof(level_t)));
  CUDA_CHECK(cudaMalloc(&d_frontier, num_nodes * sizeof(node_t)));
  CUDA_CHECK(cudaMalloc(&d_next_frontier, num_nodes * sizeof(node_t)));
  CUDA_CHECK(cudaMalloc(&d_next_frontier_size, sizeof(int)));

  // Bitmap allocation: 1 bit per node = num_nodes / 32 ints
  int bitmap_ints = (num_nodes + 31) / 32;
  CUDA_CHECK(
      cudaMalloc(&d_frontier_bitmap, bitmap_ints * sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&d_visited_bitmap, bitmap_ints * sizeof(unsigned int)));

  CUDA_CHECK(cudaMemset(d_distances, UNVISITED, num_nodes * sizeof(level_t)));
  level_t zero = 0;
  CUDA_CHECK(cudaMemcpy(d_distances + source, &zero, sizeof(level_t),
                        cudaMemcpyHostToDevice));

  node_t h_frontier[1] = {source};
  CUDA_CHECK(cudaMemcpy(d_frontier, h_frontier, sizeof(node_t),
                        cudaMemcpyHostToDevice));

  int frontier_size = 1;

  CudaTimer timer = createTimer();
  startTimer(&timer);

  int level = 0;
  // Heuristic: Switch to Bottom-Up if frontier > 1/20th of nodes
  // FIX: Disable Bottom-Up for Directed Graphs (correctness)
  int heavy_threshold = num_nodes + 1;

  while (frontier_size > 0) {
    CUDA_CHECK(cudaMemset(d_next_frontier_size, 0, sizeof(int)));

    if (frontier_size > heavy_threshold) {
      // 1. Queue -> Bitmap
      int numBlocksBitmap = (frontier_size + 1023) / 1024;
      clearBitmapKernel<<<(bitmap_ints + 1023) / 1024, 1024>>>(
          d_frontier_bitmap, bitmap_ints);
      queueToBitmapKernel<<<numBlocksBitmap, 1024>>>(d_frontier, frontier_size,
                                                     d_frontier_bitmap);
      CUDA_CHECK(cudaDeviceSynchronize()); // Ensure Bitmap is ready

      // --- STANDARD BOTTOM-UP PHASE (In-Core) ---
      // Pre-step: Build Visited Bitmap
      int numBlocksVisited = (num_nodes + 1023) / 1024;
      clearBitmapKernel<<<(bitmap_ints + 1023) / 1024, 1024>>>(d_visited_bitmap,
                                                               bitmap_ints);
      generateVisitedBitmapKernel<<<numBlocksVisited, 1024>>>(
          d_distances, d_visited_bitmap, num_nodes);
      CUDA_CHECK(cudaDeviceSynchronize());

      bfsBottomUpWarpKernel<<<numBlocksVisited, 1024>>>(
          graph->d_row_ptr, graph->d_col_idx, d_distances, d_frontier_bitmap,
          d_visited_bitmap, // New argument
          d_next_frontier_size, level, num_nodes);
      CUDA_CHECK(cudaDeviceSynchronize());

      // 3. Regenerate Queue for Next Step
      CUDA_CHECK(cudaMemset(d_next_frontier_size, 0, sizeof(int)));
      int numBlocksNodes = (num_nodes + 1023) / 1024;
      distancesToQueueKernel<<<numBlocksNodes, 1024>>>(
          d_distances, num_nodes, d_next_frontier, d_next_frontier_size,
          level + 1);
      CUDA_CHECK(cudaDeviceSynchronize());

    } else {
      // --- TOP-DOWN PHASE (Standard) ---
      int numBlocks = (frontier_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
      int blockSize = WARPS_PER_BLOCK * WARP_SIZE;
      bfsWarpKernel<<<numBlocks, blockSize>>>(
          graph->d_row_ptr, graph->d_col_idx, d_distances, d_frontier,
          frontier_size, d_next_frontier, d_next_frontier_size, level);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaGetLastError());

    int h_next_frontier_size;
    CUDA_CHECK(cudaMemcpy(&h_next_frontier_size, d_next_frontier_size,
                          sizeof(int), cudaMemcpyDeviceToHost));

    frontier_size = h_next_frontier_size;
    node_t *temp = d_frontier;
    d_frontier = d_next_frontier;
    d_next_frontier = temp;

    level++;
  }

  float elapsed = stopTimer(&timer);
  BFSResult *result = new BFSResult;
  result->num_nodes = num_nodes;
  result->source = source;
  result->elapsed_ms = elapsed;
  result->distances = new level_t[num_nodes];
  CUDA_CHECK(cudaMemcpy(result->distances, d_distances,
                        num_nodes * sizeof(level_t), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_distances));
  CUDA_CHECK(cudaFree(d_frontier));
  CUDA_CHECK(cudaFree(d_next_frontier));
  CUDA_CHECK(cudaFree(d_next_frontier_size));
  CUDA_CHECK(cudaFree(d_frontier_bitmap));
  CUDA_CHECK(cudaFree(d_visited_bitmap));

  return result;
}

// Afforest Sampling Kernel: connect nodes to random neighbors to speed up
// convergence
__global__ void afforestSamplingKernel(const edge_t *__restrict__ row_ptr,
                                       const node_t *__restrict__ col_idx,
                                       node_t *__restrict__ parent,
                                       node_t num_nodes, unsigned int seed,
                                       node_t skip_component) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_nodes) {
    // Large Component Skipping Optimization
    // If we are already in the known large component, skip sampling
    // This assumes other nodes will hook into us.
    if (skip_component != -1 && parent[tid] == skip_component) {
      return;
    }

    edge_t start = row_ptr[tid];
    edge_t end = row_ptr[tid + 1];
    edge_t degree = end - start;
    if (degree > 0) {
      // Simple pseudo-random using linear congruential generator
      unsigned int r = (seed * 1664525 + 1013904223 + tid);
      edge_t offset = r % degree;
      node_t neighbor = col_idx[start + offset];

      node_t u = tid;
      node_t v = neighbor;

      // Attempt to link u and v
      node_t p_u = parent[u];
      node_t p_v = parent[v];

      if (p_v < p_u) {
        atomicMin(&parent[p_u], p_v);
      } else if (p_u < p_v) {
        atomicMin(&parent[p_v], p_u);
      }
    }
  }
}

BFSResult *solveUnionFind(CSRGraph *graph, node_t source) {
  node_t num_nodes = graph->num_nodes;
  node_t *d_parent;
  int *d_changed;
  level_t *d_distances;

  CUDA_CHECK(cudaMalloc(&d_parent, num_nodes * sizeof(node_t)));
  CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_distances, num_nodes * sizeof(level_t)));

  int blockSize = 256;
  int numBlocks = (num_nodes + blockSize - 1) / blockSize;
  initParentKernel<<<numBlocks, blockSize>>>(d_parent, num_nodes);

  CudaTimer timer = createTimer();
  startTimer(&timer);

  // --- AFFOREST SAMPLING PHASE ---
  // Run 2 rounds of random sampling to collapse components quickly
  int sampling_rounds = 2;
  printf("UnionFind: Running %d Afforest Sampling Rounds...\n",
         sampling_rounds);

  node_t large_component = -1;

  for (int r = 0; r < sampling_rounds; r++) {
    // After first round, try to identify large component
    if (r > 0) {
      // Heuristic: Node 0 is likely in large component
      CUDA_CHECK(cudaMemcpy(&large_component, &d_parent[0], sizeof(node_t),
                            cudaMemcpyDeviceToHost));
      // Use the parent as proxy for component ID
    }

    afforestSamplingKernel<<<numBlocks, blockSize>>>(
        graph->d_row_ptr, graph->d_col_idx, d_parent, num_nodes, r + 42,
        large_component);
    compressKernel<<<numBlocks, blockSize>>>(d_parent, num_nodes);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  int h_changed = 1, iter = 0;
  while (h_changed) {
    h_changed = 0;
    CUDA_CHECK(
        cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
    hookKernel<<<numBlocks, blockSize>>>(graph->d_row_ptr, graph->d_col_idx,
                                         d_parent, d_changed, num_nodes);
    compressKernel<<<numBlocks, blockSize>>>(d_parent, num_nodes);
    CUDA_CHECK(
        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
    iter++;
  }

  for (int k = 0; k < 5; k++)
    compressKernel<<<numBlocks, blockSize>>>(d_parent, num_nodes);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Find source root (iterative copy since d_parent is on device)
  node_t source_root = source;
  node_t parent_val;
  while (true) {
    CUDA_CHECK(cudaMemcpy(&parent_val, &d_parent[source_root], sizeof(node_t),
                          cudaMemcpyDeviceToHost));
    if (parent_val == source_root)
      break;
    source_root = parent_val;
  }

  connectivityToDistanceKernel<<<numBlocks, blockSize>>>(
      d_parent, d_distances, source_root, num_nodes);
  level_t zero = 0;
  CUDA_CHECK(cudaMemcpy(d_distances + source, &zero, sizeof(level_t),
                        cudaMemcpyHostToDevice));

  float elapsed = stopTimer(&timer);
  BFSResult *result = new BFSResult;
  result->num_nodes = num_nodes;
  result->source = source;
  result->elapsed_ms = elapsed;
  result->distances = new level_t[num_nodes];
  CUDA_CHECK(cudaMemcpy(result->distances, d_distances,
                        num_nodes * sizeof(level_t), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_parent));
  CUDA_CHECK(cudaFree(d_changed));
  CUDA_CHECK(cudaFree(d_distances));
  return result;
}

// =============================================================================
// Strategy Dispatcher
// =============================================================================

BFSResult *bfsShared(CSRGraph *graph, BFSOptions *opts) {
  if (!graph->d_row_ptr)
    copyGraphToDevice(graph);

  // If user explicitly requested Afforest/Union-Find, skip BFS logic
  if (opts->algorithm == ALGO_AFFOREST) {
    printf("Decision: User requested AFFOREST (Connectivity).\n");
    printf("----------------------------\n");
    return solveUnionFind(graph, opts->source);
  }

  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);

  // BFS requirement beyond graph data: 2 frontiers (4B each) + distances (1B)
  size_t bfs_extra =
      (size_t)graph->num_nodes * (sizeof(node_t) * 2 + sizeof(level_t));

  printf("--- Strategic Dispatcher ---\n");
  printf("Free VRAM: %.2f GB\n", free_mem / (1024.0 * 1024.0 * 1024.0));
  printf("BFS Buffer Requirement: %.2f GB\n",
         bfs_extra / (1024.0 * 1024.0 * 1024.0));

  // Safety threshold: Allow up to 18GB total usage (VRAM + System RAM via
  // Managed Memory) Friendster is ~14.5GB (Use Hybrid BFS), Mawi is ~4GB (Use
  // Hybrid BFS).
  size_t system_ram_limit = 18ULL * 1024 * 1024 * 1024;

  // We check if (Graph Size + BFS Buffers) < System Limit
  size_t total_est_usage =
      bfs_extra + (size_t)graph->num_nodes * 8 + (size_t)graph->num_edges * 4;

  printf("Total Estimated Memory: %.2f GB\n",
         total_est_usage / (1024.0 * 1024.0 * 1024.0));
  printf("Unified Memory Limit:   %.2f GB\n",
         system_ram_limit / (1024.0 * 1024.0 * 1024.0));

  if (total_est_usage < system_ram_limit) {
    return solveBFS(graph, opts->source);
  } else {
    printf("Decision: Memory exceeds 16GB! Switching to MEMORY-EFFICIENT Union "
           "Find.\n");
    printf("----------------------------\n");
    return solveUnionFind(graph, opts->source);
  }
}

#include "../common/json_gpu.h"

int main(int argc, char **argv) {
  BFSOptions opts = parseArgs(argc, argv);

  if (!opts.json_output) {
    printf("=== Dynamic Hybrid Connectivity Solver ===\n\n");
    printDeviceInfo();
  }

  CSRGraph *graph = nullptr;
  const char *ext = strrchr(opts.graph_file, '.');
  if (ext && strcmp(ext, ".csrbin") == 0)
    graph = loadGraphCSRBin(opts.graph_file);
  else if (ext && strcmp(ext, ".mat") == 0)
    graph = loadGraphHDF5(opts.graph_file);
  else
    graph = loadGraph(opts.graph_file);

  if (!graph)
    return 1;

  if (!opts.json_output)
    printGraphStats(graph);

  copyGraphToDevice(graph);

  // Benchmarking Loop
  int num_trials = opts.benchmark ? opts.num_runs : 1;
  double *times = new double[num_trials];
  BFSResult *final_result = nullptr;

  for (int i = 0; i < num_trials; i++) {
    if (final_result)
      freeBFSResult(final_result);

    // We need to re-run. bfsShared handles logic.
    // Ideally we should sync before starting timer, but bfsShared does it
    // inside.
    final_result = bfsShared(graph, &opts);
    times[i] = final_result->elapsed_ms;

    if (!opts.json_output) {
      printf("Trial %d: %.3f ms\n", i + 1, final_result->elapsed_ms);
    }
  }

  // Note: For streaming detection, we'd need to modify bfsShared return
  // signature or track internally.
  bool used_streaming = false;

  if (opts.json_output) {
    // Compute traversed edges (reachable nodes * avg degree? No.)
    // Exact traversed edges requires counting degrees of visited nodes.
    // For GTEPS, we usually use Total Edges of the component or graph if fully
    // connected. Let's use total graph edges for now as approximation or count
    // it. NOTE: Standard Graph500 uses input edges / time.

    // We need edge_t alias, assuming it's long long in cuda_common.h
    print_json_gpu("Hybrid_BFS", opts.graph_file, graph->num_nodes,
                   graph->num_edges, times, num_trials, graph->num_edges,
                   used_streaming);
  } else {
    printBFSResult(final_result);
    if (opts.benchmark) {
      double sum = 0;
      for (int i = 0; i < num_trials; i++)
        sum += times[i];
      printf("Average Time: %.3f ms\n", sum / num_trials);
    }
  }

  if (opts.validate && !opts.json_output) {
    // Validate only once using final result
    validateBFSResult(final_result, graph);
  }

  freeBFSResult(final_result);
  freeGraph(graph);
  delete[] times;
  return 0;
}
