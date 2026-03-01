#ifndef BFS_KERNELS_CUH
#define BFS_KERNELS_CUH

/**
 * @file bfs_kernels.cuh
 * @brief Shared BFS kernel implementations for Direction-Optimized BFS.
 *
 * Contains:
 * - Bottom-Up Warp Kernel (for large frontiers)
 * - Bitmap utility kernels (conversion, clearing)
 * - Queue regeneration kernel
 *
 * Version 4.1: Hybrid-Optimized
 */

#include "cuda_common.h"

#define WARPS_PER_BLOCK 8

// Skip kernel definitions if the translation unit already has its own versions
// (e.g., bfs_shared.cu has legacy implementations)
#ifndef BFS_KERNELS_SKIP_DEFINITIONS

// =============================================================================
// Bitmap Utility Kernels
// =============================================================================

/**
 * Convert queue indices to bitmap representation.
 * Each bit in bitmap corresponds to a node ID.
 */
__global__ void queueToBitmapKernel(const node_t *__restrict__ queue, int size,
                                    unsigned int *__restrict__ bitmap) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    node_t node = queue[tid];
    atomicOr(&bitmap[node / 32], (1u << (node % 32)));
  }
}

/**
 * Generate visited bitmap from distances array.
 * Sets bit if distances[node] != UNVISITED.
 */
__global__ void
generateVisitedBitmapKernel(const level_t *__restrict__ distances,
                            unsigned int *__restrict__ visited_bitmap,
                            node_t num_nodes) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_nodes) {
    if (distances[tid] != UNVISITED) {
      atomicOr(&visited_bitmap[tid / 32], (1u << (tid % 32)));
    }
  }
}

/**
 * Clear bitmap (set all bits to 0).
 */
__global__ void clearBitmapKernel(unsigned int *__restrict__ bitmap,
                                  int size_ints) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size_ints)
    bitmap[tid] = 0;
}

/**
 * Regenerate queue from distances array for a specific level.
 * Used to convert back from Bottom-Up to Top-Down.
 */
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

// =============================================================================
// Bottom-Up BFS Kernel (Warp-Cooperative)
// =============================================================================

/**
 * Bottom-Up BFS traversal using warp-cooperative scanning.
 *
 * Each warp processes one unvisited node:
 * - Check if any neighbor is in the current frontier (via bitmap)
 * - If yes, mark node as visited at current_level + 1
 *
 * Optimized for large frontiers where Top-Down creates excessive atomics.
 */
// Bottom-Up BFS Traversal with Direct Queue Emission (V5.3)
__global__ void
bfsBottomUpWarpKernel_Direct(const edge_t *__restrict__ row_ptr,
                             const node_t *__restrict__ col_idx,
                             level_t *__restrict__ distances,
                             const unsigned int *__restrict__ frontier_bitmap,
                             const unsigned int *__restrict__ visited_bitmap,
                             int *__restrict__ next_frontier_size,
                             node_t *__restrict__ next_frontier, // Added
                             const level_t current_level, node_t num_nodes) {

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;

  node_t u = global_warp_id;
  if (u < num_nodes) {
    bool visited = false;
    // Visited Bitmap Check (Optional: Could assume distances[u] != UNVISITED)
    if (visited_bitmap)
      visited = (visited_bitmap[u / 32] >> (u % 32)) & 1;
    else
      visited = (distances[u] != UNVISITED);

    bool found = false;

    if (!visited) {
      edge_t start = row_ptr[u];
      edge_t end = row_ptr[u + 1];

      // Cooperative scan
      for (edge_t e = start + lane_id; e < end && !found; e += WARP_SIZE) {
        node_t neighbor = col_idx[e];
        if ((frontier_bitmap[neighbor / 32] >> (neighbor % 32)) & 1) {
          found = true;
        }
        if (__any_sync(0xFFFFFFFF, found)) {
          found = true;
          break;
        }
      }

      // V5.3: Warp-Aggregated Queue Append
      // 1. Leader updates distance
      if (found && lane_id == 0) {
        distances[u] = current_level + 1;
      }
    }
    // 2. Warp-Aggregated Queue Emission (Only if node 'u' was found by this
    // warp)
    // Note: 'found' is true for all lanes if any lane found it.
    // However, only 1 node 'u' is processed by the warp.
    // So we just need 1 Atomic add.

    if (found) {
      if (lane_id == 0) {
        int idx = atomicAdd(next_frontier_size, 1);
        next_frontier[idx] = u;
      }
    }
  }
}

// Legacy Bottom-Up BFS Kernel (Warp-Cooperative)
__global__ void
bfsBottomUpWarpKernel(const edge_t *__restrict__ row_ptr,
                      const node_t *__restrict__ col_idx,
                      level_t *__restrict__ distances,
                      const unsigned int *__restrict__ frontier_bitmap,
                      const unsigned int *__restrict__ visited_bitmap,
                      int *__restrict__ next_frontier_size,
                      const level_t current_level, node_t num_nodes) {
  // ... (Legacy body kept for bfs_shared.cu compatibility if needed)
  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;

  node_t u = global_warp_id;
  if (u < num_nodes) {
    bool visited = (visited_bitmap[u / 32] >> (u % 32)) & 1;
    if (!visited) {
      edge_t start = row_ptr[u];
      edge_t end = row_ptr[u + 1];
      bool found = false;

      for (edge_t e = start + lane_id; e < end && !found; e += WARP_SIZE) {
        node_t neighbor = col_idx[e];
        if ((frontier_bitmap[neighbor / 32] >> (neighbor % 32)) & 1) {
          found = true;
        }
        if (__any_sync(0xFFFFFFFF, found)) {
          found = true;
          break;
        }
      }

      if (found && lane_id == 0) {
        distances[u] = current_level + 1;
        atomicAdd(next_frontier_size, 1);
      }
    }
  }
}

#endif // BFS_KERNELS_SKIP_DEFINITIONS

// When definitions are skipped, we still need forward declarations
// so that the translation unit can call kernels defined elsewhere
#ifdef BFS_KERNELS_SKIP_DEFINITIONS
extern __global__ void queueToBitmapKernel(const node_t *__restrict__ queue,
                                           int size,
                                           unsigned int *__restrict__ bitmap);

extern __global__ void
generateVisitedBitmapKernel(const level_t *__restrict__ distances,
                            unsigned int *__restrict__ visited_bitmap,
                            node_t num_nodes);

extern __global__ void clearBitmapKernel(unsigned int *__restrict__ bitmap,
                                         int size_ints);

extern __global__ void
distancesToQueueKernel(const level_t *__restrict__ distances, node_t num_nodes,
                       node_t *__restrict__ queue, int *__restrict__ queue_size,
                       level_t level);

extern __global__ void
bfsBottomUpWarpKernel(const edge_t *__restrict__ row_ptr,
                      const node_t *__restrict__ col_idx,
                      level_t *__restrict__ distances,
                      const unsigned int *__restrict__ frontier_bitmap,
                      const unsigned int *__restrict__ visited_bitmap,
                      int *__restrict__ next_frontier_size,
                      const level_t current_level, node_t num_nodes);

extern __global__ void bfsBottomUpWarpKernel_Direct(
    const edge_t *__restrict__ row_ptr, const node_t *__restrict__ col_idx,
    level_t *__restrict__ distances,
    const unsigned int *__restrict__ frontier_bitmap,
    const unsigned int *__restrict__ visited_bitmap,
    int *__restrict__ next_frontier_size, node_t *__restrict__ next_frontier,
    const level_t current_level, node_t num_nodes);
#endif // BFS_KERNELS_SKIP_DEFINITIONS (declarations)

#endif // BFS_KERNELS_CUH
