#ifndef BFS_SHARED_H
#define BFS_SHARED_H

#include "graph.h"
#include "utils.h"

/**
 * Version 3: Shared Memory with Warp Cooperation BFS
 *
 * Leverages shared memory for collaboration between threads within
 * the same block. Each block contains multiple warps (32 threads each),
 * and threads within the same warp cooperate to load and process
 * neighbors of the same node.
 *
 * This approach provides:
 * - Better memory coalescing
 * - Efficient use of shared memory cache
 * - Warp-level parallelism for neighbor loading
 */

/**
 * Run BFS with shared memory and warp cooperation
 *
 * @param graph     Graph in CSR format (must be on device)
 * @param source    Source node for BFS
 * @return          BFS result with distances and timing
 */
BFSResult *bfsShared(CSRGraph *graph, node_t source);

/**
 * BFS kernel with shared memory
 * Threads in a warp cooperate to process one node's neighbors
 */
__global__ void bfsSharedKernel(const edge_t *row_ptr, const node_t *col_idx,
                                level_t *distances, node_t *frontier,
                                int *frontier_size, node_t *next_frontier,
                                int *next_frontier_size, node_t num_nodes,
                                level_t current_level);

#endif // BFS_SHARED_H
