#ifndef BFS_DYNAMIC_H
#define BFS_DYNAMIC_H

#include "graph.h"
#include "utils.h"

/**
 * Version 1: Dynamic Thread Assignment BFS
 *
 * Each thread is assigned to explore one node at a time.
 * When a thread finishes processing its node, it dynamically
 * picks up another unprocessed node from the work queue.
 *
 * This approach provides good load balancing when node degrees
 * vary significantly across the graph.
 */

/**
 * Run BFS with dynamic thread assignment
 *
 * @param graph     Graph in CSR format (must be on device)
 * @param source    Source node for BFS
 * @return          BFS result with distances and timing
 */
BFSResult *bfsDynamic(CSRGraph *graph, node_t source);

/**
 * BFS kernel with dynamic work distribution
 */
__global__ void bfsDynamicKernel(const edge_t *row_ptr, const node_t *col_idx,
                                 level_t *distances, node_t *frontier,
                                 int *frontier_size, node_t *next_frontier,
                                 int *next_frontier_size, node_t num_nodes,
                                 level_t current_level);

#endif // BFS_DYNAMIC_H
