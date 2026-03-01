#ifndef BFS_CHUNKED_H
#define BFS_CHUNKED_H

#include "graph.h"
#include "utils.h"

/**
 * Version 2: Chunk-Based Processing BFS
 *
 * Each thread is responsible for processing a fixed chunk of nodes,
 * similar to the approach used in pthreads. The kernel contains an
 * internal for-loop that iterates over the assigned node range.
 *
 * This approach has less synchronization overhead but may suffer
 * from load imbalance if node degrees vary significantly.
 */

/**
 * Run BFS with chunk-based thread assignment
 *
 * @param graph     Graph in CSR format (must be on device)
 * @param source    Source node for BFS
 * @return          BFS result with distances and timing
 */
BFSResult *bfsChunked(CSRGraph *graph, node_t source);

/**
 * BFS kernel with chunk-based work distribution
 * Each thread processes a range of nodes [start_node, end_node)
 */
__global__ void bfsChunkedKernel(const edge_t *row_ptr, const node_t *col_idx,
                                 level_t *distances, int *changed,
                                 node_t num_nodes, level_t current_level);

#endif // BFS_CHUNKED_H
