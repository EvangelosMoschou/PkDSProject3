#ifndef BFS_ADAPTIVE_H
#define BFS_ADAPTIVE_H

#include "../common/graph.h"
#include "../common/utils.h"

// Solves BFS using Adaptive strategy (Thread/Warp/Block kernels)
// Uses default Bottom-Up threshold divisor of 20 (5%)
BFSResult *solveBFSAdaptive(CSRGraph *graph, node_t source);

// Solves BFS using Adaptive strategy with configurable Bottom-Up threshold
// bu_threshold_divisor: Switch to Bottom-Up when frontier > num_nodes / divisor
BFSResult *solveBFSAdaptiveWithThreshold(CSRGraph *graph, node_t source,
                                         int bu_threshold_divisor);

// Solves Compressed BFS using Adaptive strategy
BFSResult *solveBFSCompressedAdaptive(CompressedCSRGraph *graph, node_t source);

/**
 * Solve Connected Components using Afforest Algorithm
 */
void solveAfforest(CSRGraph *graph);
void solveAfforestCompressed(CompressedCSRGraph *graph);

#endif // BFS_ADAPTIVE_H
