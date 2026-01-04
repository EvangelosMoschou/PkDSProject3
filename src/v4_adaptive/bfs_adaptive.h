#ifndef BFS_ADAPTIVE_H
#define BFS_ADAPTIVE_H

#include "../common/graph.h"
#include "../common/utils.h"

// Solves BFS using Adaptive strategy (Thread/Warp/Block kernels)
BFSResult *solveBFSAdaptive(CSRGraph *graph, node_t source);

// Solves Compressed BFS using Adaptive strategy
BFSResult *solveBFSCompressedAdaptive(CompressedCSRGraph *graph, node_t source);

/**
 * Solve Connected Components using Afforest Algorithm
 */
void solveAfforest(CSRGraph *graph);
void solveAfforestCompressed(CompressedCSRGraph *graph);

#endif // BFS_ADAPTIVE_H
