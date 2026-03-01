#ifndef BFS_ADAPTIVE_H
#define BFS_ADAPTIVE_H

#include "../common/graph.h"
#include "../common/utils.h"

// Solves BFS using Adaptive strategy (Single GPU)
BFSResult *solveBFSAdaptive(CSRGraph *graph, node_t source);

BFSResult *solveBFSAdaptiveWithThreshold(CSRGraph *graph, node_t source,
                                         int bu_threshold_divisor);

// Multi-GPU Simulated Solver
BFSResult *solveBFSMultiGPUSimulated(CSRGraph *graph, node_t source,
                                     int num_gpus);

BFSResult *solveBFSCompressedAdaptive(CompressedCSRGraph *graph, node_t source);
BFSResult *solveBFSCompressedMultiGPUSimulated(CompressedCSRGraph *graph,
                                               node_t source, int num_gpus);

void solveAfforest(CSRGraph *graph);

#endif // BFS_ADAPTIVE_H
