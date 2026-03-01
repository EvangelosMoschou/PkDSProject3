#ifndef UTILS_H
#define UTILS_H

#include "cuda_common.h"
#include "graph.h"
// =============================================================================
// Timer Utilities
// =============================================================================

typedef struct {
  cudaEvent_t start;
  cudaEvent_t stop;
} CudaTimer;

/**
 * Create a CUDA timer
 */
CudaTimer createTimer();

/**
 * Start timing
 */
void startTimer(CudaTimer *timer);

/**
 * Stop timing and return elapsed time in milliseconds
 */
float stopTimer(CudaTimer *timer);

/**
 * Destroy timer
 */
void destroyTimer(CudaTimer *timer);

// =============================================================================
// BFS Result Utilities
// =============================================================================

/**
 * BFS result structure
 */
typedef struct {
  level_t *distances; // Distance from source for each node
  node_t *parents;    // Parent node in BFS tree (optional)
  node_t num_nodes;
  node_t source;
  float elapsed_ms; // Execution time
} BFSResult;

/**
 * Allocate BFS result on host
 */
BFSResult *allocBFSResult(node_t num_nodes, node_t source);

/**
 * Free BFS result
 */
void freeBFSResult(BFSResult *result);

/**
 * Print BFS result summary
 */
void printBFSResult(const BFSResult *result);

/**
 * Validate BFS result against CPU reference
 */
bool validateBFSResult(const BFSResult *result, const CSRGraph *graph);

// =============================================================================
// CPU BFS Reference Implementation
// =============================================================================

/**
 * Sequential BFS for validation
 */
BFSResult *bfsCPU(const CSRGraph *graph, node_t source);

// =============================================================================
// Command Line Parsing
// =============================================================================

typedef enum { ALGO_BFS, ALGO_AFFOREST, ALGO_ADAPTIVE } AlgorithmType;

typedef struct {
  char *graph_file;
  node_t source;
  bool validate;
  bool verbose;
  bool benchmark;
  int num_runs;
  bool json_output;
  bool compression;
  AlgorithmType algorithm;
  int bu_threshold_divisor; // Bottom-Up threshold = num_nodes /
                            // bu_threshold_divisor (default: 20 = 5%)
} BFSOptions;

/**
 * Parse command line arguments
 */
BFSOptions parseArgs(int argc, char **argv);

/**
 * Print usage information
 */
void printUsage(const char *program);

#endif // UTILS_H
