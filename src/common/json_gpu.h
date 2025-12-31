#ifndef JSON_GPU_H
#define JSON_GPU_H

#include "cuda_common.h"

/**
 * Print benchmark results in JSON format.
 *
 * @param algo_name      Name of the algorithm (e.g., "Hybrid_BFS")
 * @param graph_file     Input graph filename
 * @param num_nodes      Number of nodes
 * @param num_edges      Number of edges
 * @param times_ms       Array of execution times in milliseconds
 * @param num_trials     Number of trials
 * @param traversed_edges Number of edges traversed (for TEPS calculation)
 * @param used_streaming Boolean, true if streaming kernel was used
 */
void print_json_gpu(const char *algo_name, const char *graph_file,
                    node_t num_nodes, edge_t num_edges, double *times_ms,
                    int num_trials, edge_t traversed_edges,
                    bool used_streaming);

#endif // JSON_GPU_H
