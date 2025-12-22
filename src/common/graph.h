#ifndef GRAPH_H
#define GRAPH_H

#include "cuda_common.h"

// =============================================================================
// CSR Graph Structure
// =============================================================================

/**
 * CSR (Compressed Sparse Row) Graph Representation
 *
 * For a graph with N nodes and M edges:
 * - row_ptr: Array of size (N+1), row_ptr[i] gives the index in col_idx
 *            where the neighbors of node i start
 * - col_idx: Array of size M, contains the neighbor node IDs
 *
 * To iterate over neighbors of node i:
 *   for (int j = row_ptr[i]; j < row_ptr[i+1]; j++) {
 *       int neighbor = col_idx[j];
 *   }
 */
typedef struct {
  node_t num_nodes; // Number of nodes (vertices)
  edge_t num_edges; // Number of edges

  // CSR arrays (host)
  edge_t *h_row_ptr; // Row pointers [num_nodes + 1]
  node_t *h_col_idx; // Column indices [num_edges]

  // CSR arrays (device)
  edge_t *d_row_ptr; // Row pointers on GPU
  node_t *d_col_idx; // Column indices on GPU
} CSRGraph;

// =============================================================================
// Graph I/O Functions
// =============================================================================

/**
 * Load graph from file (edge list format)
 * File format: First line: num_nodes num_edges
 *              Following lines: source destination
 */
CSRGraph *loadGraph(const char *filename);

/**
 * Load graph from CSR binary file (faster loading)
 */
CSRGraph *loadGraphCSRBin(const char *filename);

/**
 * Save graph to CSR binary file
 */
void saveGraphCSRBin(const CSRGraph *graph, const char *filename);

/**
 * Generate a random graph for testing
 */
CSRGraph *generateRandomGraph(node_t num_nodes, edge_t avg_degree);

// =============================================================================
// Graph Memory Management
// =============================================================================

/**
 * Allocate and copy graph data to GPU
 */
void copyGraphToDevice(CSRGraph *graph);

/**
 * Free GPU graph memory
 */
void freeGraphDevice(CSRGraph *graph);

/**
 * Free entire graph (host and device)
 */
void freeGraph(CSRGraph *graph);

// =============================================================================
// Graph Utilities
// =============================================================================

/**
 * Print graph statistics
 */
void printGraphStats(const CSRGraph *graph);

/**
 * Validate graph structure
 */
bool validateGraph(const CSRGraph *graph);

/**
 * Get degree of a node
 */
inline edge_t getNodeDegree(const CSRGraph *graph, node_t node) {
  return graph->h_row_ptr[node + 1] - graph->h_row_ptr[node];
}

#endif // GRAPH_H
