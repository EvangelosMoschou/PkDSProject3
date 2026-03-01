#ifndef GRAPH_H
#define GRAPH_H

#include "cuda_common.h"
#include <stdint.h>

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

/**
 * Delta-Compressed CSR Graph (c-CSR)
 * Optimized for PCIe-constrained scenarios (Zero-Copy streaming).
 *
 * - row_ptr: Points to BYTE OFFSETS in compressed_col_idx (not index counts)
 * - compressed_col_idx: Varint-encoded delta sequence of neighbors
 */
typedef struct {
  node_t num_nodes;
  edge_t num_edges;
  size_t compressed_size_bytes;

  // Host Arrays
  edge_t *h_row_Ptr;         // [num_nodes + 1] (Byte offsets)
  uint8_t *h_compressed_col; // [compressed_size_bytes]

  // Device Arrays (Zero-Copy Mapped)
  edge_t *d_row_Ptr;         // Device pointer to row_ptr
  uint8_t *d_compressed_col; // Device pointer to compressed data
} CompressedCSRGraph;

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
 * Load graph from HDF5 file (MAT-File v7.3)
 */
CSRGraph *loadGraphHDF5(const char *filename);

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
 * Setup Zero-Copy mapping for Compressed CSR Graph
 */
void setupZeroCopyCompressed(CompressedCSRGraph *graph);

/**
 * Free CSR Graph resources (host and device)
 */
void freeGraph(CSRGraph *graph);
void setupCompressedGraphDevice(CompressedCSRGraph *graph);

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
