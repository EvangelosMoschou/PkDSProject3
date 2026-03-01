#ifndef REORDER_H
#define REORDER_H

#include "graph.h"

// Reordering methods
typedef enum {
  REORDER_BFS,    // Standard BFS order (like RCM without reversal)
  REORDER_DEGREE, // Sort by degree descending
  REORDER_GAP_BFS // Gap-Aware BFS: preserves neighbor locality for compression
} ReorderMethod;

// Compute reordering and save to disk (streaming, memory-efficient)
void reorderAndSaveStreaming(const CSRGraph *graph, const char *out_filename,
                             ReorderMethod method);

// In-memory reordering (for smaller graphs)
CSRGraph *reorderGraph(const CSRGraph *graph, ReorderMethod method);

#endif // REORDER_H
