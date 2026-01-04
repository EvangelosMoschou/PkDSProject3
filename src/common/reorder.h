#ifndef REORDER_H
#define REORDER_H

#include "graph.h"
#include <string>
#include <vector>

enum ReorderMethod { REORDER_BFS, REORDER_DEGREE, REORDER_RCM, REORDER_NONE };

// Creates a new reordered graph. The caller is responsible for freeing the
// result.
CSRGraph *reorderGraph(const CSRGraph *graph, ReorderMethod method);
void reorderAndSaveStreaming(const CSRGraph *graph, const char *out_filename,
                             ReorderMethod method);

#endif // REORDER_H
