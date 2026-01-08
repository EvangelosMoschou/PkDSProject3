// Reorder Tool: Apply reordering and save to .csrbin
// Usage: ./reorder_graph <input.mat|.csrbin> <output.csrbin> [method]
//   method: 0=BFS, 1=DEGREE, 2=GAP_BFS (default)

#include "cuda_common.h"
#include "graph.h"
#include "reorder.h"
#include <cstring>
#include <stdio.h>

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: %s <input_graph> <output.csrbin> [method]\n", argv[0]);
    printf("  method: 0=BFS, 1=DEGREE, 2=GAP_BFS (default)\n");
    return 1;
  }

  const char *input_file = argv[1];
  const char *output_file = argv[2];

  ReorderMethod method = REORDER_GAP_BFS; // Default to Gap-Aware BFS
  if (argc >= 4) {
    int m = atoi(argv[3]);
    if (m == 0)
      method = REORDER_BFS;
    else if (m == 1)
      method = REORDER_DEGREE;
    else
      method = REORDER_GAP_BFS;
  }

  printf("=== Graph Reordering Tool ===\n");
  printf("Input:  %s\n", input_file);
  printf("Output: %s\n", output_file);
  printf("Method: %s\n", method == REORDER_BFS      ? "Standard BFS"
                         : method == REORDER_DEGREE ? "Degree Sort"
                                                    : "Gap-Aware BFS");
  printf("\n");

  // Detect file type and load appropriately
  CSRGraph *graph = nullptr;
  const char *ext = strrchr(input_file, '.');

  if (ext && strcmp(ext, ".mat") == 0) {
    printf("Loading HDF5 (.mat) file...\n");
    graph = loadGraphHDF5(input_file);
  } else if (ext && strcmp(ext, ".csrbin") == 0) {
    printf("Loading CSR binary file...\n");
    graph = loadGraphCSRBin(input_file);
  } else {
    printf("Loading text edge list file...\n");
    graph = loadGraph(input_file);
  }

  if (!graph) {
    fprintf(stderr, "Error: Failed to load graph from %s\n", input_file);
    return 1;
  }

  printf("\nGraph loaded: %d nodes, %lld edges\n\n", graph->num_nodes,
         (long long)graph->num_edges);

  // Reorder and save
  reorderAndSaveStreaming(graph, output_file, method);

  printf("\nReordering complete. Output saved to: %s\n", output_file);

  // Cleanup
  freeGraph(graph);
  return 0;
}
