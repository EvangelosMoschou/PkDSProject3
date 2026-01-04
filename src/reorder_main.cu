#include "common/graph.h"
#include "common/reorder.h"
#include "common/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Usage: %s <input_graph> <output_file> [method]\n", argv[0]);
    printf("Methods: bfs (default), degree\n");
    return 1;
  }

  const char *input_file = argv[1];
  const char *output_file = argv[2];
  const char *method_str = (argc > 3) ? argv[3] : "bfs";

  ReorderMethod method = REORDER_BFS;
  if (strcmp(method_str, "degree") == 0) {
    method = REORDER_DEGREE;
  } else if (strcmp(method_str, "bfs") != 0) {
    printf("Unknown method: %s\n", method_str);
    return 1;
  }

  printf("Loading Graph: %s\n", input_file);
  CSRGraph *graph = nullptr;

  // Auto-detect format based on extension logic mirrored from bfs_shared
  const char *ext = strrchr(input_file, '.');
  if (ext && strcmp(ext, ".csrbin") == 0)
    graph = loadGraphCSRBin(input_file);
  else if (ext && strcmp(ext, ".mat") == 0)
    graph = loadGraphHDF5(input_file);
  else
    graph = loadGraph(input_file);

  if (!graph) {
    printf("Failed to load graph.\n");
    return 1;
  }

  // Perform reordering and save (Streaming)
  CudaTimer timer = createTimer();
  startTimer(&timer);

  printf("Starting Streaming Reorder...\n");
  reorderAndSaveStreaming(graph, output_file, method);

  float elapsed = stopTimer(&timer);
  printf("Reordering + Saving Completed in %.2f ms.\n", elapsed);

  // Cleanup Original
  freeGraph(graph);

  return 0;

  return 0;
}
