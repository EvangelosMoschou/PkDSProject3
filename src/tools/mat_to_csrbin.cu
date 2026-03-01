#include "common/graph.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr,
            "Usage: %s <input.mat|input.h5> <output.csrbin>\n",
            argv[0]);
    return 1;
  }

  const char *input_file = argv[1];
  const char *output_file = argv[2];

  if (strstr(input_file, ".mat") == NULL && strstr(input_file, ".h5") == NULL) {
    fprintf(stderr, "Error: input must be .mat or .h5: %s\n", input_file);
    return 1;
  }

  printf("Loading graph from %s...\n", input_file);
  CSRGraph *graph = loadGraphHDF5(input_file);
  if (!graph) {
    fprintf(stderr, "Error: failed to load input graph %s\n", input_file);
    return 1;
  }

  printf("Saving CSR binary to %s...\n", output_file);
  saveGraphCSRBin(graph, output_file);

  freeGraph(graph);
  printf("Done.\n");
  return 0;
}
