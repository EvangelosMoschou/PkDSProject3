#include "common/graph.h"
#include <stdio.h>

bool isSymmetric(CSRGraph *graph) {
  printf("Checking symmetry...\n");
  for (node_t u = 0; u < graph->num_nodes; u++) {
    for (edge_t e = graph->h_row_ptr[u]; e < graph->h_row_ptr[u + 1]; e++) {
      node_t v = graph->h_col_idx[e];
      // Check if (v, u) exists
      bool found = false;
      for (edge_t e2 = graph->h_row_ptr[v]; e2 < graph->h_row_ptr[v + 1];
           e2++) {
        if (graph->h_col_idx[e2] == u) {
          found = true;
          break;
        }
      }
      if (!found) {
        printf("Missing reverse edge: (%d, %d) for edge (%d, %d)\n", v, u, u,
               v);
        return false;
      }
    }
  }
  return true;
}

int main(int argc, char **argv) {
  if (argc < 2)
    return 1;
  CSRGraph *graph = loadGraphCSRBin(argv[1]);
  if (!graph)
    return 1;
  if (isSymmetric(graph))
    printf("Graph is SYMMETRIC\n");
  else
    printf("Graph is ASYMMETRIC\n");
  freeGraph(graph);
  return 0;
}
