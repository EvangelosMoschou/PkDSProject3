#include "common/graph.h"
#include <iostream>
#include <set>

bool isSymmetric(CSRGraph *graph) {
  std::cout << "Checking symmetry..." << std::endl;
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
        std::cout << "Missing reverse edge: (" << v << ", " << u
                  << ") for edge (" << u << ", " << v << ")" << std::endl;
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
    std::cout << "Graph is SYMMETRIC" << std::endl;
  else
    std::cout << "Graph is ASYMMETRIC" << std::endl;
  return 0;
}
