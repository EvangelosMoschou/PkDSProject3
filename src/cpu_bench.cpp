#include "common/graph.h"
#include "common/utils.h"
#include <chrono>
#include <queue>
#include <stdio.h>
#include <string>
#include <vector>

// Standalone CPU BFS for benchmarking
void benchCPU(const CSRGraph *graph, int source) {
  printf("Starting CPU BFS (Source: %d)...\n", source);

  // Use std::vector for distances (faster than new/delete raw array for C++)
  std::vector<int> distances(graph->num_nodes, -1);
  std::queue<int> q;

  auto start_time = std::chrono::high_resolution_clock::now();

  q.push(source);
  distances[source] = 0;

  node_t visited_count = 1;

  while (!q.empty()) {
    node_t u = q.front();
    q.pop();

    int dist_new = distances[u] + 1;
    edge_t start = graph->h_row_ptr[u];
    edge_t end = graph->h_row_ptr[u + 1];

    for (edge_t e = start; e < end; e++) {
      node_t v = graph->h_col_idx[e];
      if (distances[v] == -1) {
        distances[v] = dist_new;
        q.push(v);
        visited_count++;
      }
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

  printf("CPU BFS Completed.\n");
  printf("Time: %.2f ms\n", elapsed.count());
  printf("Visited: %d nodes\n", visited_count);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s <graph.csrbin>\n", argv[0]);
    return 1;
  }

  printf("Loading Graph: %s\n", argv[1]);
  CSRGraph *graph = loadGraphCSRBin(argv[1]);
  if (!graph) {
    printf("Failed to load graph.\n");
    return 1;
  }

  int source = 0;
  for (int i = 2; i < argc; i++) {
    if (std::string(argv[i]) == "--source" && i + 1 < argc) {
      source = std::atoi(argv[++i]);
    }
  }

  benchCPU(graph, source);

  freeGraph(graph);
  return 0;
}
