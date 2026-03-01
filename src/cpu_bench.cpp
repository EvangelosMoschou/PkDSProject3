#include "common/graph.h"
#include "common/utils.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Standalone CPU BFS for benchmarking
void benchCPU(const CSRGraph *graph, int source) {
  printf("Starting CPU BFS (Source: %d)...\n", source);

  int *distances = (int *)malloc((size_t)graph->num_nodes * sizeof(int));
  node_t *queue = (node_t *)malloc((size_t)graph->num_nodes * sizeof(node_t));
  if (!distances || !queue) {
    fprintf(stderr, "Allocation failed in CPU benchmark.\n");
    free(distances);
    free(queue);
    return;
  }
  for (node_t i = 0; i < graph->num_nodes; i++) {
    distances[i] = -1;
  }

  struct timespec start_time;
  struct timespec end_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  node_t head = 0;
  node_t tail = 0;
  queue[tail++] = (node_t)source;
  distances[source] = 0;

  node_t visited_count = 1;

  while (head < tail) {
    node_t u = queue[head++];

    int dist_new = distances[u] + 1;
    edge_t start = graph->h_row_ptr[u];
    edge_t end = graph->h_row_ptr[u + 1];

    for (edge_t e = start; e < end; e++) {
      node_t v = graph->h_col_idx[e];
      if (distances[v] == -1) {
        distances[v] = dist_new;
        queue[tail++] = v;
        visited_count++;
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &end_time);
  double elapsed_ms =
      (double)(end_time.tv_sec - start_time.tv_sec) * 1000.0 +
      (double)(end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;

  printf("CPU BFS Completed.\n");
  printf("Time: %.2f ms\n", elapsed_ms);
  printf("Visited: %d nodes\n", visited_count);

  free(distances);
  free(queue);
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
    if (strcmp(argv[i], "--source") == 0 && i + 1 < argc) {
      source = atoi(argv[++i]);
    }
  }

  benchCPU(graph, source);

  freeGraph(graph);
  return 0;
}
