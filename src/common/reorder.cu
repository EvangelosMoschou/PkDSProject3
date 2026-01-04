#include "cuda_common.h"
#include "reorder.h"
#include <algorithm>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;

// Helper: Get max degree node
node_t findMaxDegreeNode(const CSRGraph *g) {
  node_t max_node = 0;
  edge_t max_deg = 0;
  for (node_t i = 0; i < g->num_nodes; i++) {
    edge_t deg = g->h_row_ptr[i + 1] - g->h_row_ptr[i];
    if (deg > max_deg) {
      max_deg = deg;
      max_node = i;
    }
  }
  return max_node;
}

// Helper functions for computing orders (assumed to be defined elsewhere or
// added later) For now, I'll put the original logic into these helper
// functions.
void computeBFSOrder(const CSRGraph *graph, std::vector<node_t> &new_to_old,
                     std::vector<node_t> &old_to_new) {
  node_t N = graph->num_nodes;
  std::fill(old_to_new.begin(), old_to_new.end(), (node_t)-1);
  std::fill(new_to_old.begin(), new_to_old.end(), (node_t)-1);

  vector<bool> visited(N, false);
  queue<node_t> q;
  node_t new_id_counter = 0;

  // Start from max degree node to hit hub first
  node_t start_node = findMaxDegreeNode(graph);

  q.push(start_node);
  visited[start_node] = true;

  // Main BFS Loop
  while (new_id_counter < N) {
    if (q.empty()) {
      // Find unvisited node for disconnected components
      for (node_t i = 0; i < N; i++) {
        if (!visited[i]) {
          q.push(i);
          visited[i] = true;
          break;
        }
      }
    }

    if (q.empty())
      break; // Should be done

    node_t u = q.front();
    q.pop();

    node_t nid = new_id_counter++;
    old_to_new[u] = nid;
    new_to_old[nid] = u;

    // Enqueue neighbors
    // Access Original Neighbors
    edge_t start = graph->h_row_ptr[u];
    edge_t end = graph->h_row_ptr[u + 1];

    // Optimization: Sort neighbors by degree? No, standard BFS is fine for
    // RCM-like.
    for (edge_t e = start; e < end; e++) {
      node_t v = graph->h_col_idx[e];
      if (!visited[v]) {
        visited[v] = true;
        q.push(v);
      }
    }
  }
}

void computeDegreeOrder(const CSRGraph *graph, std::vector<node_t> &new_to_old,
                        std::vector<node_t> &old_to_new) {
  node_t N = graph->num_nodes;
  std::fill(old_to_new.begin(), old_to_new.end(), (node_t)-1);
  std::fill(new_to_old.begin(), new_to_old.end(), (node_t)-1);

  vector<pair<edge_t, node_t>> nodes(N);
  for (node_t i = 0; i < N; i++) {
    nodes[i] = {graph->h_row_ptr[i + 1] - graph->h_row_ptr[i], i};
  }
  // Sort descending
  sort(nodes.rbegin(), nodes.rend());

  for (node_t i = 0; i < N; i++) {
    node_t old_id = nodes[i].second;
    old_to_new[old_id] = i;
    new_to_old[i] = old_id;
  }
}

// -----------------------------------------------------------------------------
// Streaming Implementation
// -----------------------------------------------------------------------------

void reorderAndSaveStreaming(const CSRGraph *graph, const char *out_filename,
                             ReorderMethod method) {
  printf("Reordering Graph: %d Nodes, %lld Edges. Method: %d\n",
         graph->num_nodes, (long long)graph->num_edges, (int)method);

  // 1. Compute Permutation
  std::vector<node_t> new_to_old(graph->num_nodes);
  std::vector<node_t> old_to_new(graph->num_nodes);

  if (method == REORDER_BFS) {
    printf("  Computing BFS Order...\n");
    computeBFSOrder(graph, new_to_old, old_to_new);
  } else { // REORDER_DEGREE
    printf("  Computing Degree Order...\n");
    computeDegreeOrder(graph, new_to_old, old_to_new);
  }

  // 2. Open Output File
  printf("  Streaming Reordered Graph to Disk: %s\n", out_filename);
  FILE *file = fopen(out_filename, "wb");
  if (!file) {
    fprintf(stderr, "Error: Cannot create file %s\n", out_filename);
    return;
  }

  // 3. Write Header
  unsigned long long n = (unsigned long long)graph->num_nodes;
  unsigned long long m = (unsigned long long)graph->num_edges;
  fwrite(&n, sizeof(unsigned long long), 1, file);
  fwrite(&m, sizeof(unsigned long long), 1, file);

  // 4. Compute and Write Row Pointers
  // We need to write the FULL row_ptr array first (N+1 elements).
  // We can compute it in memory (N*8 bytes ~ 0.5GB for Friendster) which fits.
  // Note: graph->h_row_ptr is edge_t (unsigned long long)

  // Allocate new_row_ptr
  edge_t *new_row_ptr = new edge_t[graph->num_nodes + 1];
  new_row_ptr[0] = 0;

  for (node_t i = 0; i < graph->num_nodes; i++) {
    node_t u_old = new_to_old[i];
    edge_t degree = graph->h_row_ptr[u_old + 1] - graph->h_row_ptr[u_old];
    new_row_ptr[i + 1] = new_row_ptr[i] + degree;
  }

  // Validate total edges
  if (new_row_ptr[graph->num_nodes] != graph->num_edges) {
    fprintf(stderr, "Error: Edge count mismatch! Calced: %lld, Orig: %lld\n",
            (long long)new_row_ptr[graph->num_nodes],
            (long long)graph->num_edges);
  }

  fwrite(new_row_ptr, sizeof(edge_t), graph->num_nodes + 1, file);

  // 5. Stream Column Indices
  // We can reuse new_row_ptr to speed up writing? No, we just iterate.
  // We need a buffer to minimize write calls.
  size_t buffer_cap = 1024 * 1024 * 64; // 64 million edges buffer (~256MB)
  node_t *buffer = new node_t[buffer_cap];
  size_t buffer_idx = 0;

  printf("  Streaming Edges... (Total: %lld)\n", (long long)m);

  for (node_t i = 0; i < graph->num_nodes; i++) {
    node_t u_old = new_to_old[i];
    edge_t start = graph->h_row_ptr[u_old];
    edge_t end = graph->h_row_ptr[u_old + 1];

    for (edge_t e = start; e < end; e++) {
      node_t v_old = graph->h_col_idx[e];
      node_t v_new = old_to_new[v_old];

      buffer[buffer_idx++] = v_new;

      if (buffer_idx >= buffer_cap) {
        fwrite(buffer, sizeof(node_t), buffer_cap, file);
        buffer_idx = 0;
        // Progress bar?
        if (i % 1000000 == 0)
          printf("\r    Processed %d / %d Nodes", i, graph->num_nodes);
      }
    }
  }

  // Flash remaining buffer
  if (buffer_idx > 0) {
    fwrite(buffer, sizeof(node_t), buffer_idx, file);
  }
  printf("\n  Done.\n");

  // Cleanup
  delete[] buffer;
  delete[] new_row_ptr;
  fclose(file);
}

// -----------------------------------------------------------------------------
// Legacy / Fallback In-Memory Reorder (Used directly by other tools if needed)
// -----------------------------------------------------------------------------
CSRGraph *reorderGraph(const CSRGraph *graph, ReorderMethod method) {
  if (graph->num_edges > 1000000000LL) {
    fprintf(stderr, "WARNING: reorderGraph called on massive graph. Use "
                    "reorderAndSaveStreaming instead.\n");
  }

  printf("Reordering Graph: %d Nodes, %lld Edges. Method: %d\n",
         graph->num_nodes, (long long)graph->num_edges, (int)method);

  std::vector<node_t> new_to_old(graph->num_nodes);
  std::vector<node_t> old_to_new(graph->num_nodes);

  if (method == REORDER_BFS) {
    printf("  Computing BFS Order...\n");
    computeBFSOrder(graph, new_to_old, old_to_new);
  } else {
    printf("  Computing Degree Order...\n");
    computeDegreeOrder(graph, new_to_old, old_to_new);
  }

  printf("  Reconstructing Graph...\n");
  CSRGraph *new_graph = new CSRGraph;
  new_graph->num_nodes = graph->num_nodes;
  new_graph->num_edges = graph->num_edges;

  CUDA_CHECK(cudaMallocHost(&new_graph->h_row_ptr,
                            (new_graph->num_nodes + 1) * sizeof(edge_t)));
  CUDA_CHECK(cudaMallocHost(&new_graph->h_col_idx,
                            new_graph->num_edges * sizeof(node_t)));

  if (!new_graph->h_row_ptr || !new_graph->h_col_idx) {
    fprintf(stderr, "Error: Failed to allocate memory for reordered graph.\n");
    // If we fail here, we should probably clean up and return nullptr
    delete new_graph;
    return nullptr;
  }

  // Fill Row Pointers
  new_graph->h_row_ptr[0] = 0;
  for (node_t i = 0; i < graph->num_nodes; i++) {
    node_t u_old = new_to_old[i];
    edge_t degree = graph->h_row_ptr[u_old + 1] - graph->h_row_ptr[u_old];
    new_graph->h_row_ptr[i + 1] = new_graph->h_row_ptr[i] + degree;
  }

// Fill Col Indices
#pragma omp parallel for
  for (node_t i = 0; i < graph->num_nodes; i++) {
    node_t u_old = new_to_old[i];
    edge_t start = graph->h_row_ptr[u_old];
    edge_t end = graph->h_row_ptr[u_old + 1];
    edge_t new_start = new_graph->h_row_ptr[i];

    for (edge_t e = 0; e < (end - start); e++) {
      node_t v_old = graph->h_col_idx[start + e];
      new_graph->h_col_idx[new_start + e] = old_to_new[v_old];
    }
  }

  return new_graph;
}
