#include "cuda_common.h"
#include "reorder.h"
#include <algorithm>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;

// Helper: Get max degree node (good BFS starting point)
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

// =============================================================================
// Standard BFS Order (RCM-like, but without degree sorting or reversal)
// =============================================================================
void computeBFSOrder(const CSRGraph *graph, std::vector<node_t> &new_to_old,
                     std::vector<node_t> &old_to_new) {
  node_t N = graph->num_nodes;
  std::fill(old_to_new.begin(), old_to_new.end(), (node_t)-1);
  std::fill(new_to_old.begin(), new_to_old.end(), (node_t)-1);

  vector<bool> visited(N, false);
  queue<node_t> q;
  node_t new_id_counter = 0;

  node_t start_node = findMaxDegreeNode(graph);
  q.push(start_node);
  visited[start_node] = true;

  while (new_id_counter < N) {
    if (q.empty()) {
      for (node_t i = 0; i < N; i++) {
        if (!visited[i]) {
          q.push(i);
          visited[i] = true;
          break;
        }
      }
    }
    if (q.empty())
      break;

    node_t u = q.front();
    q.pop();

    node_t nid = new_id_counter++;
    old_to_new[u] = nid;
    new_to_old[nid] = u;

    edge_t start = graph->h_row_ptr[u];
    edge_t end = graph->h_row_ptr[u + 1];

    for (edge_t e = start; e < end; e++) {
      node_t v = graph->h_col_idx[e];
      if (!visited[v]) {
        visited[v] = true;
        q.push(v);
      }
    }
  }
}

// =============================================================================
// Gap-Aware BFS Order
// Key insight: Sort neighbors by ORIGINAL ID before enqueueing
// This keeps nodes that were originally close together (small gaps)
// assigned to consecutive new IDs, preserving compression efficiency
// =============================================================================
void computeGapAwareBFSOrder(const CSRGraph *graph,
                             std::vector<node_t> &new_to_old,
                             std::vector<node_t> &old_to_new) {
  node_t N = graph->num_nodes;
  std::fill(old_to_new.begin(), old_to_new.end(), (node_t)-1);
  std::fill(new_to_old.begin(), new_to_old.end(), (node_t)-1);

  vector<bool> visited(N, false);
  queue<node_t> q;
  node_t new_id_counter = 0;

  // Start from node 0 (often a good choice for preserving original locality)
  // Alternatively, could use findMaxDegreeNode, but that might scatter things
  node_t start_node = 0;
  q.push(start_node);
  visited[start_node] = true;

  // Temporary buffer for sorting neighbors
  vector<node_t> neighbors;
  neighbors.reserve(10000); // Pre-allocate for efficiency

  printf("  Computing Gap-Aware BFS Order...\n");

  while (new_id_counter < N) {
    if (q.empty()) {
      // Find unvisited node - pick the SMALLEST unvisited ID to preserve
      // locality
      for (node_t i = 0; i < N; i++) {
        if (!visited[i]) {
          q.push(i);
          visited[i] = true;
          break;
        }
      }
    }
    if (q.empty())
      break;

    node_t u = q.front();
    q.pop();

    node_t nid = new_id_counter++;
    old_to_new[u] = nid;
    new_to_old[nid] = u;

    // Collect unvisited neighbors
    edge_t start = graph->h_row_ptr[u];
    edge_t end = graph->h_row_ptr[u + 1];

    neighbors.clear();
    for (edge_t e = start; e < end; e++) {
      node_t v = graph->h_col_idx[e];
      if (!visited[v]) {
        neighbors.push_back(v);
        visited[v] = true; // Mark visited now to avoid duplicates
      }
    }

    // KEY DIFFERENCE: Sort neighbors by original ID (ascending)
    // This ensures that nodes originally close together get consecutive new IDs
    std::sort(neighbors.begin(), neighbors.end());

    // Enqueue in sorted order
    for (node_t v : neighbors) {
      q.push(v);
    }

    // Progress indicator
    if (new_id_counter % 10000000 == 0) {
      printf("    Processed %d / %d nodes (%.1f%%)\n", new_id_counter, N,
             100.0 * new_id_counter / N);
    }
  }
  printf("  Gap-Aware BFS Order complete.\n");
}

// =============================================================================
// Degree Order (sort by degree descending)
// =============================================================================
void computeDegreeOrder(const CSRGraph *graph, std::vector<node_t> &new_to_old,
                        std::vector<node_t> &old_to_new) {
  node_t N = graph->num_nodes;
  std::fill(old_to_new.begin(), old_to_new.end(), (node_t)-1);
  std::fill(new_to_old.begin(), new_to_old.end(), (node_t)-1);

  vector<pair<edge_t, node_t>> nodes(N);
  for (node_t i = 0; i < N; i++) {
    nodes[i] = {graph->h_row_ptr[i + 1] - graph->h_row_ptr[i], i};
  }
  sort(nodes.rbegin(), nodes.rend());

  for (node_t i = 0; i < N; i++) {
    node_t old_id = nodes[i].second;
    old_to_new[old_id] = i;
    new_to_old[i] = old_id;
  }
}

// =============================================================================
// Streaming Implementation (Memory-efficient disk output)
// =============================================================================
void reorderAndSaveStreaming(const CSRGraph *graph, const char *out_filename,
                             ReorderMethod method) {
  printf("Reordering Graph: %d Nodes, %lld Edges. Method: %d\n",
         graph->num_nodes, (long long)graph->num_edges, (int)method);

  std::vector<node_t> new_to_old(graph->num_nodes);
  std::vector<node_t> old_to_new(graph->num_nodes);

  if (method == REORDER_BFS) {
    printf("  Computing Standard BFS Order...\n");
    computeBFSOrder(graph, new_to_old, old_to_new);
  } else if (method == REORDER_GAP_BFS) {
    computeGapAwareBFSOrder(graph, new_to_old, old_to_new);
  } else {
    printf("  Computing Degree Order...\n");
    computeDegreeOrder(graph, new_to_old, old_to_new);
  }

  printf("  Streaming Reordered Graph to Disk: %s\n", out_filename);
  FILE *file = fopen(out_filename, "wb");
  if (!file) {
    fprintf(stderr, "Error: Cannot create file %s\n", out_filename);
    return;
  }

  // Write header
  unsigned long long n = (unsigned long long)graph->num_nodes;
  unsigned long long m = (unsigned long long)graph->num_edges;
  fwrite(&n, sizeof(unsigned long long), 1, file);
  fwrite(&m, sizeof(unsigned long long), 1, file);

  // Compute new row pointers
  edge_t *new_row_ptr = new edge_t[graph->num_nodes + 1];
  new_row_ptr[0] = 0;

  for (node_t i = 0; i < graph->num_nodes; i++) {
    node_t u_old = new_to_old[i];
    edge_t degree = graph->h_row_ptr[u_old + 1] - graph->h_row_ptr[u_old];
    new_row_ptr[i + 1] = new_row_ptr[i] + degree;
  }

  fwrite(new_row_ptr, sizeof(edge_t), graph->num_nodes + 1, file);

  // Stream column indices with sorting for better compression
  size_t buffer_cap = 1024 * 1024 * 64;
  node_t *buffer = new node_t[buffer_cap];
  size_t buffer_idx = 0;

  vector<node_t> row_neighbors;
  row_neighbors.reserve(10000);

  printf("  Streaming Edges... (Total: %lld)\n", (long long)m);

  for (node_t i = 0; i < graph->num_nodes; i++) {
    node_t u_old = new_to_old[i];
    edge_t start = graph->h_row_ptr[u_old];
    edge_t end = graph->h_row_ptr[u_old + 1];

    // Collect and translate neighbors
    row_neighbors.clear();
    for (edge_t e = start; e < end; e++) {
      node_t v_old = graph->h_col_idx[e];
      node_t v_new = old_to_new[v_old];
      row_neighbors.push_back(v_new);
    }

    // Sort neighbors by NEW ID for optimal delta compression
    std::sort(row_neighbors.begin(), row_neighbors.end());

    // Write to buffer
    for (node_t v_new : row_neighbors) {
      buffer[buffer_idx++] = v_new;
      if (buffer_idx >= buffer_cap) {
        fwrite(buffer, sizeof(node_t), buffer_cap, file);
        buffer_idx = 0;
      }
    }

    if (i % 5000000 == 0 && i > 0) {
      printf("\r    Processed %d / %d Nodes (%.1f%%)", i, graph->num_nodes,
             100.0 * i / graph->num_nodes);
      fflush(stdout);
    }
  }

  if (buffer_idx > 0) {
    fwrite(buffer, sizeof(node_t), buffer_idx, file);
  }
  printf("\n  Done.\n");

  delete[] buffer;
  delete[] new_row_ptr;
  fclose(file);
}

// =============================================================================
// In-Memory Reorder (for smaller graphs or when memory permits)
// =============================================================================
CSRGraph *reorderGraph(const CSRGraph *graph, ReorderMethod method) {
  if (graph->num_edges > 1000000000LL) {
    fprintf(stderr, "WARNING: reorderGraph on massive graph. Use streaming.\n");
  }

  printf("Reordering Graph: %d Nodes, %lld Edges. Method: %d\n",
         graph->num_nodes, (long long)graph->num_edges, (int)method);

  std::vector<node_t> new_to_old(graph->num_nodes);
  std::vector<node_t> old_to_new(graph->num_nodes);

  if (method == REORDER_BFS) {
    printf("  Computing Standard BFS Order...\n");
    computeBFSOrder(graph, new_to_old, old_to_new);
  } else if (method == REORDER_GAP_BFS) {
    computeGapAwareBFSOrder(graph, new_to_old, old_to_new);
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

  // Fill Row Pointers
  new_graph->h_row_ptr[0] = 0;
  for (node_t i = 0; i < graph->num_nodes; i++) {
    node_t u_old = new_to_old[i];
    edge_t degree = graph->h_row_ptr[u_old + 1] - graph->h_row_ptr[u_old];
    new_graph->h_row_ptr[i + 1] = new_graph->h_row_ptr[i] + degree;
  }

  // Fill Col Indices (with sorting for compression)
  vector<node_t> row_neighbors;
  row_neighbors.reserve(10000);

  for (node_t i = 0; i < graph->num_nodes; i++) {
    node_t u_old = new_to_old[i];
    edge_t start = graph->h_row_ptr[u_old];
    edge_t end = graph->h_row_ptr[u_old + 1];
    edge_t new_start = new_graph->h_row_ptr[i];

    // Collect and translate
    row_neighbors.clear();
    for (edge_t e = start; e < end; e++) {
      node_t v_old = graph->h_col_idx[e];
      row_neighbors.push_back(old_to_new[v_old]);
    }

    // Sort by new ID for optimal compression
    std::sort(row_neighbors.begin(), row_neighbors.end());

    // Write
    for (size_t j = 0; j < row_neighbors.size(); j++) {
      new_graph->h_col_idx[new_start + j] = row_neighbors[j];
    }
  }

  return new_graph;
}
