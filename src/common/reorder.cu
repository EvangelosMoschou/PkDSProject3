#include "cuda_common.h"
#include "reorder.h"
#include <algorithm>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;

// =============================================================================
// Helper Functions
// =============================================================================

// Get degree of a node
inline edge_t getDegree(const CSRGraph *g, node_t u) {
  return g->h_row_ptr[u + 1] - g->h_row_ptr[u];
}

// Find a pseudo-peripheral node using repeated BFS
// Algorithm:
// 1. Start from arbitrary node u
// 2. Perform BFS to find node v at max distance (eccentricity)
// 3. Perform BFS from v to find node w at max distance
// 4. If dist(v, w) > dist(u, v), set u=v, v=w, repeat
// 5. Else, returns v as the peripheral node
node_t findPseudoPeripheralNode(const CSRGraph *g) {
  node_t num_nodes = g->num_nodes;
  if (num_nodes == 0)
    return 0;

  // Start with node having minimum degree (heuristic)
  node_t start_node = 0;
  edge_t min_deg = getDegree(g, 0);

  for (node_t i = 1; i < min(num_nodes, (node_t)100); i++) { // Check first 100
    edge_t deg = getDegree(g, i);
    if (deg < min_deg) {
      min_deg = deg;
      start_node = i;
    }
  }

  // Helper for BFS
  vector<bool> visited(num_nodes);
  vector<node_t> q;
  q.reserve(num_nodes);

  node_t u = start_node;
  int max_dist = -1;

  for (int iter = 0; iter < 5; iter++) { // Limit iterations
    fill(visited.begin(), visited.end(), false);
    q.clear();

    q.push_back(u);
    visited[u] = true;

    int current_level_dist = 0;
    size_t head = 0;
    node_t last_node = u;

    // Simple BFS to find farthest node
    while (head < q.size()) {
      // In strict BFS, we would track levels.
      // Here we just want the *last* node visited, which is approx farthest.
      // To get actual distance, we need level tracking, but "last visited"
      // is a good enough proxy for peripheral node in connected components.
      node_t curr = q[head++];
      last_node = curr;

      edge_t start_edge = g->h_row_ptr[curr];
      edge_t end_edge = g->h_row_ptr[curr + 1];

      for (edge_t e = start_edge; e < end_edge; e++) {
        node_t neighbor = g->h_col_idx[e];
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          q.push_back(neighbor);
        }
      }
    }

    // Now last_node is the farthest
    if (last_node == u)
      break; // Single node component

    // If q.size() (width) is narrow, it's good?
    // Actually we just swap and retry from the far end.
    node_t v = last_node;
    u = v;
  }

  return u;
}

// =============================================================================
// Reordering Algorithms
// =============================================================================

// Reverse Cuthill-McKee (RCM)
// 1. Find pseudo-peripheral start node
// 2. BFS, but visit neighbors in ASCENDING order of degree
// 3. Reverse the result array
void computeRCM(const CSRGraph *g, vector<node_t> &new_to_old) {
  printf("  [RCM] Finding pseudo-peripheral start node...\n");
  node_t start_node = findPseudoPeripheralNode(g);
  printf("  [RCM] Start Node: %d (Degree: %ld)\n", start_node,
         getDegree(g, start_node));

  node_t num_nodes = g->num_nodes;
  vector<bool> visited(num_nodes, false);

  new_to_old.clear();
  new_to_old.reserve(num_nodes);

  // Queue for BFS
  // Note: we need to sort neighbors, not the queue itself.
  // Standard queue is fine.
  deque<node_t> q;

  q.push_back(start_node);
  visited[start_node] = true;
  new_to_old.push_back(start_node);

  // Process connected component
  while (!q.empty()) {
    node_t u = q.front();
    q.pop_front();

    // Collect unvisited neighbors
    edge_t start_edge = g->h_row_ptr[u];
    edge_t end_edge = g->h_row_ptr[u + 1];
    vector<node_t> neighbors;

    for (edge_t e = start_edge; e < end_edge; e++) {
      node_t v = g->h_col_idx[e];
      if (!visited[v]) {
        visited[v] = true;
        neighbors.push_back(v);
        new_to_old.push_back(v); // Add to permutation immediately
      }
    }

    // Sort neighbors by degree (ASCENDING)
    // "Quiet" nodes first -> keeps wavefront narrow
    sort(neighbors.begin(), neighbors.end(),
         [&](node_t a, node_t b) { return getDegree(g, a) < getDegree(g, b); });

    // Add to queue
    for (node_t v : neighbors) {
      q.push_back(v);
    }
  }

  // Handle disconnected components
  // If graph is not connected, run RCM on other components too
  for (node_t i = 0; i < num_nodes; i++) {
    if (!visited[i]) {
      // Start a new RCM traversal from this unvisited node
      // Ideally find peripheral for this component too, but for speed just use
      // i
      q.push_back(i);
      visited[i] = true;
      new_to_old.push_back(i);

      while (!q.empty()) {
        node_t u = q.front();
        q.pop_front();

        edge_t start_edge = g->h_row_ptr[u];
        edge_t end_edge = g->h_row_ptr[u + 1];
        vector<node_t> neighbors;

        for (edge_t e = start_edge; e < end_edge; e++) {
          node_t v = g->h_col_idx[e];
          if (!visited[v]) {
            visited[v] = true;
            neighbors.push_back(v);
            new_to_old.push_back(v);
          }
        }
        sort(neighbors.begin(), neighbors.end(), [&](node_t a, node_t b) {
          return getDegree(g, a) < getDegree(g, b);
        });
        for (node_t v : neighbors) {
          q.push_back(v);
        }
      }
    }
  }

  // REVERSE the ordering for RCM
  reverse(new_to_old.begin(), new_to_old.end());
}

// BFS Order (Cluster Friends)
// Standard BFS traversal order
void computeBFSOrder(const CSRGraph *g, vector<node_t> &new_to_old) {
  // Use max degree node as root for better reachability?
  // Or node 0? Let's use Node 0 or max degree.
  // For clustering, starting from a HUB is actually good for 'BFS-Order',
  // but RCM prefers peripheral. The prompt says BFS-Order failed because
  // we started from max degree (wavefront explosion).
  // But BFS-Order *is* designed to explode wavefront (level by level).
  // Let's stick to standard max-degree start for 'BFS' mode comparison.

  node_t max_node = 0;
  edge_t max_deg = 0;
  for (node_t i = 0; i < g->num_nodes; i++) {
    edge_t deg = g->h_row_ptr[i + 1] - g->h_row_ptr[i];
    if (deg > max_deg) {
      max_deg = deg;
      max_node = i;
    }
  }

  printf("  [BFS-Order] Start Node: %d (Degree: %ld)\n", max_node, max_deg);

  vector<bool> visited(g->num_nodes, false);
  queue<node_t> q;

  q.push(max_node);
  visited[max_node] = true;
  new_to_old.push_back(max_node);

  while (!q.empty()) {
    node_t u = q.front();
    q.pop();

    edge_t start = g->h_row_ptr[u];
    edge_t end = g->h_row_ptr[u + 1];

    for (edge_t e = start; e < end; e++) {
      node_t v = g->h_col_idx[e];
      if (!visited[v]) {
        visited[v] = true;
        new_to_old.push_back(v);
        q.push(v);
      }
    }
  }

  // Disconnected components
  for (node_t i = 0; i < g->num_nodes; i++) {
    if (!visited[i]) {
      q.push(i);
      visited[i] = true;
      new_to_old.push_back(i);
      while (!q.empty()) {
        node_t u = q.front();
        q.pop();
        for (edge_t e = g->h_row_ptr[u]; e < g->h_row_ptr[u + 1]; e++) {
          node_t v = g->h_col_idx[e];
          if (!visited[v]) {
            visited[v] = true;
            new_to_old.push_back(v);
            q.push(v);
          }
        }
      }
    }
  }
}

// Degree Sort (Hubs First)
void computeDegreeOrder(const CSRGraph *g, vector<node_t> &new_to_old) {
  for (node_t i = 0; i < g->num_nodes; i++) {
    new_to_old.push_back(i);
  }
  sort(new_to_old.begin(), new_to_old.end(), [&](node_t a, node_t b) {
    // Descending Degree
    return getDegree(g, a) > getDegree(g, b);
  });
}

// =============================================================================
// Main Reorder Function
// =============================================================================

CSRGraph *reorderGraph(const CSRGraph *input, ReorderMethod method) {
  printf("Reordering Graph...\n");

  node_t num_nodes = input->num_nodes;
  edge_t num_edges = input->num_edges;

  vector<node_t> new_to_old; // Mapping: new_id -> old_id
  new_to_old.reserve(num_nodes);

  // 1. Compute Permutation
  if (method == REORDER_BFS) {
    printf("  Method: BFS (Level Order)\n");
    computeBFSOrder(input, new_to_old);
  } else if (method == REORDER_DEGREE) {
    printf("  Method: Degree Sort (Hubs First)\n");
    computeDegreeOrder(input, new_to_old);
  } else if (method == REORDER_RCM) {
    printf("  Method: Reverse Cuthill-McKee (Bandwidth Minimization)\n");
    computeRCM(input, new_to_old);
  } else {
    printf("  Method: Unknown, skipping.\n");
    return NULL;
  }

  // Create Inverse Mapping: old_id -> new_id
  vector<node_t> old_to_new(num_nodes);
  for (node_t new_id = 0; new_id < num_nodes; new_id++) {
    old_to_new[new_to_old[new_id]] = new_id;
  }

  // 2. Build New Graph
  CSRGraph *output = new CSRGraph;
  output->num_nodes = num_nodes;
  output->num_edges = num_edges;

  CUDA_CHECK(
      cudaMallocHost(&output->h_row_ptr, (num_nodes + 1) * sizeof(edge_t)));
  CUDA_CHECK(cudaMallocHost(&output->h_col_idx, num_edges * sizeof(node_t)));

  output->h_row_ptr[0] = 0;
  edge_t current_edge = 0;

  for (node_t new_id = 0; new_id < num_nodes; new_id++) {
    node_t old_id = new_to_old[new_id];

    // Get old neighbors
    edge_t start = input->h_row_ptr[old_id];
    edge_t end = input->h_row_ptr[old_id + 1];

    // Transform neighbors to new IDs
    vector<node_t> new_neighbors;
    new_neighbors.reserve(end - start);

    for (edge_t e = start; e < end; e++) {
      node_t old_neighbor = input->h_col_idx[e];
      new_neighbors.push_back(old_to_new[old_neighbor]);
    }

    // Sort neighbors (critical for CSR efficiency and compression)
    sort(new_neighbors.begin(), new_neighbors.end());

    // Write to new graph
    for (node_t neighbor : new_neighbors) {
      output->h_col_idx[current_edge++] = neighbor;
    }
    output->h_row_ptr[new_id + 1] = current_edge;
  }

  printf("  Reordering Complete.\n");
  return output;
}

// =============================================================================
// Streaming Reorder and Save
// =============================================================================

void reorderAndSaveStreaming(const CSRGraph *graph, const char *out_filename,
                             ReorderMethod method) {
  printf("Reordering Graph (Streaming Mode)...\n");

  node_t num_nodes = graph->num_nodes;
  edge_t num_edges = graph->num_edges;

  vector<node_t> new_to_old; // Mapping: new_id -> old_id
  new_to_old.reserve(num_nodes);

  // 1. Compute Permutation
  if (method == REORDER_BFS) {
    printf("  Method: BFS (Level Order)\n");
    computeBFSOrder(graph, new_to_old);
  } else if (method == REORDER_DEGREE) {
    printf("  Method: Degree Sort (Hubs First)\n");
    computeDegreeOrder(graph, new_to_old);
  } else if (method == REORDER_RCM) {
    printf("  Method: Reverse Cuthill-McKee (Bandwidth Minimization)\n");
    computeRCM(graph, new_to_old);
  } else {
    printf("  Method: Unknown, skipping.\n");
    return;
  }

  // Create Inverse Mapping: old_id -> new_id
  // This is needed to translate neighbor IDs
  vector<node_t> old_to_new(num_nodes);
  for (node_t new_id = 0; new_id < num_nodes; new_id++) {
    old_to_new[new_to_old[new_id]] = new_id;
  }

  // 2. Stream to Disk
  // Format: [num_nodes (8B)] [num_edges (8B)] [row_ptr (8B * (N+1))] [col_idx
  // (4B * M)] Wait, our bin format uses sizeof(node_t) for col_idx which is int
  // (4 bytes). Check loadGraphCSRBin: header is 8 bytes, row_ptr is edge_t (8
  // bytes), col_idx is node_t (4 bytes).

  if (!out_filename) {
    printf("No output file specified. Skipping save.\n");
    return;
  }

  FILE *f = fopen(out_filename, "wb");
  if (!f) {
    perror("Error opening output file");
    return;
  }

  printf("  Streaming to %s...\n", out_filename);

  unsigned long long n = (unsigned long long)num_nodes;
  unsigned long long m = (unsigned long long)num_edges;

  // Header
  fwrite(&n, sizeof(unsigned long long), 1, f);
  fwrite(&m, sizeof(unsigned long long), 1, f);

  // Row Pointers
  // We need to compute them incrementally.
  // row_ptr[i] = row_ptr[i-1] + degree(new_i-1)
  // But we can't write row_ptr first easily because they are contiguous at the
  // start of the file. We have a choice: A) Buffer row_ptr in memory (Size: N *
  // 8 bytes = 0.5 GB for Friendster). Safe. B) Seek (slow on some systems, but
  // fine for SSD).

  // Strategy A is best: 500MB is cheap vs 15GB edges.

  vector<edge_t> new_row_ptr;
  new_row_ptr.reserve(num_nodes + 1);
  new_row_ptr.push_back(0);

  edge_t current_edge_count = 0;

  // Pass 1: Compute Row Pointers (Fast)
  for (node_t new_id = 0; new_id < num_nodes; new_id++) {
    node_t old_id = new_to_old[new_id];
    edge_t deg = graph->h_row_ptr[old_id + 1] - graph->h_row_ptr[old_id];
    current_edge_count += deg;
    new_row_ptr.push_back(current_edge_count);
  }

  // Write Row Pointers
  fwrite(new_row_ptr.data(), sizeof(edge_t), new_row_ptr.size(), f);

  // Pass 2: Write Enumerate Edges (Streaming)
  // We can afford a small buffer for neighbors (e.g. max degree).

  vector<node_t> local_neighbors;
  local_neighbors.reserve(100000); // Reasonable initial size

  for (node_t new_id = 0; new_id < num_nodes; new_id++) {
    node_t old_id = new_to_old[new_id];

    edge_t start = graph->h_row_ptr[old_id];
    edge_t end = graph->h_row_ptr[old_id + 1];

    local_neighbors.clear();

    for (edge_t e = start; e < end; e++) {
      node_t old_neighbor = graph->h_col_idx[e];
      local_neighbors.push_back(old_to_new[old_neighbor]);
    }

    // Sort neighbors
    sort(local_neighbors.begin(), local_neighbors.end());

    // Write to disk
    if (!local_neighbors.empty()) {
      fwrite(local_neighbors.data(), sizeof(node_t), local_neighbors.size(), f);
    }

    if (new_id % 1000000 == 0) {
      printf("\r  Progress: %.1f%%", 100.0 * new_id / num_nodes);
      fflush(stdout);
    }
  }

  printf("\n  Save Complete.\n");
  fclose(f);
}
