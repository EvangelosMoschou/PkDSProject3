#include "graph.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

// =============================================================================
// Graph Loading
// =============================================================================

CSRGraph *loadGraph(const char *filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    fprintf(stderr, "Error: Cannot open file %s\n", filename);
    return nullptr;
  }

  node_t num_nodes;
  edge_t num_edges;
  file >> num_nodes >> num_edges;

  // Read edges into adjacency list
  std::vector<std::vector<node_t>> adj(num_nodes);

  for (edge_t i = 0; i < num_edges; i++) {
    node_t src, dst;
    file >> src >> dst;

    if (src >= 0 && src < num_nodes && dst >= 0 && dst < num_nodes) {
      adj[src].push_back(dst);
      // For undirected graphs, uncomment:
      // adj[dst].push_back(src);
    }
  }
  file.close();

  // Allocate CSR graph
  CSRGraph *graph = new CSRGraph;
  graph->num_nodes = num_nodes;
  graph->num_edges = 0;

  // Count actual edges and allocate
  for (node_t i = 0; i < num_nodes; i++) {
    graph->num_edges += adj[i].size();
  }

  graph->h_row_ptr = new edge_t[num_nodes + 1];
  graph->h_col_idx = new node_t[graph->num_edges];

  // Build CSR structure
  edge_t edge_idx = 0;
  for (node_t i = 0; i < num_nodes; i++) {
    graph->h_row_ptr[i] = edge_idx;

    // Sort neighbors for better cache locality
    std::sort(adj[i].begin(), adj[i].end());

    for (node_t neighbor : adj[i]) {
      graph->h_col_idx[edge_idx++] = neighbor;
    }
  }
  graph->h_row_ptr[num_nodes] = edge_idx;

  // Initialize device pointers to null
  graph->d_row_ptr = nullptr;
  graph->d_col_idx = nullptr;

  return graph;
}

CSRGraph *loadGraphCSRBin(const char *filename) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "Error: Cannot open file %s\n", filename);
    return nullptr;
  }

  CSRGraph *graph = new CSRGraph;

  // Read header (both are 8 bytes in our binary format)
  unsigned long long n, m;
  if (fread(&n, sizeof(unsigned long long), 1, file) != 1 ||
      fread(&m, sizeof(unsigned long long), 1, file) != 1) {
    fprintf(stderr, "Error reading header from %s\n", filename);
    fclose(file);
    return nullptr;
  }
  graph->num_nodes = (node_t)n;
  graph->num_edges = (edge_t)m;

  // Allocate and read arrays
  graph->h_row_ptr = new edge_t[graph->num_nodes + 1];
  graph->h_col_idx = new node_t[graph->num_edges];

  fread(graph->h_row_ptr, sizeof(edge_t), graph->num_nodes + 1, file);
  fread(graph->h_col_idx, sizeof(node_t), graph->num_edges, file);

  fclose(file);

  graph->d_row_ptr = nullptr;
  graph->d_col_idx = nullptr;

  return graph;
}

void saveGraphCSRBin(const CSRGraph *graph, const char *filename) {
  FILE *file = fopen(filename, "wb");
  if (!file) {
    fprintf(stderr, "Error: Cannot create file %s\n", filename);
    return;
  }

  // Write header
  fwrite(&graph->num_nodes, sizeof(node_t), 1, file);
  fwrite(&graph->num_edges, sizeof(edge_t), 1, file);

  // Write arrays
  fwrite(graph->h_row_ptr, sizeof(edge_t), graph->num_nodes + 1, file);
  fwrite(graph->h_col_idx, sizeof(node_t), graph->num_edges, file);

  fclose(file);
}

CSRGraph *generateRandomGraph(node_t num_nodes, edge_t avg_degree) {
  CSRGraph *graph = new CSRGraph;
  graph->num_nodes = num_nodes;

  std::vector<std::vector<node_t>> adj(num_nodes);

  // Generate random edges
  srand(42); // Fixed seed for reproducibility
  for (node_t i = 0; i < num_nodes; i++) {
    int degree = rand() % (2 * avg_degree) + 1;
    for (int j = 0; j < degree; j++) {
      node_t neighbor = rand() % num_nodes;
      if (neighbor != i) {
        adj[i].push_back(neighbor);
      }
    }
  }

  // Count edges
  graph->num_edges = 0;
  for (node_t i = 0; i < num_nodes; i++) {
    graph->num_edges += adj[i].size();
  }

  // Allocate CSR arrays
  graph->h_row_ptr = new edge_t[num_nodes + 1];
  graph->h_col_idx = new node_t[graph->num_edges];

  // Build CSR
  edge_t edge_idx = 0;
  for (node_t i = 0; i < num_nodes; i++) {
    graph->h_row_ptr[i] = edge_idx;
    std::sort(adj[i].begin(), adj[i].end());
    for (node_t neighbor : adj[i]) {
      graph->h_col_idx[edge_idx++] = neighbor;
    }
  }
  graph->h_row_ptr[num_nodes] = edge_idx;

  graph->d_row_ptr = nullptr;
  graph->d_col_idx = nullptr;

  return graph;
}

// =============================================================================
// Memory Management
// =============================================================================

void copyGraphToDevice(CSRGraph *graph) {
  size_t row_ptr_size = (graph->num_nodes + 1) * sizeof(edge_t);
  size_t col_idx_size = graph->num_edges * sizeof(node_t);

  // Allocate device memory (managed to support large graphs)
  // Allocate device memory (managed to support large graphs)
  CUDA_CHECK(cudaMallocManaged(&graph->d_row_ptr, row_ptr_size));
  CUDA_CHECK(cudaMemcpy(graph->d_row_ptr, graph->h_row_ptr, row_ptr_size,
                        cudaMemcpyHostToDevice));

  // Huge Graph Optimization:
  // If > 8GB, assume Friendster. Skip d_col_idx allocation.
  // Friendster col_idx = 14GB. Host = 14GB. Unified = 14GB. Total 28GB. Too
  // much. We will stream from Host Pinned Memory.
  size_t huge_threshold = 8ULL * 1024 * 1024 * 1024;
  if (col_idx_size > huge_threshold) {
    printf("NOTICE: Graph Edges > 8GB. Using Zero-Copy Mapped Memory for "
           "col_idx.\n");
    // 1. Register Host Memory as Mapped (Device Accessible)
    CUDA_CHECK(cudaHostRegister(graph->h_col_idx, col_idx_size,
                                cudaHostRegisterMapped));
    // 2. Get Device Pointer
    CUDA_CHECK(cudaHostGetDevicePointer((void **)&graph->d_col_idx,
                                        (void *)graph->h_col_idx, 0));
  } else {
    // Standard Allocation
    CUDA_CHECK(cudaMallocManaged(&graph->d_col_idx, col_idx_size));
    CUDA_CHECK(cudaMemcpy(graph->d_col_idx, graph->h_col_idx, col_idx_size,
                          cudaMemcpyHostToDevice));
    // For small graphs, we can also register for streaming optimization if
    // needed, but Managed Memory handles Mawi fine.
    CUDA_CHECK(cudaHostRegister(graph->h_col_idx, col_idx_size,
                                cudaHostRegisterDefault));
  }
  // Remove duplicate calls from end of function

  // IMPORTANT: Do NOT free host memory. We need it to stream data to the GPU.
  // We accepted the trade-off of using 29GB system RAM (which is available).
  // delete[] graph->h_row_ptr;
  // graph->h_row_ptr = nullptr;
  // delete[] graph->h_col_idx;
  // graph->h_col_idx = nullptr;
}

void freeGraphDevice(CSRGraph *graph) {
  if (graph->d_row_ptr) {
    CUDA_CHECK(cudaFree(graph->d_row_ptr));
    graph->d_row_ptr = nullptr;
  }
  if (graph->d_col_idx) {
    CUDA_CHECK(cudaFree(graph->d_col_idx));
    graph->d_col_idx = nullptr;
  }
}

void freeGraph(CSRGraph *graph) {
  if (!graph)
    return;

  freeGraphDevice(graph);

  delete[] graph->h_row_ptr;
  delete[] graph->h_col_idx;
  delete graph;
}

// =============================================================================
// Utilities
// =============================================================================

void printGraphStats(const CSRGraph *graph) {
  printf("=== Graph Statistics ===\n");
  printf("Nodes: %d\n", graph->num_nodes);
  printf("Edges: %d\n", graph->num_edges);
  printf("Average Degree: %.2f\n", (double)graph->num_edges / graph->num_nodes);

  // Find min/max degree
  edge_t min_deg = graph->num_edges, max_deg = 0;
  for (node_t i = 0; i < graph->num_nodes; i++) {
    edge_t deg = graph->h_row_ptr[i + 1] - graph->h_row_ptr[i];
    min_deg = std::min(min_deg, deg);
    max_deg = std::max(max_deg, deg);
  }
  printf("Min Degree: %d\n", min_deg);
  printf("Max Degree: %d\n", max_deg);
  printf("========================\n\n");
}

bool validateGraph(const CSRGraph *graph) {
  // Check row_ptr is monotonically increasing
  for (node_t i = 0; i < graph->num_nodes; i++) {
    if (graph->h_row_ptr[i] > graph->h_row_ptr[i + 1]) {
      fprintf(stderr,
              "Error: row_ptr not monotonically increasing at node %d\n", i);
      return false;
    }
  }

  // Check col_idx values are valid
  for (edge_t i = 0; i < graph->num_edges; i++) {
    if (graph->h_col_idx[i] < 0 || graph->h_col_idx[i] >= graph->num_nodes) {
      fprintf(stderr, "Error: Invalid neighbor %d at edge %d\n",
              graph->h_col_idx[i], i);
      return false;
    }
  }

  return true;
}
