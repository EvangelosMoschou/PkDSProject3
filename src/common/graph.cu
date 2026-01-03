#include "graph.h"
#include "io_utils.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

// =============================================================================
// Graph Loading (Text Format)
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

// Updated loadGraphCSRBin using pread_full
CSRGraph *loadGraphCSRBin(const char *filename) {
  int fd = open(filename, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "Error: Cannot open file %s\n", filename);
    return nullptr;
  }

  CSRGraph *graph = new CSRGraph;
  off_t offset = 0;

  // Read header (both are 8 bytes in our binary format)
  unsigned long long n, m;

  if (pread_full(fd, &n, sizeof(unsigned long long), offset) != 0) {
    fprintf(stderr, "Error reading num_nodes from %s\n", filename);
    close(fd);
    delete graph;
    return nullptr;
  }
  offset += sizeof(unsigned long long);

  if (pread_full(fd, &m, sizeof(unsigned long long), offset) != 0) {
    fprintf(stderr, "Error reading num_edges from %s\n", filename);
    close(fd);
    delete graph;
    return nullptr;
  }
  offset += sizeof(unsigned long long);

  graph->num_nodes = (node_t)n;
  graph->num_edges = (edge_t)m;

  printf("Loading Graph Binary: Nodes=%d, Edges=%lld\n", graph->num_nodes,
         (long long)graph->num_edges);

  // Allocate arrays
  // Note: For massive graphs, we might want to map this instead of new[],
  // but let's stick to standard allocation for now and trust copyGraphToDevice
  // to handle the GPU side. Friendster edges = 1.8B * 4 bytes = 7.2 GB. System
  // RAM is 32GB, so this fits easily.

  graph->h_row_ptr = new edge_t[graph->num_nodes + 1];
  graph->h_col_idx = new node_t[graph->num_edges];

  printf("Allocated Host Memory. Reading Data...\n");

  if (pread_full(fd, graph->h_row_ptr, (graph->num_nodes + 1) * sizeof(edge_t),
                 offset) != 0) {
    fprintf(stderr, "Error reading row_ptr from %s\n", filename);
    close(fd);
    delete[] graph->h_row_ptr;
    delete[] graph->h_col_idx;
    delete graph;
    return nullptr;
  }
  offset += (graph->num_nodes + 1) * sizeof(edge_t);

  if (pread_full(fd, graph->h_col_idx, graph->num_edges * sizeof(node_t),
                 offset) != 0) {
    fprintf(stderr, "Error reading col_idx from %s\n", filename);
    close(fd);
    delete[] graph->h_row_ptr;
    delete[] graph->h_col_idx;
    delete graph;
    return nullptr;
  }

  close(fd);

  graph->d_row_ptr = nullptr;
  graph->d_col_idx = nullptr;

  printf("Graph Loaded Successfully.\n");
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

// =============================================================================
// HDF5 Loader
// =============================================================================

#include <hdf5.h>

CSRGraph *loadGraphHDF5(const char *filename) {
  printf("Loading HDF5 Graph: %s\n", filename);

  // Turn off HDF5 auto-printing errors
  H5Eset_auto(H5E_DEFAULT, NULL, NULL);

  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    fprintf(stderr, "Error: Cannot open HDF5 file %s\n", filename);
    return nullptr;
  }

  // Navigate to /Problem/A
  hid_t group_id = H5Gopen(file_id, "Problem", H5P_DEFAULT);
  if (group_id < 0) {
    H5Fclose(file_id);
    fprintf(stderr, "Error: 'Problem' group not found in %s\n", filename);
    return nullptr;
  }
  hid_t a_id = H5Gopen(group_id, "A", H5P_DEFAULT);
  if (a_id < 0) {
    H5Gclose(group_id);
    H5Fclose(file_id);
    fprintf(stderr, "Error: 'A' group/dataset not found in %s\n", filename);
    return nullptr;
  }

  hid_t ir_dset = H5Dopen(a_id, "ir", H5P_DEFAULT);
  hid_t jc_dset = H5Dopen(a_id, "jc", H5P_DEFAULT);

  if (ir_dset < 0 || jc_dset < 0) {
    fprintf(stderr, "Error: 'ir' or 'jc' datasets not found\n");
    if (ir_dset >= 0)
      H5Dclose(ir_dset);
    if (jc_dset >= 0)
      H5Dclose(jc_dset);
    H5Gclose(a_id);
    H5Gclose(group_id);
    H5Fclose(file_id);
    return nullptr;
  }

  // Get Dimensions
  hid_t ir_space = H5Dget_space(ir_dset);
  hid_t jc_space = H5Dget_space(jc_dset);

  hsize_t nnz = H5Sget_simple_extent_npoints(ir_space);
  hsize_t ncol_ptr = H5Sget_simple_extent_npoints(jc_space);

  node_t num_nodes = (node_t)(ncol_ptr - 1);
  edge_t num_edges = (edge_t)nnz;

  printf("HDF5: Nodes=%d, Edges=%lld\n", num_nodes, (long long)num_edges);

  CSRGraph *graph = new CSRGraph;
  graph->num_nodes = num_nodes;
  graph->num_edges = num_edges;

  // Allocate Memory
  printf(
      "HDF5: Allocating Host Memory (%.2f GB)...\n",
      (double)((num_nodes + 1) * sizeof(edge_t) + num_edges * sizeof(node_t)) /
          1024.0 / 1024.0 / 1024.0);

  graph->h_row_ptr = new edge_t[num_nodes + 1];
  graph->h_col_idx = new node_t[num_edges];

  // READ DATA
  // Assumption: Symmetric Graph (CSC == CSR)
  // jc -> row_ptr
  // ir -> col_idx

  printf("HDF5: Reading jc (row_ptr)...\n");
  if (H5Dread(jc_dset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              graph->h_row_ptr) < 0) {
    fprintf(stderr, "Error reading 'jc'\n");
    delete[] graph->h_row_ptr;
    delete[] graph->h_col_idx;
    delete graph;
    H5Dclose(ir_dset);
    H5Dclose(jc_dset);
    H5Gclose(a_id);
    H5Gclose(group_id);
    H5Fclose(file_id);
    return nullptr;
  }

  printf("HDF5: Reading ir (col_idx)...\n");
  if (H5Dread(ir_dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              graph->h_col_idx) < 0) {
    fprintf(stderr, "Error reading 'ir'\n");
    delete[] graph->h_row_ptr;
    delete[] graph->h_col_idx;
    delete graph;
    H5Dclose(ir_dset);
    H5Dclose(jc_dset);
    H5Gclose(a_id);
    H5Gclose(group_id);
    H5Fclose(file_id);
    return nullptr;
  }

  printf("HDF5: Load Complete.\n");

  H5Sclose(ir_space);
  H5Sclose(jc_space);
  H5Dclose(ir_dset);
  H5Dclose(jc_dset);
  H5Gclose(a_id);
  H5Gclose(group_id);
  H5Fclose(file_id);

  graph->d_row_ptr = nullptr;
  graph->d_col_idx = nullptr;
  return graph;
}

// =============================================================================
// Memory Management
// =============================================================================

#include "io_utils.h"
#include <fcntl.h>
#include <unistd.h>

// =============================================================================
// Memory Management
// =============================================================================

void copyGraphToDevice(CSRGraph *graph) {
  size_t row_ptr_size = (graph->num_nodes + 1) * sizeof(edge_t);
  size_t col_idx_size = graph->num_edges * sizeof(node_t);

  printf("Copying Graph to Device...\n");

  // Allocate device memory (managed to support large graphs)
  CUDA_CHECK(cudaMallocManaged(&graph->d_row_ptr, row_ptr_size));
  CUDA_CHECK(cudaMemcpy(graph->d_row_ptr, graph->h_row_ptr, row_ptr_size,
                        cudaMemcpyHostToDevice));

  // Huge Graph Optimization:
  // If > 4GB, use Zero-Copy to avoid VRAM Thrashing on 6GB cards.
  // Friendster is ~7.2GB, so this will trigger.
  size_t huge_threshold = 4ULL * 1024 * 1024 * 1024; // 4GB

  if (col_idx_size > huge_threshold) {
    printf("NOTICE: Graph Edges > 4GB (Size: %.2f GB). Using Zero-Copy Mapped "
           "Memory for col_idx.\n",
           col_idx_size / (1024.0 * 1024.0 * 1024.0));

    // 1. Register Host Memory as Mapped (Device Accessible)
    CUDA_CHECK(cudaHostRegister(graph->h_col_idx, col_idx_size,
                                cudaHostRegisterMapped));
    // 2. Get Device Pointer
    CUDA_CHECK(cudaHostGetDevicePointer((void **)&graph->d_col_idx,
                                        (void *)graph->h_col_idx, 0));
  } else {
    // Standard Allocation in VRAM (Managed)
    CUDA_CHECK(cudaMallocManaged(&graph->d_col_idx, col_idx_size));
    CUDA_CHECK(cudaMemcpy(graph->d_col_idx, graph->h_col_idx, col_idx_size,
                          cudaMemcpyHostToDevice));

    // Advanced Memory Hints for VRAM Residents
    int deviceId;
    CUDA_CHECK(cudaGetDevice(&deviceId));

    // Hint: Prefetch to GPU to avoid page faults during execution
    CUDA_CHECK(
        cudaMemPrefetchAsync(graph->d_row_ptr, row_ptr_size, deviceId, 0));
    CUDA_CHECK(
        cudaMemPrefetchAsync(graph->d_col_idx, col_idx_size, deviceId, 0));

    // Hint: Data is read-mostly (enables caching)
    CUDA_CHECK(cudaMemAdvise(graph->d_row_ptr, row_ptr_size,
                             cudaMemAdviseSetReadMostly, deviceId));
    CUDA_CHECK(cudaMemAdvise(graph->d_col_idx, col_idx_size,
                             cudaMemAdviseSetReadMostly, deviceId));
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
  printf("Graph Stats:\n");
  printf("Nodes: %d\n", graph->num_nodes);
  printf("Edges: %lld\n", (long long)graph->num_edges);

  // Calculate degrees to check stats
  edge_t min_deg = graph->num_edges;
  edge_t max_deg = 0;

  // Checking degree of first few and last few if possible, or just skip full
  // check for speed? Let's iterate all if not huge. Actually, printGraphStats
  // iterates all.
  for (node_t i = 0; i < graph->num_nodes; i++) {
    edge_t deg = graph->h_row_ptr[i + 1] - graph->h_row_ptr[i];
    if (deg < min_deg)
      min_deg = deg;
    if (deg > max_deg)
      max_deg = deg;
  }

  printf("Min Degree: %lld\n", (long long)min_deg);
  printf("Max Degree: %lld\n", (long long)max_deg);
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
      fprintf(stderr, "Error: Invalid neighbor %d at edge %lld\n",
              graph->h_col_idx[i], (long long)i);
      return false;
    }
  }

  return true;
}
