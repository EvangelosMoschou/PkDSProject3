#include "compression.h"
#include <cstdio>
#include <cstdlib>

static int compare_node_t_asc(const void *a, const void *b) {
  node_t va = *(const node_t *)a;
  node_t vb = *(const node_t *)b;
  if (va < vb)
    return -1;
  if (va > vb)
    return 1;
  return 0;
}

static int ensureNodeBuffer(node_t **buffer, size_t *capacity, size_t needed) {
  if (needed <= *capacity)
    return 1;
  node_t *tmp = (node_t *)realloc(*buffer, needed * sizeof(node_t));
  if (!tmp)
    return 0;
  *buffer = tmp;
  *capacity = needed;
  return 1;
}

// =============================================================================
// Variable-Length Quantity (Varint) Encoding (Host)
// =============================================================================

// Encodes a 32-bit unsigned integer into 1-5 bytes
// Returns number of bytes written
int encode_varint(uint8_t *buffer, node_t value) {
  int len = 0;
  while (value > 127) {
    buffer[len++] = (uint8_t)((value & 127) | 128);
    value >>= 7;
  }
  buffer[len++] = (uint8_t)(value & 127);
  return len;
}

// =============================================================================
// Graph Compression Utility
// =============================================================================

void compressGraph(const CSRGraph *input, CompressedCSRGraph *output) {
  printf("Compressing Graph (c-CSR)... ");

  node_t num_nodes = input->num_nodes;
  output->num_nodes = num_nodes;
  output->num_edges = input->num_edges;

  // Allocate row pointer array
  // Allocate row pointer array (Pinned Memory)
  CUDA_CHECK(
      cudaMallocHost(&output->h_row_Ptr, (num_nodes + 1) * sizeof(edge_t)));

  // PASS 1: Calculate Size
  size_t total_bytes = 0;
  node_t *neighbors = NULL;
  size_t neighbors_cap = 0;

  for (node_t i = 0; i < num_nodes; i++) {
    edge_t start = input->h_row_ptr[i];
    edge_t end = input->h_row_ptr[i + 1];
    edge_t degree = end - start;

    if (!ensureNodeBuffer(&neighbors, &neighbors_cap, (size_t)degree)) {
      free(neighbors);
      fprintf(stderr, "compressGraph: allocation failed in pass 1\n");
      output->compressed_size_bytes = 0;
      return;
    }

    for (edge_t e = 0; e < degree; e++) {
      neighbors[e] = input->h_col_idx[start + e];
    }

    if (degree > 1) {
      qsort(neighbors, (size_t)degree, sizeof(node_t), compare_node_t_asc);
    }

    // Calculate encoded size
    node_t prev = 0;
    size_t row_bytes = 0;

    for (edge_t k = 0; k < degree; k++) {
      node_t neighbor = neighbors[k];
      node_t delta = neighbor - prev;
      int len = 0;
      node_t val = delta;
      while (val > 127) {
        len++;
        val >>= 7;
      }
      len++;
      row_bytes += len;
      prev = neighbor;
    }
    total_bytes += row_bytes;
  }

  output->compressed_size_bytes = total_bytes;
  printf("Original: %.2f GB -> Compressed: %.2f GB (Ratio: %.2fx)\n",
         (double)input->num_edges * 4.0 / 1e9, (double)total_bytes / 1e9,
         (double)(input->num_edges * 4) / total_bytes);

  // PASS 2: Encode
  size_t alloc_bytes = total_bytes > 0 ? total_bytes : 1;
  CUDA_CHECK(cudaMallocHost(&output->h_compressed_col,
                            alloc_bytes * sizeof(uint8_t)));

  // Parallelize? Simple loop for now.
  size_t current_byte_offset = 0;

  for (node_t i = 0; i < num_nodes; i++) {
    edge_t start = input->h_row_ptr[i];
    edge_t end = input->h_row_ptr[i + 1];
    edge_t degree = end - start;

    output->h_row_Ptr[i] = (edge_t)current_byte_offset;

    if (!ensureNodeBuffer(&neighbors, &neighbors_cap, (size_t)degree)) {
      free(neighbors);
      fprintf(stderr, "compressGraph: allocation failed in pass 2\n");
      return;
    }

    for (edge_t e = 0; e < degree; e++) {
      neighbors[e] = input->h_col_idx[start + e];
    }
    if (degree > 1) {
      qsort(neighbors, (size_t)degree, sizeof(node_t), compare_node_t_asc);
    }

    node_t prev = 0;
    for (edge_t k = 0; k < degree; k++) {
      node_t neighbor = neighbors[k];
      node_t delta = neighbor - prev;
      current_byte_offset +=
          encode_varint(&output->h_compressed_col[current_byte_offset], delta);
      prev = neighbor;
    }
  }
  output->h_row_Ptr[num_nodes] = (edge_t)total_bytes;

  free(neighbors);

  printf("Compression Complete.\n");
}

// =============================================================================
// In-Place Compression
// =============================================================================

bool compressGraphInPlace(CSRGraph *input, CompressedCSRGraph *output) {
  printf("Attempting In-Place Compression (Reuse Input Buffer)...\n");

  node_t num_nodes = input->num_nodes;
  output->num_nodes = num_nodes;
  output->num_edges = input->num_edges;

  // Allocate row pointer array (Pinned)
  CUDA_CHECK(
      cudaMallocHost(&output->h_row_Ptr, (num_nodes + 1) * sizeof(edge_t)));

  // Re-use input column index buffer
  // We treat the original host buffer as the destination byte array
  // CAST: node_t* -> uint8_t*
  output->h_compressed_col = (uint8_t *)input->h_col_idx;

  // Safety & Size Calculation (Pass 1)
  size_t write_offset = 0; // In Bytes
  bool safe = true;

  node_t *neighbors = NULL;
  size_t neighbors_cap = 0;

  // We need to calculate row offsets first to fill h_row_Ptr

  for (node_t i = 0; i < num_nodes; i++) {
    output->h_row_Ptr[i] = (edge_t)write_offset;

    edge_t start_idx = input->h_row_ptr[i];
    edge_t end_idx = input->h_row_ptr[i + 1];
    edge_t degree = end_idx - start_idx;

    // Check Safety:
    // Reads will start at 'start_idx' (byte offset: start_idx * 4)
    // Writes will extend to 'write_offset' + compressed_size
    // Since we buffer the WHOLE row in 'neighbors', the critical check is:
    // Can we write the compressed row at 'write_offset' without overwriting
    // the *next* row's start?
    // Actually, we don't care about the *current* row's input data once loaded.
    // We only care about not overwriting FUTURE rows (i+1, i+2...).
    // So check: write_offset (end of this row) <= (end_idx * 4)
    // Wait, if write_offset > end_idx*4, we have overflowed into the next
    // unread row.

    if (!ensureNodeBuffer(&neighbors, &neighbors_cap, (size_t)degree)) {
      free(neighbors);
      CUDA_CHECK(cudaFreeHost(output->h_row_Ptr));
      output->h_row_Ptr = NULL;
      output->h_compressed_col = NULL;
      return false;
    }

    for (edge_t e = 0; e < degree; e++) {
      neighbors[e] = input->h_col_idx[start_idx + e];
    }
    if (degree > 1) {
      qsort(neighbors, (size_t)degree, sizeof(node_t), compare_node_t_asc);
    }

    // Calculate Size
    size_t row_bytes = 0;
    node_t prev = 0;
    for (edge_t k = 0; k < degree; k++) {
      node_t neighbor = neighbors[k];
      node_t delta = neighbor - prev;
      node_t val = delta;
      while (val > 127) {
        row_bytes++;
        val >>= 7;
      }
      row_bytes++;
      prev = neighbor;
    }

    // CHECK SAFETY
    if ((write_offset + row_bytes) > (size_t)end_idx * 4) {
      printf("SAFETY VIOLATION at Row %u! WriteOffset (%lu) > ReadHead (%lu). "
             "Aborting In-Place.\n",
             i, write_offset + row_bytes, (size_t)end_idx * 4);
      safe = false;
      break;
    }

    // If safe, we can Encode NOW (Single Pass approach since we buffered
    // neighbors) Optimization: Single Pass In-Place

    node_t prev_enc = 0;
    for (edge_t k = 0; k < degree; k++) {
      node_t neighbor = neighbors[k];
      node_t delta = neighbor - prev_enc;
      write_offset +=
          encode_varint(&output->h_compressed_col[write_offset], delta);
      prev_enc = neighbor;
    }
  }

  output->h_row_Ptr[num_nodes] = (edge_t)write_offset;
  output->compressed_size_bytes = write_offset;

  if (!safe) {
    // Cleanup allocated row ptr
    CUDA_CHECK(cudaFreeHost(output->h_row_Ptr));
    output->h_row_Ptr = NULL;
    output->h_compressed_col = NULL;
    free(neighbors);
    return false;
  }

  printf("In-Place Compression Successful. Size: %.2f GB\n",
         (double)write_offset / 1e9);

  // Register the REUSED buffer for Zero-Copy
  // ALREADY PINNED via cudaMallocHost (in loadGraph)
  // CUDA_CHECK(cudaHostRegister(output->h_compressed_col,
  // output->compressed_size_bytes, cudaHostRegisterMapped));
  free(neighbors);
  return true;
}
