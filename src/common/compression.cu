#include "compression.h"
#include <algorithm>
#include <cstdio>
#include <vector>

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

  // Temporary buffer to store compressed data (conservative size)
  // Worst case: 5 bytes per edge + overhead.
  // Optimization: We could do 2 passes (measure size, then allocate).
  // For now, let's use std::vector for safety and copy at end, or just measure
  // first.

  // PASS 1: Calculate Size
  size_t total_bytes = 0;

  /*
   * Note: We need to sort neighbors to perform Delta Encoding efficiently.
   * We assume original CSR might not be sorted.
   * Making a mutable copy of neighbors row by row might be slow but safe.
   */

  std::vector<node_t> neighbors;
  std::vector<size_t> row_offsets(num_nodes + 1);
  row_offsets[0] = 0;

  for (node_t i = 0; i < num_nodes; i++) {
    edge_t start = input->h_row_ptr[i];
    edge_t end = input->h_row_ptr[i + 1];
    edge_t degree = end - start;

    // Copy neighbors
    neighbors.clear();
    for (edge_t e = 0; e < degree; e++) {
      neighbors.push_back(input->h_col_idx[start + e]);
    }

    // Sort
    std::sort(neighbors.begin(), neighbors.end());

    // Calculate encoded size
    node_t prev = 0;
    size_t row_bytes = 0;

    // Helper buffer for size calc
    uint8_t tmp[5];

    for (node_t neighbor : neighbors) {
      node_t delta = neighbor - prev;
      int len = 0;
      // Inline varint logic for speed or call func
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
    row_offsets[i + 1] = total_bytes;
  }

  output->compressed_size_bytes = total_bytes;
  printf("Original: %.2f GB -> Compressed: %.2f GB (Ratio: %.2fx)\n",
         (double)input->num_edges * 4.0 / 1e9, (double)total_bytes / 1e9,
         (double)(input->num_edges * 4) / total_bytes);

  // PASS 2: Encode
  CUDA_CHECK(
      cudaMallocHost(&output->h_compressed_col, total_bytes * sizeof(uint8_t)));

  // Parallelize? Simple loop for now.
  size_t current_byte_offset = 0;

  for (node_t i = 0; i < num_nodes; i++) {
    edge_t start = input->h_row_ptr[i];
    edge_t end = input->h_row_ptr[i + 1];
    edge_t degree = end - start;

    output->h_row_Ptr[i] = (edge_t)current_byte_offset;

    // Copy & Sort (Repeated work, but allows Pass 1 to be just a size sum if we
    // optimized) Optimization: We re-do internal logic to avoid storing vector
    // of vectors
    neighbors.clear();
    for (edge_t e = 0; e < degree; e++) {
      neighbors.push_back(input->h_col_idx[start + e]);
    }
    std::sort(neighbors.begin(), neighbors.end());

    node_t prev = 0;
    for (node_t neighbor : neighbors) {
      node_t delta = neighbor - prev;
      current_byte_offset +=
          encode_varint(&output->h_compressed_col[current_byte_offset], delta);
      prev = neighbor;
    }
  }
  output->h_row_Ptr[num_nodes] = (edge_t)total_bytes;

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

  std::vector<node_t> neighbors;

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

    // Copy & Sort
    neighbors.clear();
    for (edge_t e = 0; e < degree; e++) {
      neighbors.push_back(input->h_col_idx[start_idx + e]);
    }
    std::sort(neighbors.begin(), neighbors.end());

    // Calculate Size
    size_t row_bytes = 0;
    node_t prev = 0;
    for (node_t neighbor : neighbors) {
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
    for (node_t neighbor : neighbors) {
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
    output->h_row_Ptr = nullptr;
    output->h_compressed_col = nullptr;
    return false;
  }

  printf("In-Place Compression Successful. Size: %.2f GB\n",
         (double)write_offset / 1e9);

  // Register the REUSED buffer for Zero-Copy
  // ALREADY PINNED via cudaMallocHost (in loadGraph)
  // CUDA_CHECK(cudaHostRegister(output->h_compressed_col,
  // output->compressed_size_bytes, cudaHostRegisterMapped));
  return true;
}
