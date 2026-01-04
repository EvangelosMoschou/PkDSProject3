#ifndef COMPRESSION_H
#define COMPRESSION_H

#include "cuda_common.h"
#include "graph.h"
#include <cstdint>

// Encodes a varint to buffer, returns bytes written
int encode_varint(uint8_t *buffer, node_t value);

// Compresses a standard CSR graph into c-CSR format
void compressGraph(const CSRGraph *input, CompressedCSRGraph *output);

// In-Place Compression (Reuses input->col_idx buffer)
bool compressGraphInPlace(CSRGraph *input, CompressedCSRGraph *output);

// Device function for decoding (Header implementation for inlining)
#ifdef __CUDACC__
__device__ __forceinline__ node_t decode_varint(const uint8_t *buffer,
                                                edge_t &offset) {
  node_t value = 0;
  int shift = 0;
  uint8_t byte;

  do {
    byte = buffer[offset++];
    value |= (node_t)(byte & 127) << shift;
    shift += 7;
  } while (byte & 128);

  return value;
}
#endif

#endif // COMPRESSION_H
