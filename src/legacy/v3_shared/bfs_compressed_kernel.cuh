#ifndef BFS_COMPRESSED_KERNEL_CUH
#define BFS_COMPRESSED_KERNEL_CUH

#include "compression.h"
#include "cuda_common.h"
#include "graph.h"

// Thread-Centric Compressed BFS Kernel (Small Nodes)
__global__ void bfsCompressedThreadKernel(
    const edge_t *__restrict__ row_ptr,         // Byte offsets
    const uint8_t *__restrict__ compressed_col, // Byte stream
    level_t *__restrict__ distances, const node_t *__restrict__ frontier,
    const int frontier_size, node_t *__restrict__ next_frontier,
    int *__restrict__ next_frontier_size, const level_t current_level);

// Warp-Cooperative Compressed BFS Kernel
__global__ void bfsCompressedWarpKernel(
    const edge_t *__restrict__ row_ptr,         // Byte offsets
    const uint8_t *__restrict__ compressed_col, // Byte stream
    level_t *__restrict__ distances, const node_t *__restrict__ frontier,
    const int frontier_size, node_t *__restrict__ next_frontier,
    int *__restrict__ next_frontier_size, const level_t current_level);

#endif
