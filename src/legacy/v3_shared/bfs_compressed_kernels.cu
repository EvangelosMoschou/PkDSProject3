#define CUDA_ATOMICS_IMPL
#include "bfs_compressed_kernel.cuh"
#include "cuda_common.h"
#include "graph.h"

#define WARPS_PER_BLOCK 32
#define SHARED_NEIGHBORS_PER_WARP 32

// Thread-Centric Compressed BFS Kernel (Small Nodes)
// One Thread per Node. Serial Decoding.
__global__ void bfsCompressedThreadKernel(
    const edge_t *__restrict__ row_ptr,         // Byte offsets
    const uint8_t *__restrict__ compressed_col, // Byte stream
    level_t *__restrict__ distances, const node_t *__restrict__ frontier,
    const int frontier_size, node_t *__restrict__ next_frontier,
    int *__restrict__ next_frontier_size, const level_t current_level) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < frontier_size) {
    node_t current = frontier[tid];

    // 1. Get Byte Range
    edge_t curr_byte_offset = row_ptr[current];
    edge_t end_byte = row_ptr[current + 1];

    if (curr_byte_offset == end_byte)
      return;

    // 2. Serial Delta Decoding
    node_t prev_neighbor = 0;

    while (curr_byte_offset < end_byte) {
      // Inline Varint Decode
      node_t val = 0;
      int shift = 0;
      uint8_t byte;
      do {
        byte = compressed_col[curr_byte_offset++];
        val |= (node_t)(byte & 127) << shift;
        shift += 7;
      } while (byte & 128);

      node_t neighbor = prev_neighbor + val;
      prev_neighbor = neighbor;

      // 3. Visit Neighbor
      level_t old =
          atomicCAS(&distances[neighbor], UNVISITED, current_level + 1);
      if (old == UNVISITED) {
        int pos = atomicAdd(next_frontier_size, 1);
        next_frontier[pos] = neighbor;
      }
    }
  }
}

// Warp-Cooperative Compressed BFS Kernel
__global__ void bfsCompressedWarpKernel(
    const edge_t *__restrict__ row_ptr,         // Byte offsets
    const uint8_t *__restrict__ compressed_col, // Byte stream
    level_t *__restrict__ distances, const node_t *__restrict__ frontier,
    const int frontier_size, node_t *__restrict__ next_frontier,
    int *__restrict__ next_frontier_size, const level_t current_level) {

  __shared__ node_t s_neighbors[WARPS_PER_BLOCK][SHARED_NEIGHBORS_PER_WARP];

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;

  if (global_warp_id < frontier_size) {
    node_t current = frontier[global_warp_id];

    edge_t start_byte = row_ptr[current];
    edge_t end_byte = row_ptr[current + 1];

    if (start_byte == end_byte)
      return;

    edge_t curr_byte_offset = start_byte;
    node_t prev_neighbor = 0;

    while (curr_byte_offset < end_byte) {
      int neighbors_count = 0;
      int bytes_processed = 0;

      // Serial Decoding by Leader
      if (lane_id == 0) {
        edge_t temp_offset = curr_byte_offset;
        for (int i = 0; i < SHARED_NEIGHBORS_PER_WARP; i++) {
          if (temp_offset >= end_byte)
            break;

          // Inline decode logic
          node_t val = 0;
          int shift = 0;
          uint8_t byte;
          do {
            byte = compressed_col[temp_offset++];
            val |= (node_t)(byte & 127) << shift;
            shift += 7;
          } while (byte & 128);

          node_t neighbor = val + prev_neighbor;
          prev_neighbor = neighbor;

          s_neighbors[warp_id][i] = neighbor;
          neighbors_count++;
        }
        bytes_processed = (int)(temp_offset - curr_byte_offset);
      }

      // Broadcast count and bytes processed
      neighbors_count = __shfl_sync(0xFFFFFFFF, neighbors_count, 0);
      bytes_processed = __shfl_sync(0xFFFFFFFF, bytes_processed, 0);

      // Update offset for ALL threads
      curr_byte_offset += bytes_processed;

      // Process Batch
      if (neighbors_count > 0) {
        bool found = false;
        node_t neighbor = 0;

        if (lane_id < neighbors_count) {
          neighbor = s_neighbors[warp_id][lane_id];

          // FIX: Use 32-bit Atomic CAS
          level_t old_val =
              atomicCAS(&distances[neighbor], UNVISITED, current_level + 1);
          if (old_val == UNVISITED) {
            found = true;
          }
        }

        // Converged Execution for Ballot
        unsigned int ballot = __ballot_sync(0xFFFFFFFF, found);
        int pop_count = __popc(ballot);

        if (pop_count > 0) {
          int base_idx = 0;
          if (lane_id == 0)
            base_idx = atomicAdd(next_frontier_size, pop_count);
          base_idx = __shfl_sync(0xFFFFFFFF, base_idx, 0);

          unsigned int lower_mask = (1u << lane_id) - 1;
          int local_offset = __popc(ballot & lower_mask);

          if (found) {
            next_frontier[base_idx + local_offset] = neighbor;
          }
        }
      }
      __syncwarp();
    }
  }
}
