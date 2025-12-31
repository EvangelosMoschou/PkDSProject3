#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// =============================================================================
// CUDA Error Checking Macros
// =============================================================================

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_LAST()                                                      \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// =============================================================================
// Constants
// =============================================================================

// Thread block sizes
#define BLOCK_SIZE_1D 256
#define BLOCK_SIZE_2D 16

// Warp size (NVIDIA GPUs)
#define WARP_SIZE 32

// Maximum shared memory per block (adjust based on GPU architecture)
#define MAX_SHARED_MEM 49152 // 48 KB

// BFS level constants (optimized for uint8_t)
#define UNVISITED 255
#define INF_DISTANCE 254

// =============================================================================
// Type Definitions
// =============================================================================

typedef int node_t;            // Node ID type
typedef long long edge_t;      // Edge count type
typedef unsigned char level_t; // BFS level type (max 255 levels)

// =============================================================================
// Utility Functions
// =============================================================================

// Get optimal grid dimensions
inline dim3 getGridDim(int n, int blockSize) {
  return dim3((n + blockSize - 1) / blockSize);
}

// Get device properties
inline void printDeviceInfo() {
  int device;
  cudaDeviceProp prop;

  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  printf("=== GPU Device Info ===\n");
  printf("Device: %s\n", prop.name);
  printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
  printf("Total Global Memory: %.2f GB\n",
         prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
  printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
  printf("Warp Size: %d\n", prop.warpSize);
  printf("Max Grid Size: (%d, %d, %d)\n", prop.maxGridSize[0],
         prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("=======================\n\n");
}

#endif // CUDA_COMMON_H

// =============================================================================
// Device Helper Functions (must be after #endif for guard inclusion reasons)
// Include only once in .cu files that need it
// =============================================================================

#ifdef CUDA_ATOMICS_IMPL
// Helper for 8-bit atomic CAS (Simulates byte atomic on 32-bit word)
__device__ inline unsigned char atomicCAS_uint8(unsigned char *address,
                                                unsigned char compare,
                                                unsigned char val) {
  unsigned int *base = (unsigned int *)((size_t)address & ~3);
  unsigned int shift = ((size_t)address & 3) * 8;
  unsigned int mask = 0xFF << shift;
  unsigned int old = *base, assumed;
  do {
    assumed = old;
    if (((old >> shift) & 0xFF) != compare)
      return (old >> shift) & 0xFF;
    old =
        atomicCAS(base, assumed, (old & ~mask) | ((unsigned int)val << shift));
  } while (assumed != old);
  return (old >> shift) & 0xFF;
}
#endif
