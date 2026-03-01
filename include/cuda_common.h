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

// BFS level constants (optimized for native atomics)
#define UNVISITED 0xFFFFFFFF
#define INF_DISTANCE 0xFFFFFFFE

// =============================================================================
// Type Definitions
// =============================================================================

typedef int node_t;       // Node ID type
typedef long long edge_t; // Edge count type
typedef unsigned int
    level_t; // BFS level type (native 32-bit for hardware atomics)

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
