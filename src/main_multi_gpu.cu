#include "common/compression.h"
#include "common/graph.h"
#include "common/utils.h"
#ifdef USE_V41_HYBRID
#include "v4_1_hybrid/bfs_adaptive.h"
#else
#include "v5_multi_gpu/bfs_multi_gpu.h"
#endif
#include <stdio.h>
#include <string.h>
#include <omp.h>

int main(int argc, char **argv) {
  printf("DEBUG: Main started\n");
  fflush(stdout);
  // 1. Parse arguments using existing utility
  BFSOptions options = parseArgs(argc, argv);

  if (!options.graph_file) {
    fprintf(stderr, "Error: No graph file provided.\n");
    printUsage(argv[0]);
    return 1;
  }

#ifdef USE_V41_HYBRID
  printf("=================================================\n");
  printf(" V4.1 Hybrid-Optimized Adaptive BFS / Afforest   \n");
  printf("=================================================\n");
#else
  printf("=================================================\n");
  printf(" V5 Multi-GPU (Locally Simulated) BFS / Afforest \n");
  printf("=================================================\n");
#endif

  // 2. Load the graph
  printf("Loading graph from %s...\n", options.graph_file);
  CSRGraph *graph = NULL;

  // Auto-detect extension or use default logic. For now use base Loader.
  // In your actual code you might switch on .bin vs .mtx, etc.
  // We'll use the .bin if it ends in bin, else try standard loadGraph.
  if (strstr(options.graph_file, ".bin") != NULL) {
    graph = loadGraphCSRBin(options.graph_file);
  } else if (strstr(options.graph_file, ".h5") != NULL ||
             strstr(options.graph_file, ".mat") != NULL) {
    graph = loadGraphHDF5(options.graph_file);
  } else {
    graph = loadGraph(options.graph_file);
  }

  if (!graph) {
    fprintf(stderr, "Failed to load graph.\n");
    return 1;
  }

  printGraphStats(graph);

  // 3. Move graph to Device (Currently unified Device 0)
  copyGraphToDevice(graph);

  // 4. Run the Algorithm
  if (options.algorithm == ALGO_AFFOREST) {
    solveAfforest(graph);
  } else {
    BFSResult *result = NULL;
    if (options.compression) {
#ifdef USE_V41_HYBRID
      printf("\nCompressing Graph (c-CSR)...\n");
      CompressedCSRGraph comp_graph;
      compressGraph(graph, &comp_graph);
      setupCompressedGraphDevice(&comp_graph);
      printf("Running Adaptive BFS (Compressed Single GPU, V4.1)...\n");
      result = solveBFSCompressedAdaptive(&comp_graph, options.source);
#else
      printf("\nCompressing Graph (c-CSR)...\n");
      CompressedCSRGraph comp_graph;
      compressGraph(graph, &comp_graph);
      setupCompressedGraphDevice(&comp_graph);

      if (options.num_gpus > 1) {
        printf("Running V5 Simulated Multi-GPU BFS (Compressed, %d GPUs)...\n",
               options.num_gpus);
        result = solveBFSCompressedMultiGPUSimulated(
            &comp_graph, options.source, options.num_gpus);
      } else {
        printf("Running Adaptive BFS (Compressed Single GPU)...\n");
        result = solveBFSCompressedAdaptive(&comp_graph, options.source);
      }
#endif
    } else if (options.num_gpus > 1) {
#ifdef USE_V41_HYBRID
      fprintf(stderr,
              "Multi-GPU not supported in V4.1. Running single GPU...\n");
      result = solveBFSAdaptiveWithThreshold(graph, options.source,
                                             options.bu_threshold_divisor);
#else
      printf("\nRunning V5 Simulated Multi-GPU BFS (%d GPUs)...\n",
             options.num_gpus);
      result =
          solveBFSMultiGPUSimulated(graph, options.source, options.num_gpus);
#endif
    } else {
      printf("\nRunning Adaptive BFS (Single GPU)...\n");
      result = solveBFSAdaptiveWithThreshold(graph, options.source,
                                             options.bu_threshold_divisor);
    }

    if (result) {
      printBFSResult(result);

      if (options.validate) {
        printf("Validating result against CPU...\n");
        bool valid = validateBFSResult(result, graph);
        if (valid) {
          printf("[SUCCESS] GPU Result matches CPU reference.\n");
        } else {
          printf("[FAILED] GPU Result does NOT match CPU reference.\n");
        }
      }
      freeBFSResult(result);
    }
  }

  freeGraph(graph);
  printf("Done.\n");

  return 0;
}
