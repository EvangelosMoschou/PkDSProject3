#include "common/compression.h"
#include "common/graph.h"
#include "common/utils.h"
#ifdef USE_V41_HYBRID
#include "v4_1_hybrid/bfs_adaptive.h"
#else
#include "v5_multi_gpu/bfs_multi_gpu.h"
#endif
#include <iostream>
#include <omp.h>
#include <vector>

int main(int argc, char **argv) {
  printf("DEBUG: Main started\n");
  fflush(stdout);
  // 1. Parse arguments using existing utility
  BFSOptions options = parseArgs(argc, argv);

  if (!options.graph_file) {
    std::cerr << "Error: No graph file provided." << std::endl;
    printUsage(argv[0]);
    return 1;
  }

#ifdef USE_V41_HYBRID
  std::cout << "=================================================" << std::endl;
  std::cout << " V4.1 Hybrid-Optimized Adaptive BFS / Afforest   " << std::endl;
  std::cout << "=================================================" << std::endl;
#else
  std::cout << "=================================================" << std::endl;
  std::cout << " V5 Multi-GPU (Locally Simulated) BFS / Afforest " << std::endl;
  std::cout << "=================================================" << std::endl;
#endif

  // 2. Load the graph
  std::cout << "Loading graph from " << options.graph_file << "..."
            << std::endl;
  CSRGraph *graph = nullptr;

  // Auto-detect extension or use default logic. For now use base Loader.
  // In your actual code you might switch on .bin vs .mtx, etc.
  // We'll use the .bin if it ends in bin, else try standard loadGraph.
  std::string filename(options.graph_file);
  if (filename.find(".bin") != std::string::npos) {
    graph = loadGraphCSRBin(options.graph_file);
  } else if (filename.find(".h5") != std::string::npos ||
             filename.find(".mat") != std::string::npos) {
    graph = loadGraphHDF5(options.graph_file);
  } else {
    graph = loadGraph(options.graph_file);
  }

  if (!graph) {
    std::cerr << "Failed to load graph." << std::endl;
    return 1;
  }

  printGraphStats(graph);

  // 3. Move graph to Device (Currently unified Device 0)
  copyGraphToDevice(graph);

  // 4. Run the Algorithm
  if (options.algorithm == ALGO_AFFOREST) {
    solveAfforest(graph);
  } else {
    BFSResult *result = nullptr;
    if (options.compression) {
#ifdef USE_V41_HYBRID
      std::cerr
          << "Compressed BFS not available in V4.1. Running standard BFS..."
          << std::endl;
      result = solveBFSAdaptiveWithThreshold(graph, options.source,
                                             options.bu_threshold_divisor);
#else
      std::cout << "\nCompressing Graph (c-CSR)..." << std::endl;
      CompressedCSRGraph comp_graph;
      compressGraph(graph, &comp_graph);
      setupCompressedGraphDevice(&comp_graph);

      if (options.num_gpus > 1) {
        std::cout << "Running V5 Simulated Multi-GPU BFS (Compressed, "
                  << options.num_gpus << " GPUs)..." << std::endl;
        result = solveBFSCompressedMultiGPUSimulated(
            &comp_graph, options.source, options.num_gpus);
      } else {
        std::cout << "Running Adaptive BFS (Compressed Single GPU)..."
                  << std::endl;
        result = solveBFSCompressedAdaptive(&comp_graph, options.source);
      }
#endif
    } else if (options.num_gpus > 1) {
#ifdef USE_V41_HYBRID
      std::cerr << "Multi-GPU not supported in V4.1. Running single GPU..."
                << std::endl;
      result = solveBFSAdaptiveWithThreshold(graph, options.source,
                                             options.bu_threshold_divisor);
#else
      std::cout << "\nRunning V5 Simulated Multi-GPU BFS (" << options.num_gpus
                << " GPUs)..." << std::endl;
      result =
          solveBFSMultiGPUSimulated(graph, options.source, options.num_gpus);
#endif
    } else {
      std::cout << "\nRunning Adaptive BFS (Single GPU)..." << std::endl;
      result = solveBFSAdaptiveWithThreshold(graph, options.source,
                                             options.bu_threshold_divisor);
    }

    if (result) {
      printBFSResult(result);

      if (options.validate) {
        std::cout << "Validating result against CPU..." << std::endl;
        bool valid = validateBFSResult(result, graph);
        if (valid) {
          std::cout << "[SUCCESS] GPU Result matches CPU reference."
                    << std::endl;
        } else {
          std::cout << "[FAILED] GPU Result does NOT match CPU reference."
                    << std::endl;
        }
      }
      freeBFSResult(result);
    }
  }

  freeGraph(graph);
  std::cout << "Done." << std::endl;

  return 0;
}
