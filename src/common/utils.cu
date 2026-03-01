#include "utils.h"
#include <cstring>
#include <queue>

// =============================================================================
// Timer Implementation
// =============================================================================

CudaTimer createTimer() {
  CudaTimer timer;
  CUDA_CHECK(cudaEventCreate(&timer.start));
  CUDA_CHECK(cudaEventCreate(&timer.stop));
  return timer;
}

void startTimer(CudaTimer *timer) {
  CUDA_CHECK(cudaEventRecord(timer->start, 0));
}

float stopTimer(CudaTimer *timer) {
  float elapsed;
  CUDA_CHECK(cudaEventRecord(timer->stop, 0));
  CUDA_CHECK(cudaEventSynchronize(timer->stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, timer->start, timer->stop));
  return elapsed;
}

void destroyTimer(CudaTimer *timer) {
  CUDA_CHECK(cudaEventDestroy(timer->start));
  CUDA_CHECK(cudaEventDestroy(timer->stop));
}

// =============================================================================
// BFS Result Implementation
// =============================================================================

BFSResult *allocBFSResult(node_t num_nodes, node_t source) {
  BFSResult *result = new BFSResult;
  result->num_nodes = num_nodes;
  result->source = source;
  result->elapsed_ms = 0.0f;

  result->distances = new level_t[num_nodes];
  result->parents = nullptr; // Optional, allocate if needed

  // Initialize distances to unvisited
  for (node_t i = 0; i < num_nodes; i++) {
    result->distances[i] = UNVISITED;
  }
  result->distances[source] = 0;

  return result;
}

void freeBFSResult(BFSResult *result) {
  if (!result)
    return;
  delete[] result->distances;
  if (result->parents)
    delete[] result->parents;
  delete result;
}

void printBFSResult(const BFSResult *result) {
  printf("=== BFS Result ===\n");
  printf("Source: %d\n", result->source);
  printf("Time: %.3f ms\n", result->elapsed_ms);

  // Count reachable nodes and max distance
  node_t reachable = 0;
  level_t max_dist = 0;
  for (node_t i = 0; i < result->num_nodes; i++) {
    if (result->distances[i] != UNVISITED) {
      reachable++;
      if (result->distances[i] > max_dist) {
        max_dist = result->distances[i];
      }
    }
  }

  printf("Reachable Nodes: %d / %d (%.2f%%)\n", reachable, result->num_nodes,
         100.0 * reachable / result->num_nodes);
  printf("Max Distance (Diameter): %d\n", max_dist);
  printf("==================\n\n");
}

bool validateBFSResult(const BFSResult *result, const CSRGraph *graph) {
  // Run CPU BFS for reference
  BFSResult *ref = bfsCPU(graph, result->source);

  bool valid = true;
  node_t mismatch_count = 0;

  for (node_t i = 0; i < result->num_nodes; i++) {
    if (result->distances[i] != ref->distances[i]) {
      if (mismatch_count < 10) {
        printf("Mismatch at node %d: got %d, expected %d\n", i,
               result->distances[i], ref->distances[i]);
      }
      mismatch_count++;
      valid = false;
    }
  }

  if (mismatch_count > 0) {
    printf("Total mismatches: %d\n", mismatch_count);
  } else {
    printf("Validation PASSED!\n");
  }

  freeBFSResult(ref);
  return valid;
}

// =============================================================================
// CPU BFS Reference
// =============================================================================

BFSResult *bfsCPU(const CSRGraph *graph, node_t source) {
  BFSResult *result = allocBFSResult(graph->num_nodes, source);

  std::queue<node_t> frontier;
  frontier.push(source);

  while (!frontier.empty()) {
    node_t current = frontier.front();
    frontier.pop();

    level_t current_dist = result->distances[current];

    // Iterate over neighbors
    for (edge_t e = graph->h_row_ptr[current];
         e < graph->h_row_ptr[current + 1]; e++) {
      node_t neighbor = graph->h_col_idx[e];

      if (result->distances[neighbor] == UNVISITED) {
        result->distances[neighbor] = current_dist + 1;
        frontier.push(neighbor);
      }
    }
  }

  return result;
}

// =============================================================================
// Command Line Parsing
// =============================================================================

BFSOptions parseArgs(int argc, char **argv) {
  BFSOptions opts;
  opts.graph_file = nullptr;
  opts.source = 0;
  opts.validate = true;
  opts.verbose = false;
  opts.benchmark = false;
  opts.num_runs = 1;
  opts.json_output = false;
  opts.compression = false;
  opts.algorithm = ALGO_BFS;
  opts.bu_threshold_divisor = 20; // Default: 5% of nodes triggers Bottom-Up
  opts.num_gpus = 1;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
      opts.verbose = true;
    } else if (strcmp(argv[i], "--algo") == 0) {
      if (i + 1 < argc) {
        char *algo = argv[++i];
        if (strcmp(algo, "afforest") == 0) {
          opts.algorithm = ALGO_AFFOREST;
        } else if (strcmp(algo, "adaptive") == 0) {
          opts.algorithm = ALGO_ADAPTIVE;
        } else {
          opts.algorithm = ALGO_BFS;
        }
      }
    } else if (strcmp(argv[i], "-n") == 0 ||
               strcmp(argv[i], "--no-validate") == 0) {
      opts.validate = false;
    } else if (strcmp(argv[i], "--json") == 0) {
      opts.json_output = true;
    } else if (strcmp(argv[i], "--compress") == 0) {
      opts.compression = true;
    } else if (strcmp(argv[i], "-b") == 0 ||
               strcmp(argv[i], "--benchmark") == 0) {
      opts.benchmark = true;
      if (i + 1 < argc) {
        opts.num_runs = atoi(argv[++i]);
      }
    } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--source") == 0) {
      if (i + 1 < argc) {
        opts.source = atoi(argv[++i]);
      }
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      printUsage(argv[0]);
      exit(0);
    } else if (strcmp(argv[i], "--bu-threshold") == 0) {
      if (i + 1 < argc) {
        opts.bu_threshold_divisor = atoi(argv[++i]);
        if (opts.bu_threshold_divisor <= 0)
          opts.bu_threshold_divisor = 20;
      }
    } else if (strcmp(argv[i], "--gpus") == 0) {
      if (i + 1 < argc) {
        opts.num_gpus = atoi(argv[++i]);
        if (opts.num_gpus <= 0)
          opts.num_gpus = 1;
      }
    } else if (argv[i][0] != '-') {
      opts.graph_file = argv[i];
    }
  }

  if (!opts.graph_file) {
    fprintf(stderr, "Error: No graph file specified\n");
    printUsage(argv[0]);
    exit(1);
  }

  return opts;
}

void printUsage(const char *program) {
  printf("Usage: %s [options] <graph_file>\n", program);
  printf("\nOptions:\n");
  printf("  -s, --source <n>     Source node for BFS (default: 0)\n");
  printf("  --algo <type>        Algorithm: 'bfs', 'adaptive', 'afforest'\n");
  printf("  --bu-threshold <n>   Bottom-Up threshold divisor (default: 20 = "
         "5%%)\n");
  printf("                       Switches to Bottom-Up when frontier > N/n\n");
  printf("  -v, --verbose        Enable verbose output\n");
  printf("  -n, --no-validate    Skip validation against CPU\n");
  printf("  -b, --benchmark <n>  Run benchmark with n iterations\n");
  printf(
      "  --gpus <n>           Number of simulated GPUs to use (default: 1)\n");
  printf("  --json               Output results in JSON format\n");
  printf("  --compress           Enable graph compression (Zero-Copy)\n");
  printf("  -h, --help           Show this help message\n");
  printf("\nGraph file formats supported:\n");
  printf("  .txt    Edge list (first line: nodes edges)\n");
  printf("  .csrbin CSR binary format\n");
  printf("  .mat    HDF5/MAT format\n");
}
