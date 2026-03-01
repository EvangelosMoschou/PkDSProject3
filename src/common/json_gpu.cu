#include "json_gpu.h"
#include <math.h>
#include <string.h>
#include <time.h>

// Comparison function for qsort
static int compare_doubles(const void *a, const void *b) {
  double da = *(const double *)a;
  double db = *(const double *)b;
  return (da > db) - (da < db);
}

void print_json_gpu(const char *algo_name, const char *graph_file,
                    node_t num_nodes, edge_t num_edges, double *times_ms,
                    int num_trials, edge_t traversed_edges,
                    bool used_streaming) {

  // Sort times for stats
  if (num_trials > 0 && times_ms != NULL) {
    qsort(times_ms, num_trials, sizeof(double), compare_doubles);
  }

  double min_t = 0, max_t = 0, mean_t = 0, median_t = 0, std_dev = 0;
  if (num_trials > 0) {
    min_t = times_ms[0];
    max_t = times_ms[num_trials - 1];

    double sum = 0;
    for (int i = 0; i < num_trials; i++)
      sum += times_ms[i];
    mean_t = sum / num_trials;

    if (num_trials % 2 == 1)
      median_t = times_ms[num_trials / 2];
    else
      median_t =
          (times_ms[num_trials / 2 - 1] + times_ms[num_trials / 2]) / 2.0;

    double sum_sq = 0;
    for (int i = 0; i < num_trials; i++)
      sum_sq += (times_ms[i] - mean_t) * (times_ms[i] - mean_t);
    std_dev = sqrt(sum_sq / num_trials);
  }

  // Get Device Info
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

  // Get Timestamp
  time_t now = time(NULL);
  struct tm *t = localtime(&now);
  char time_str[64];
  strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", t);

  // Calculate GTEPS (Edges / Time) based on MEAN time
  // GTEPS = (Edges / 1e9) / (Time_ms / 1000)
  double gteps = 0.0;
  if (mean_t > 0) {
    gteps = (double)traversed_edges /
            (mean_t * 1e6); // edges / (ms * 10^6) -> edges * 10^3 / ms * 10^9 ?
                            // Edges / (Time_s * 10^9)
                            // = Edges / (Time_ms/1000 * 10^9)
                            // = Edges / (Time_ms * 10^6)
  }

  printf("{\n");
  printf("  \"algorithm\": \"%s\",\n", algo_name);
  printf("  \"graph_file\": \"%s\",\n", graph_file);
  printf("  \"streaming_enabled\": %s,\n", used_streaming ? "true" : "false");
  printf("  \"num_nodes\": %d,\n", num_nodes);
  printf("  \"num_edges\": %lld,\n", num_edges);
  printf("  \"traversed_edges\": %lld,\n", traversed_edges);
  printf("  \"device_info\": {\n");
  printf("    \"name\": \"%s\",\n", prop.name);
  printf("    \"compute_capability\": \"%d.%d\",\n", prop.major, prop.minor);
  printf("    \"global_mem_gb\": %.2f,\n",
         prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  printf("    \"warp_size\": %d\n", prop.warpSize);
  printf("  },\n");
  printf("  \"sys_info\": {\n");
  printf("    \"timestamp\": \"%s\"\n", time_str);
  printf("  },\n");
  printf("  \"stats\": {\n");
  printf("    \"num_trials\": %d,\n", num_trials);
  printf("    \"min_ms\": %.4f,\n", min_t);
  printf("    \"max_ms\": %.4f,\n", max_t);
  printf("    \"mean_ms\": %.4f,\n", mean_t);
  printf("    \"median_ms\": %.4f,\n", median_t);
  printf("    \"std_dev_ms\": %.4f,\n", std_dev);
  printf("    \"gteps\": %.4f\n", gteps);
  printf("  }\n");
  printf("}\n");
}
