#include "cuda_common.h"
#include "reorder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  edge_t degree;
  node_t node;
} DegreeNode;

static int compare_node_t_asc(const void *a, const void *b) {
  node_t va = *(const node_t *)a;
  node_t vb = *(const node_t *)b;
  if (va < vb)
    return -1;
  if (va > vb)
    return 1;
  return 0;
}

static int compare_degree_desc(const void *a, const void *b) {
  const DegreeNode *da = (const DegreeNode *)a;
  const DegreeNode *db = (const DegreeNode *)b;
  if (da->degree > db->degree)
    return -1;
  if (da->degree < db->degree)
    return 1;
  if (da->node < db->node)
    return -1;
  if (da->node > db->node)
    return 1;
  return 0;
}

static node_t findMaxDegreeNode(const CSRGraph *g) {
  node_t max_node = 0;
  edge_t max_deg = 0;
  for (node_t i = 0; i < g->num_nodes; i++) {
    edge_t deg = g->h_row_ptr[i + 1] - g->h_row_ptr[i];
    if (deg > max_deg) {
      max_deg = deg;
      max_node = i;
    }
  }
  return max_node;
}

static void initOrderArrays(node_t *new_to_old, node_t *old_to_new, node_t n) {
  for (node_t i = 0; i < n; i++) {
    new_to_old[i] = (node_t)-1;
    old_to_new[i] = (node_t)-1;
  }
}

static void computeBFSOrder(const CSRGraph *graph, node_t *new_to_old,
                            node_t *old_to_new) {
  node_t n = graph->num_nodes;
  initOrderArrays(new_to_old, old_to_new, n);

  unsigned char *visited = (unsigned char *)calloc((size_t)n, sizeof(unsigned char));
  node_t *queue = (node_t *)malloc((size_t)n * sizeof(node_t));
  if (!visited || !queue) {
    free(visited);
    free(queue);
    return;
  }

  node_t head = 0;
  node_t tail = 0;
  node_t new_id_counter = 0;
  node_t start_node = findMaxDegreeNode(graph);
  queue[tail++] = start_node;
  visited[start_node] = 1;

  while (new_id_counter < n) {
    if (head >= tail) {
      for (node_t i = 0; i < n; i++) {
        if (!visited[i]) {
          queue[tail++] = i;
          visited[i] = 1;
          break;
        }
      }
    }
    if (head >= tail)
      break;

    node_t u = queue[head++];
    node_t nid = new_id_counter++;
    old_to_new[u] = nid;
    new_to_old[nid] = u;

    for (edge_t e = graph->h_row_ptr[u]; e < graph->h_row_ptr[u + 1]; e++) {
      node_t v = graph->h_col_idx[e];
      if (!visited[v]) {
        visited[v] = 1;
        queue[tail++] = v;
      }
    }
  }

  free(visited);
  free(queue);
}

static void computeGapAwareBFSOrder(const CSRGraph *graph, node_t *new_to_old,
                                    node_t *old_to_new) {
  node_t n = graph->num_nodes;
  initOrderArrays(new_to_old, old_to_new, n);

  unsigned char *visited = (unsigned char *)calloc((size_t)n, sizeof(unsigned char));
  node_t *queue = (node_t *)malloc((size_t)n * sizeof(node_t));
  node_t *neighbors = NULL;
  size_t neighbors_cap = 0;
  if (!visited || !queue) {
    free(visited);
    free(queue);
    return;
  }

  node_t head = 0;
  node_t tail = 0;
  node_t new_id_counter = 0;

  queue[tail++] = 0;
  visited[0] = 1;

  printf("  Computing Gap-Aware BFS Order...\n");

  while (new_id_counter < n) {
    if (head >= tail) {
      for (node_t i = 0; i < n; i++) {
        if (!visited[i]) {
          queue[tail++] = i;
          visited[i] = 1;
          break;
        }
      }
    }
    if (head >= tail)
      break;

    node_t u = queue[head++];
    node_t nid = new_id_counter++;
    old_to_new[u] = nid;
    new_to_old[nid] = u;

    edge_t start = graph->h_row_ptr[u];
    edge_t end = graph->h_row_ptr[u + 1];
    size_t count = 0;

    for (edge_t e = start; e < end; e++) {
      node_t v = graph->h_col_idx[e];
      if (!visited[v]) {
        if (count >= neighbors_cap) {
          size_t new_cap = neighbors_cap == 0 ? 1024 : neighbors_cap * 2;
          node_t *tmp = (node_t *)realloc(neighbors, new_cap * sizeof(node_t));
          if (!tmp) {
            free(neighbors);
            free(visited);
            free(queue);
            return;
          }
          neighbors = tmp;
          neighbors_cap = new_cap;
        }
        neighbors[count++] = v;
        visited[v] = 1;
      }
    }

    if (count > 1) {
      qsort(neighbors, count, sizeof(node_t), compare_node_t_asc);
    }
    for (size_t i = 0; i < count; i++) {
      queue[tail++] = neighbors[i];
    }

    if (new_id_counter % 10000000 == 0) {
      printf("    Processed %d / %d nodes (%.1f%%)\n", new_id_counter, n,
             100.0 * new_id_counter / n);
    }
  }

  printf("  Gap-Aware BFS Order complete.\n");
  free(neighbors);
  free(visited);
  free(queue);
}

static void computeDegreeOrder(const CSRGraph *graph, node_t *new_to_old,
                               node_t *old_to_new) {
  node_t n = graph->num_nodes;
  initOrderArrays(new_to_old, old_to_new, n);

  DegreeNode *nodes = (DegreeNode *)malloc((size_t)n * sizeof(DegreeNode));
  if (!nodes)
    return;

  for (node_t i = 0; i < n; i++) {
    nodes[i].degree = graph->h_row_ptr[i + 1] - graph->h_row_ptr[i];
    nodes[i].node = i;
  }

  qsort(nodes, (size_t)n, sizeof(DegreeNode), compare_degree_desc);

  for (node_t i = 0; i < n; i++) {
    node_t old_id = nodes[i].node;
    old_to_new[old_id] = i;
    new_to_old[i] = old_id;
  }

  free(nodes);
}

static int computeOrder(const CSRGraph *graph, ReorderMethod method,
                        node_t *new_to_old, node_t *old_to_new) {
  if (method == REORDER_BFS) {
    printf("  Computing Standard BFS Order...\n");
    computeBFSOrder(graph, new_to_old, old_to_new);
  } else if (method == REORDER_GAP_BFS) {
    computeGapAwareBFSOrder(graph, new_to_old, old_to_new);
  } else {
    printf("  Computing Degree Order...\n");
    computeDegreeOrder(graph, new_to_old, old_to_new);
  }
  return 0;
}

void reorderAndSaveStreaming(const CSRGraph *graph, const char *out_filename,
                             ReorderMethod method) {
  printf("Reordering Graph: %d Nodes, %lld Edges. Method: %d\n",
         graph->num_nodes, (long long)graph->num_edges, (int)method);

  node_t *new_to_old = (node_t *)malloc((size_t)graph->num_nodes * sizeof(node_t));
  node_t *old_to_new = (node_t *)malloc((size_t)graph->num_nodes * sizeof(node_t));
  if (!new_to_old || !old_to_new) {
    free(new_to_old);
    free(old_to_new);
    return;
  }
  computeOrder(graph, method, new_to_old, old_to_new);

  printf("  Streaming Reordered Graph to Disk: %s\n", out_filename);
  FILE *file = fopen(out_filename, "wb");
  if (!file) {
    fprintf(stderr, "Error: Cannot create file %s\n", out_filename);
    free(new_to_old);
    free(old_to_new);
    return;
  }

  unsigned long long n = (unsigned long long)graph->num_nodes;
  unsigned long long m = (unsigned long long)graph->num_edges;
  fwrite(&n, sizeof(unsigned long long), 1, file);
  fwrite(&m, sizeof(unsigned long long), 1, file);

  edge_t *new_row_ptr = (edge_t *)malloc((size_t)(graph->num_nodes + 1) * sizeof(edge_t));
  if (!new_row_ptr) {
    fclose(file);
    free(new_to_old);
    free(old_to_new);
    return;
  }
  new_row_ptr[0] = 0;
  for (node_t i = 0; i < graph->num_nodes; i++) {
    node_t u_old = new_to_old[i];
    edge_t degree = graph->h_row_ptr[u_old + 1] - graph->h_row_ptr[u_old];
    new_row_ptr[i + 1] = new_row_ptr[i] + degree;
  }
  fwrite(new_row_ptr, sizeof(edge_t), (size_t)graph->num_nodes + 1, file);

  size_t buffer_cap = 1024 * 1024 * 16;
  node_t *buffer = (node_t *)malloc(buffer_cap * sizeof(node_t));
  node_t *row_neighbors = NULL;
  size_t row_neighbors_cap = 0;
  size_t buffer_idx = 0;

  if (!buffer) {
    free(new_row_ptr);
    fclose(file);
    free(new_to_old);
    free(old_to_new);
    return;
  }

  printf("  Streaming Edges... (Total: %lld)\n", (long long)m);
  for (node_t i = 0; i < graph->num_nodes; i++) {
    node_t u_old = new_to_old[i];
    edge_t start = graph->h_row_ptr[u_old];
    edge_t end = graph->h_row_ptr[u_old + 1];
    size_t row_size = (size_t)(end - start);

    if (row_size > row_neighbors_cap) {
      node_t *tmp = (node_t *)realloc(row_neighbors, row_size * sizeof(node_t));
      if (!tmp) {
        free(row_neighbors);
        free(buffer);
        free(new_row_ptr);
        fclose(file);
        free(new_to_old);
        free(old_to_new);
        return;
      }
      row_neighbors = tmp;
      row_neighbors_cap = row_size;
    }

    for (size_t j = 0; j < row_size; j++) {
      node_t v_old = graph->h_col_idx[start + (edge_t)j];
      row_neighbors[j] = old_to_new[v_old];
    }

    if (row_size > 1) {
      qsort(row_neighbors, row_size, sizeof(node_t), compare_node_t_asc);
    }

    for (size_t j = 0; j < row_size; j++) {
      buffer[buffer_idx++] = row_neighbors[j];
      if (buffer_idx >= buffer_cap) {
        fwrite(buffer, sizeof(node_t), buffer_cap, file);
        buffer_idx = 0;
      }
    }

    if (i % 5000000 == 0 && i > 0) {
      printf("\r    Processed %d / %d Nodes (%.1f%%)", i, graph->num_nodes,
             100.0 * i / graph->num_nodes);
      fflush(stdout);
    }
  }

  if (buffer_idx > 0) {
    fwrite(buffer, sizeof(node_t), buffer_idx, file);
  }
  printf("\n  Done.\n");

  free(row_neighbors);
  free(buffer);
  free(new_row_ptr);
  fclose(file);
  free(new_to_old);
  free(old_to_new);
}

CSRGraph *reorderGraph(const CSRGraph *graph, ReorderMethod method) {
  if (graph->num_edges > 1000000000LL) {
    fprintf(stderr, "WARNING: reorderGraph on massive graph. Use streaming.\n");
  }

  printf("Reordering Graph: %d Nodes, %lld Edges. Method: %d\n",
         graph->num_nodes, (long long)graph->num_edges, (int)method);

  node_t *new_to_old = (node_t *)malloc((size_t)graph->num_nodes * sizeof(node_t));
  node_t *old_to_new = (node_t *)malloc((size_t)graph->num_nodes * sizeof(node_t));
  if (!new_to_old || !old_to_new) {
    free(new_to_old);
    free(old_to_new);
    return NULL;
  }

  computeOrder(graph, method, new_to_old, old_to_new);

  printf("  Reconstructing Graph...\n");
  CSRGraph *new_graph = (CSRGraph *)malloc(sizeof(CSRGraph));
  if (!new_graph) {
    free(new_to_old);
    free(old_to_new);
    return NULL;
  }

  new_graph->num_nodes = graph->num_nodes;
  new_graph->num_edges = graph->num_edges;
  new_graph->d_row_ptr = NULL;
  new_graph->d_col_idx = NULL;

  CUDA_CHECK(cudaMallocHost(&new_graph->h_row_ptr,
                            (size_t)(new_graph->num_nodes + 1) * sizeof(edge_t)));
  CUDA_CHECK(cudaMallocHost(&new_graph->h_col_idx,
                            (size_t)new_graph->num_edges * sizeof(node_t)));

  new_graph->h_row_ptr[0] = 0;
  for (node_t i = 0; i < graph->num_nodes; i++) {
    node_t u_old = new_to_old[i];
    edge_t degree = graph->h_row_ptr[u_old + 1] - graph->h_row_ptr[u_old];
    new_graph->h_row_ptr[i + 1] = new_graph->h_row_ptr[i] + degree;
  }

  node_t *row_neighbors = NULL;
  size_t row_neighbors_cap = 0;

  for (node_t i = 0; i < graph->num_nodes; i++) {
    node_t u_old = new_to_old[i];
    edge_t start = graph->h_row_ptr[u_old];
    edge_t end = graph->h_row_ptr[u_old + 1];
    edge_t new_start = new_graph->h_row_ptr[i];
    size_t row_size = (size_t)(end - start);

    if (row_size > row_neighbors_cap) {
      node_t *tmp = (node_t *)realloc(row_neighbors, row_size * sizeof(node_t));
      if (!tmp) {
        free(row_neighbors);
        free(new_to_old);
        free(old_to_new);
        return new_graph;
      }
      row_neighbors = tmp;
      row_neighbors_cap = row_size;
    }

    for (size_t j = 0; j < row_size; j++) {
      node_t v_old = graph->h_col_idx[start + (edge_t)j];
      row_neighbors[j] = old_to_new[v_old];
    }

    if (row_size > 1) {
      qsort(row_neighbors, row_size, sizeof(node_t), compare_node_t_asc);
    }

    for (size_t j = 0; j < row_size; j++) {
      new_graph->h_col_idx[new_start + (edge_t)j] = row_neighbors[j];
    }
  }

  free(row_neighbors);
  free(new_to_old);
  free(old_to_new);
  return new_graph;
}
