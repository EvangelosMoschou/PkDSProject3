# Parallel BFS using CUDA - Assignment Description

## Overview

This assignment focuses on implementing Breadth-First Search (BFS) on graphs using CUDA, exploring different parallelization strategies with GPU threads.

## Implementation Versions

### Version 1: Dynamic Thread Assignment (One Thread per Node)

**Concept**: Each thread is assigned to explore one node of the graph. Since the number of threads is limited compared to the number of nodes, once a thread finishes processing its assigned node, it dynamically picks up another unprocessed node.

**Characteristics**:
- Dynamic work distribution
- Good load balancing when node degrees vary significantly
- Thread picks up new work as it becomes available
- Similar to a work-stealing approach

**Implementation Notes**:
- Use atomic operations for work queue management
- Each thread processes one node at a time
- Threads continue until all nodes are processed

---

### Version 2: Chunk-Based Processing (One Thread for Multiple Nodes)

**Concept**: Similar to the pthreads approach, each thread is responsible for processing a fixed chunk of nodes. The kernel contains an internal for-loop that iterates over the assigned node range.

**Characteristics**:
- Static work distribution
- Simple implementation
- Each thread processes `ceil(N / num_threads)` nodes
- Kernel contains a for-loop over assigned nodes

**Implementation Notes**:
- Divide nodes evenly among threads
- Each thread iterates through its assigned range
- Less synchronization overhead compared to Version 1

---

### Version 3: Shared Memory with Warp Cooperation

**Concept**: Leverage shared memory for collaboration between threads within the same block. Each block contains multiple warps (groups of 32 threads), and threads within the same warp cooperate to load and process neighbors of the same node.

**Characteristics**:
- Utilizes shared memory for intra-block communication
- Warp-level parallelism for neighbor loading
- Each thread in a warp loads a portion of the current node's neighbors
- Better memory coalescing and cache utilization

**Implementation Notes**:
- Use `__shared__` memory for frontier and neighbor data
- Threads in a warp collaborate on the same node's neighbors
- Synchronization within blocks using `__syncthreads()`
- Warp-level primitives can be used for efficient reductions

---

## Optional Extension: One Thread for Multiple Nodes (Alternative)

**Concept**: A single thread handles multiple nodes throughout the BFS traversal.

**Note**: This approach is generally not as efficient due to:
- Reduced parallelism
- Potential load imbalance
- Less effective use of GPU resources

---

## Technical Requirements

### Graph Representation
- Use CSR (Compressed Sparse Row) format for efficient GPU processing
- Row pointers and column indices arrays

### CUDA Considerations
- Proper memory management (host/device transfers)
- Atomic operations for concurrent updates
- Synchronization strategies
- Block and grid dimension optimization

### Performance Metrics
- Execution time for each version
- Comparison with sequential CPU implementation
- Scalability analysis with different graph sizes

## Deliverables

1. Source code for all three versions
2. Performance comparison report
3. Documentation of implementation choices
4. Analysis of results and observations
