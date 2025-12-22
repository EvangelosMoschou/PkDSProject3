# Test Validation

This directory contains validation utilities for verifying BFS correctness.

## Validation Process

1. Run CUDA BFS implementation
2. Run CPU reference implementation
3. Compare distance arrays
4. Report any mismatches

## Adding Tests

Place test graph files in `tests/graphs/` directory with `.txt` extension.

Graph format:
```
num_nodes num_edges
src1 dst1
src2 dst2
...
```
