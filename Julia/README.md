# Julia Implementation for Google Colab (V4 Compatible)

Parallel BFS & Afforest implementation in Julia with CUDA.jl for GPU acceleration.
**Matches the V4 CUDA implementation** with Delta-Compressed CSR support.

## 📁 Files

| File | Description |
|------|-------------|
| `setup_julia_colab.ipynb` | Main notebook with BFS & Afforest (CPU/GPU) |
| `bfs_module.jl` | Standalone Julia module (V4 compatible) |
| `test_friendster.jl` | Quick test on Friendster (CPU) |
| `test_friendster_zerocopy.jl` | Zero-Copy GPU test for large graphs |
| `README.md` | This file |

## 🚀 Quick Start

1. Upload `setup_julia_colab.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Run first two cells to install Julia
3. **Restart runtime** → Change kernel to **Julia**
4. Enable GPU: *Runtime → Change runtime type → GPU*
5. Run remaining cells

## 📊 Algorithms (V4 Features)

### BFS (Breadth-First Search)
- **Standard BFS**: Level-synchronous parallel traversal
- **Zero-Copy BFS**: Streams graph from RAM to GPU over PCIe
- **Compressed BFS**: Delta+Varint decoding (30-40% bandwidth reduction)

### Afforest (Connected Components)
- Union-Find with random neighbor sampling
- **Single-Pass Mode**: Matches V4 CUDA kernel (faster for large graphs)
- **GCC Pruning**: Skip edges within Giant Connected Component
- GPU: Atomic min for concurrent union operations

### Delta Compression (c-CSR)
```julia
# Compress a graph for bandwidth efficiency
c_graph = compress_graph(graph)

# Use compressed BFS
distances = bfs_compressed_cpu(c_graph, source)
```

## 📂 Supported Graph Formats

| Format | Function | Description |
|--------|----------|-------------|
| `.mat` | `load_graph_mat()` | MATLAB HDF5 (Friendster, Mawi) |
| `.txt` | `load_graph_text()` | Edge list: `src dst` per line |
| `.csrbin` | `load_graph_csrbin()` | Binary CSR format |

### MAT File Structure
```
/Problem/A/
  ├── ir  → column indices (col_idx)
  └── jc  → row pointers (row_ptr)
```

## 📈 Expected Performance (Matched to V4 CUDA)

| Algorithm | Graph | Runtime | Notes |
|-----------|-------|---------|-------|
| BFS (Zero-Copy) | Friendster | ~8-12s | GPU streaming from RAM |
| BFS (Compressed) | Friendster | ~5-8s | 30% bandwidth reduction |
| Afforest (CPU) | Friendster | ~20-25s | Single-pass |
| Afforest (Compressed) | Friendster | ~18-22s | With GCC pruning |

## 🔧 V4 Feature Mapping

| CUDA V4 Feature | Julia Implementation |
|-----------------|---------------------|
| `afforest_init_kernel` | `init_parent_kernel!` |
| `afforest_link_pruned_kernel` | `afforest_compressed_cpu` (single-pass) |
| `afforest_compress_kernel` | `compress_kernel!` |
| `bfsCompressedWarpKernel` | `bfs_compressed_cpu` |
| `compressGraph()` | `compress_graph()` |
| Varint encoding | `encode_varint()` / `decode_varint()` |

## 📚 References

- [CUDA.jl](https://cuda.juliagpu.org/stable/)
- [Afforest Paper](https://dl.acm.org/doi/10.1145/3178487.3178495)
- V4 CUDA Source: `src/v4_adaptive/`
