# Julia Implementation for Google Colab

Parallel BFS & Afforest implementation in Julia with CUDA.jl for GPU acceleration.

## 📁 Files

| File | Description |
|------|-------------|
| `setup_julia_colab.ipynb` | Main notebook with BFS & Afforest (CPU/GPU) |
| `bfs_module.jl` | Standalone Julia module |
| `README.md` | This file |

## 🚀 Quick Start

1. Upload `setup_julia_colab.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Run first two cells to install Julia
3. **Restart runtime** → Change kernel to **Julia**
4. Enable GPU: *Runtime → Change runtime type → GPU*
5. Run remaining cells

## 📊 Algorithms

### BFS (Breadth-First Search)
- Level-synchronous parallel traversal
- GPU: Atomic frontier expansion with warp cooperation

### Afforest (Connected Components)
- Union-Find with random neighbor sampling
- 2-phase: Sampling (fast convergence) → Hook (full scan)
- GPU: Atomic min for concurrent union operations

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

## 📈 Expected Performance

| Graph Size | BFS Speedup | Afforest Speedup |
|------------|-------------|------------------|
| 10K nodes  | ~3-5x       | ~2-4x            |
| 100K nodes | ~10-15x     | ~8-12x           |
| 1M nodes   | ~20-30x     | ~15-25x          |

## 📚 References

- [CUDA.jl](https://cuda.juliagpu.org/stable/)
- [Afforest Paper](https://dl.acm.org/doi/10.1145/3178487.3178495)
