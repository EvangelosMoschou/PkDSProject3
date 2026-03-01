#!/usr/bin/env julia
# Zero-Copy BFS Test for Large Graphs (Friendster)
# This uses "Pinned Memory" to stream data from RAM to GPU over PCIe

push!(LOAD_PATH, @__DIR__)
using Pkg
for pkg in ["CUDA", "HDF5", "SparseArrays"]
    try
        @eval using $(Symbol(pkg))
    catch
        Pkg.add(pkg)
        @eval using $(Symbol(pkg))
    end
end

include("bfs_module.jl")
using .BFSModule

# ==============================================================================
# Zero-Copy Kernel
# ==============================================================================

# Redefine kernel to accept raw pointers (DevicePtr) instead of arrays
# This allows us to pass a pointer to host memory
function bfs_zerocopy_kernel!(
    row_ptr::Core.LLVMPtr{Int64, 0}, 
    col_idx::Core.LLVMPtr{Int32, 0},
    distances, frontier, frontier_size,
    next_frontier, next_frontier_size,
    current_level::Int32
)
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if tid <= frontier_size[]
        node = frontier[tid]
        
        # Access host memory directly via pointer arithmetic
        # Note: unsafe_load is slow if not coalesced, but functional for this test
        start_idx = unsafe_load(row_ptr, node)
        end_idx = unsafe_load(row_ptr, node + 1) - 1
        
        # Iterate neighbors
        # For zero-copy, sequential access is better than random
        for edge_idx in start_idx:end_idx
            neighbor = unsafe_load(col_idx, edge_idx)
            
            # Atomic compare-and-swap
            old_val = CUDA.atomic_cas!(
                pointer(distances, neighbor), 
                UNVISITED, 
                current_level + Int32(1)
            )
            
            if old_val == UNVISITED
                idx = CUDA.atomic_add!(pointer(next_frontier_size, 1), Int32(1)) + 1
                next_frontier[idx] = neighbor
            end
        end
    end
    return nothing
end

"""
    bfs_gpu_zerocopy(graph::CSRGraph, source::Int32)

Runs BFS using Zero-Copy memory for the graph structure.
"""
function bfs_gpu_zerocopy(graph, source::Int32)
    println("🔧 Setting up Zero-Copy Memory...")
    
    # 1. Register Host Memory (Pinning)
    # This tells the OS/CUDA driver not to swap these pages, allowing GPU to access them
    CUDA.Mem.register(CUDA.Mem.Host, pointer(graph.row_ptr), sizeof(graph.row_ptr), CUDA.Mem.HOSTREGISTER_DEVICEMAP)
    CUDA.Mem.register(CUDA.Mem.Host, pointer(graph.col_idx), sizeof(graph.col_idx), CUDA.Mem.HOSTREGISTER_DEVICEMAP)
    
    # 2. Get Device Pointers
    # These pointers are valid on the GPU but point to Host RAM
    ptr_row = reinterpret(Core.LLVMPtr{Int64, 0}, pointer(graph.row_ptr))
    ptr_col = reinterpret(Core.LLVMPtr{Int32, 0}, pointer(graph.col_idx))
    
    println("  ✅ Host memory pinned and mapped")
    
    # Initialize distances and frontiers on VRAM (these are small enough)
    println("📦 Allocating work buffers on VRAM...")
    d_distances = CUDA.fill(UNVISITED, graph.num_nodes)
    
    d_frontier = CuArray{Int32}(undef, graph.num_nodes)
    d_next_frontier = CuArray{Int32}(undef, graph.num_nodes)

    CUDA.@allowscalar begin
        d_distances[source] = Int32(0)
        d_frontier[1] = source
    end
    
    d_frontier_size = CuArray([Int32(1)])
    d_next_frontier_size = CuArray([Int32(0)])
    
    current_level = Int32(0)
    block_size = 256
    
    println("🚀 Starting Zero-Copy BFS...")
    total_time = 0.0
    
    while true
        h_frontier_size = Array(d_frontier_size)[1]
        if h_frontier_size == 0
            break
        end
        
        d_next_frontier_size .= Int32(0)
        num_blocks = cld(h_frontier_size, block_size)
        
        # Launch kernel with pointers instead of arrays
        @cuda threads=block_size blocks=num_blocks bfs_zerocopy_kernel!(
            ptr_row, ptr_col,
            d_distances, d_frontier, d_frontier_size,
            d_next_frontier, d_next_frontier_size,
            current_level
        )
        CUDA.synchronize()
        
        d_frontier, d_next_frontier = d_next_frontier, d_frontier
        d_frontier_size, d_next_frontier_size = d_next_frontier_size, d_frontier_size
        current_level += Int32(1)
    end
    
    # Unregister (cleanup) - Skipped to avoid MethodError, OS will reclaim
    # CUDA.Mem.unregister(pointer(graph.row_ptr))
    # CUDA.Mem.unregister(pointer(graph.col_idx))
    
    return Array(d_distances)
end

# ==============================================================================
# Main execution
# ==============================================================================

println("=" ^ 50)
println("Julia Zero-Copy BFS Test (RAM -> GPU)")
println("=" ^ 50)

graph_path = "../Mat Files/com-Friendster.mat.csrbin"

if !isfile(graph_path)
    println("❌ Graph file not found!")
    exit(1)
end

println("\n📥 Loading graph: $graph_path")
# Force CPU load
graph = load_graph_csrbin(graph_path)
print_graph_stats(graph)

if CUDA.functional()
    println("\n✅ CUDA detected: ", CUDA.name(CUDA.device()))
    
    # Run Zero-Copy BFS
    @time distances = bfs_gpu_zerocopy(graph, Int32(1))
    
    # Stats
    reachable = count(d -> d != UNVISITED, distances)
    max_dist = maximum(d for d in distances if d != UNVISITED)
    
    println("\n=== BFS Result (Zero-Copy GPU) ===")
    println("Source: 1")
    println("Reachable Nodes: $reachable / $(graph.num_nodes) ($(round(100*reachable/graph.num_nodes, digits=2))%)")
    println("Max Distance (Diameter): $max_dist")
    println("==================")
else
    println("❌ CUDA not available.")
end
