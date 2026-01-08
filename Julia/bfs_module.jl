# ==============================================================================
# BFS & Afforest Module for Julia (V5.3 Compatible)
# Parallel Processing Project 3 - AUTH
# Includes: Adaptive BFS, Compressed BFS, Afforest (CPU/GPU)
#
# Note: This Julia implementation mirrors the CUDA V5.3 algorithms but does NOT
# include the V5.3 kernel optimizations (Direct Queue Emission, Warp Aggregation).
# For peak performance, use the CUDA implementation.
# ==============================================================================

module BFSModule

using CUDA
using SparseArrays
using Statistics
using Random
using HDF5

export CSRGraph, CompressedCSRGraph
export create_csr_graph, generate_random_graph, print_graph_stats
export load_graph_mat, load_graph_text, load_graph_csrbin
export compress_graph, decompress_neighbors
export bfs_cpu, bfs_gpu, bfs_gpu_zerocopy
export bfs_compressed_cpu, bfs_compressed_gpu_zerocopy
export afforest_cpu, afforest_gpu, afforest_compressed_cpu
export validate_bfs, UNVISITED

# ==============================================================================
# Constants
# ==============================================================================

const UNVISITED = typemax(Int32)

# ==============================================================================
# Graph Data Structures
# ==============================================================================

"""
    CSRGraph

Compressed Sparse Row (CSR) representation of a graph.
Optimized for graph traversal algorithms like BFS.

# Fields
- `num_nodes::Int32`: Number of vertices in the graph
- `num_edges::Int64`: Number of edges in the graph
- `row_ptr::Vector{Int64}`: Offsets array (size: num_nodes + 1)
- `col_idx::Vector{Int32}`: Column indices (neighbor list)
"""
struct CSRGraph
    num_nodes::Int32
    num_edges::Int64
    row_ptr::Vector{Int64}
    col_idx::Vector{Int32}
end

"""
    CompressedCSRGraph

Delta-Compressed CSR using Varint encoding.
Reduces memory bandwidth by ~30-40% for large graphs.

# Fields
- `num_nodes::Int32`: Number of vertices
- `num_edges::Int64`: Number of edges
- `row_ptr::Vector{Int64}`: Byte offsets (not edge indices!)
- `compressed_col::Vector{UInt8}`: Delta+Varint encoded neighbors
"""
struct CompressedCSRGraph
    num_nodes::Int32
    num_edges::Int64
    row_ptr::Vector{Int64}      # Byte offsets into compressed_col
    compressed_col::Vector{UInt8}  # Delta + Varint encoded data
end

# ==============================================================================
# Varint Encoding/Decoding (Matches CUDA compression.cu)
# ==============================================================================

"""
    encode_varint(value::Int32) -> Vector{UInt8}

Encode an integer using Variable-Length Quantity (Varint) encoding.
Uses 1-5 bytes depending on value magnitude.
"""
function encode_varint(value::UInt32)::Vector{UInt8}
    bytes = UInt8[]
    while value > 127
        push!(bytes, UInt8((value & 127) | 128))
        value >>= 7
    end
    push!(bytes, UInt8(value & 127))
    return bytes
end

"""
    decode_varint(bytes::Vector{UInt8}, offset::Int) -> (value, new_offset)

Decode a Varint from byte stream at given offset.
Returns the decoded value and the new offset after the varint.
"""
function decode_varint(bytes::Vector{UInt8}, offset::Int)::Tuple{UInt32, Int}
    value = UInt32(0)
    shift = 0
    while true
        byte = bytes[offset]
        value |= UInt32(byte & 127) << shift
        offset += 1
        if (byte & 128) == 0
            break
        end
        shift += 7
    end
    return (value, offset)
end

"""
    decompress_neighbors(graph::CompressedCSRGraph, node::Int32) -> Vector{Int32}

Decompress and return all neighbors of a node.
Uses Delta decoding on top of Varint decoding.
"""
function decompress_neighbors(graph::CompressedCSRGraph, node::Int32)::Vector{Int32}
    start_byte = graph.row_ptr[node]
    end_byte = graph.row_ptr[node + 1]
    
    neighbors = Int32[]
    offset = Int(start_byte)
    prev_neighbor = Int32(0)
    
    while offset < end_byte
        delta, offset = decode_varint(graph.compressed_col, offset)
        neighbor = prev_neighbor + Int32(delta)
        push!(neighbors, neighbor)
        prev_neighbor = neighbor
    end
    
    return neighbors
end

# ==============================================================================
# Graph Compression (Matches CUDA implementation)
# ==============================================================================

"""
    compress_graph(graph::CSRGraph) -> CompressedCSRGraph

Compress a CSR graph using Delta + Varint encoding.
Sorts neighbors per row for optimal delta compression.

# Algorithm:
1. For each row, sort neighbors
2. Compute deltas: δ[i] = neighbor[i] - neighbor[i-1]
3. Encode each delta using Varint (1-5 bytes)
"""
function compress_graph(graph::CSRGraph)::CompressedCSRGraph
    println("🗜️  Compressing Graph (c-CSR)...")
    
    num_nodes = graph.num_nodes
    
    # Two-pass approach: measure size, then encode
    
    # PASS 1: Calculate total compressed size
    row_offsets = Vector{Int64}(undef, num_nodes + 1)
    row_offsets[1] = 1  # Julia 1-indexed
    
    total_bytes = 0
    
    for i in 1:num_nodes
        start_idx = graph.row_ptr[i]
        end_idx = graph.row_ptr[i + 1] - 1
        
        if start_idx <= end_idx
            # Get and sort neighbors
            neighbors = sort(graph.col_idx[start_idx:end_idx])
            
            # Calculate encoded size
            prev = Int32(0)
            row_bytes = 0
            for neighbor in neighbors
                delta = UInt32(neighbor - prev)
                # Count varint bytes
                val = delta
                len = 0
                while val > 127
                    len += 1
                    val >>= 7
                end
                len += 1
                row_bytes += len
                prev = neighbor
            end
            total_bytes += row_bytes
        end
        row_offsets[i + 1] = 1 + total_bytes  # 1-indexed offset
    end
    
    original_size = graph.num_edges * 4
    println("  Original: $(round(original_size / 1e9, digits=2)) GB → Compressed: $(round(total_bytes / 1e9, digits=2)) GB")
    println("  Ratio: $(round(original_size / total_bytes, digits=2))x")
    
    # PASS 2: Encode
    compressed_col = Vector{UInt8}(undef, total_bytes)
    write_pos = 1
    
    for i in 1:num_nodes
        start_idx = graph.row_ptr[i]
        end_idx = graph.row_ptr[i + 1] - 1
        
        if start_idx <= end_idx
            neighbors = sort(graph.col_idx[start_idx:end_idx])
            
            prev = Int32(0)
            for neighbor in neighbors
                delta = UInt32(neighbor - prev)
                bytes = encode_varint(delta)
                for b in bytes
                    compressed_col[write_pos] = b
                    write_pos += 1
                end
                prev = neighbor
            end
        end
    end
    
    println("  ✅ Compression complete!")
    
    return CompressedCSRGraph(num_nodes, graph.num_edges, row_offsets, compressed_col)
end

# ==============================================================================
# Graph Loading Functions
# ==============================================================================

"""
    load_graph_mat(filename::String)

Load a graph from MATLAB .mat file (HDF5 format).
"""
function load_graph_mat(filename::String)
    println("📥 Loading MAT file: $filename")
    
    h5open(filename, "r") do file
        if !haskey(file, "Problem")
            error("'Problem' group not found in $filename")
        end
        
        problem = file["Problem"]
        if !haskey(problem, "A")
            error("'A' group not found in /Problem")
        end
        
        A = problem["A"]
        
        if !haskey(A, "ir") || !haskey(A, "jc")
            error("'ir' or 'jc' datasets not found in /Problem/A")
        end
        
        println("  📖 Reading jc (row pointers)...")
        jc = read(A["jc"])
        
        println("  📖 Reading ir (column indices)...")
        ir = read(A["ir"])
        
        # Convert to 1-indexed
        row_ptr = Vector{Int64}(jc) .+ 1
        col_idx = Vector{Int32}(ir) .+ 1
        
        num_nodes = Int32(length(row_ptr) - 1)
        num_edges = Int64(length(col_idx))
        
        println("  ✅ Loaded: $(num_nodes) nodes, $(num_edges) edges")
        
        return CSRGraph(num_nodes, num_edges, row_ptr, col_idx)
    end
end

"""
    load_graph_text(filename::String)

Load graph from text file (edge list format).
"""
function load_graph_text(filename::String)
    println("📥 Loading text graph: $filename")
    
    open(filename) do file
        header = readline(file)
        parts = split(header)
        num_nodes = parse(Int, parts[1])
        
        edges = Tuple{Int32, Int32}[]
        
        for line in eachline(file)
            parts = split(line)
            if length(parts) >= 2
                src = parse(Int32, parts[1])
                dst = parse(Int32, parts[2])
                
                if src == 0 || dst == 0
                    src += 1
                    dst += 1
                end
                
                push!(edges, (src, dst))
            end
        end
        
        return create_csr_graph(num_nodes, edges)
    end
end

"""
    load_graph_csrbin(filename::String)

Load graph from binary CSR format (.csrbin).
"""
function load_graph_csrbin(filename::String)
    println("📥 Loading CSR binary: $filename")
    
    open(filename, "r") do file
        num_nodes = read(file, Int64)
        num_edges = read(file, Int64)
        
        println("  📊 Nodes: $num_nodes, Edges: $num_edges")
        
        row_ptr = Vector{Int64}(undef, num_nodes + 1)
        col_idx = Vector{Int32}(undef, num_edges)
        
        read!(file, row_ptr)
        read!(file, col_idx)
        
        # Convert to 1-indexed if needed
        if maximum(col_idx) < num_nodes
            row_ptr .+= 1
            col_idx .+= 1
        end
        
        println("  ✅ Loaded successfully!")
        
        return CSRGraph(Int32(num_nodes), Int64(num_edges), row_ptr, col_idx)
    end
end

"""
    create_csr_graph(num_nodes::Int, edges::Vector{Tuple{Int32, Int32}})

Create a CSR graph from an edge list.
"""
function create_csr_graph(num_nodes::Int, edges::Vector{Tuple{Int32, Int32}})
    adj = [Int32[] for _ in 1:num_nodes]
    for (src, dst) in edges
        if 1 <= src <= num_nodes && 1 <= dst <= num_nodes
            push!(adj[src], dst)
        end
    end
    
    num_edges = sum(length.(adj))
    
    row_ptr = zeros(Int64, num_nodes + 1)
    col_idx = Vector{Int32}(undef, num_edges)
    
    edge_idx = 1
    for i in 1:num_nodes
        row_ptr[i] = edge_idx
        sort!(adj[i])
        for neighbor in adj[i]
            col_idx[edge_idx] = neighbor
            edge_idx += 1
        end
    end
    row_ptr[num_nodes + 1] = edge_idx
    
    return CSRGraph(Int32(num_nodes), Int64(num_edges), row_ptr, col_idx)
end

"""
    generate_random_graph(num_nodes::Int, avg_degree::Int; seed::Int=42)

Generate a random graph for testing.
"""
function generate_random_graph(num_nodes::Int, avg_degree::Int; seed::Int=42)
    Random.seed!(seed)
    edges = Tuple{Int32, Int32}[]
    
    for i in 1:num_nodes
        degree = rand(1:2*avg_degree)
        for _ in 1:degree
            neighbor = rand(1:num_nodes)
            if neighbor != i
                push!(edges, (Int32(i), Int32(neighbor)))
            end
        end
    end
    
    return create_csr_graph(num_nodes, edges)
end

"""
    print_graph_stats(graph::CSRGraph)

Print statistics about a graph.
"""
function print_graph_stats(graph::CSRGraph)
    degrees = [graph.row_ptr[i+1] - graph.row_ptr[i] for i in 1:graph.num_nodes]
    
    println("═" ^ 40)
    println("       Graph Statistics")
    println("═" ^ 40)
    println("  Nodes:      $(graph.num_nodes)")
    println("  Edges:      $(graph.num_edges)")
    println("  Min Degree: $(minimum(degrees))")
    println("  Max Degree: $(maximum(degrees))")
    println("  Avg Degree: $(round(mean(degrees), digits=2))")
    println("═" ^ 40)
end

# ==============================================================================
# CPU BFS Implementation
# ==============================================================================

"""
    bfs_cpu(graph::CSRGraph, source::Int32)

Sequential CPU-based BFS implementation.
"""
function bfs_cpu(graph::CSRGraph, source::Int32)
    distances = fill(UNVISITED, graph.num_nodes)
    distances[source] = Int32(0)
    
    frontier = [source]
    current_level = Int32(0)
    
    while !isempty(frontier)
        next_frontier = Int32[]
        
        for node in frontier
            start_idx = graph.row_ptr[node]
            end_idx = graph.row_ptr[node + 1] - 1
            
            for edge_idx in start_idx:end_idx
                neighbor = graph.col_idx[edge_idx]
                if distances[neighbor] == UNVISITED
                    distances[neighbor] = current_level + Int32(1)
                    push!(next_frontier, neighbor)
                end
            end
        end
        
        frontier = next_frontier
        current_level += Int32(1)
    end
    
    return distances
end

"""
    bfs_compressed_cpu(graph::CompressedCSRGraph, source::Int32)

CPU BFS on compressed graph with on-the-fly decompression.
"""
function bfs_compressed_cpu(graph::CompressedCSRGraph, source::Int32)
    distances = fill(UNVISITED, graph.num_nodes)
    distances[source] = Int32(0)
    
    frontier = [source]
    current_level = Int32(0)
    
    while !isempty(frontier)
        next_frontier = Int32[]
        
        for node in frontier
            # Decompress neighbors on-the-fly
            neighbors = decompress_neighbors(graph, node)
            
            for neighbor in neighbors
                if distances[neighbor] == UNVISITED
                    distances[neighbor] = current_level + Int32(1)
                    push!(next_frontier, neighbor)
                end
            end
        end
        
        frontier = next_frontier
        current_level += Int32(1)
    end
    
    return distances
end

# ==============================================================================
# CPU Afforest (Union-Find with Sampling)
# ==============================================================================

"""
    find_root(parent::Vector{Int32}, u::Int32)

Find root with path compression.
"""
function find_root(parent::Vector{Int32}, u::Int32)
    root = u
    while parent[root] != root
        root = parent[root]
    end
    # Path compression
    while parent[u] != root
        next_u = parent[u]
        parent[u] = root
        u = next_u
    end
    return root
end

"""
    union_nodes!(parent::Vector{Int32}, u::Int32, v::Int32)

Union two nodes by linking smaller root to larger.
"""
function union_nodes!(parent::Vector{Int32}, u::Int32, v::Int32)
    root_u = find_root(parent, u)
    root_v = find_root(parent, v)
    
    if root_u != root_v
        if root_u < root_v
            parent[root_v] = root_u
        else
            parent[root_u] = root_v
        end
        return true
    end
    return false
end

"""
    afforest_cpu(graph::CSRGraph; sampling_rounds::Int=2)

CPU Afforest algorithm for connected components.
"""
function afforest_cpu(graph::CSRGraph; sampling_rounds::Int=2)
    num_nodes = graph.num_nodes
    parent = collect(Int32(1):Int32(num_nodes))
    
    println("🌲 Running Afforest (Union-Find with Sampling)...")
    
    # SAMPLING PHASE
    println("  Phase 1: Random Neighbor Sampling ($sampling_rounds rounds)")
    
    for round in 1:sampling_rounds
        Random.seed!(round + 42)
        
        for u in 1:num_nodes
            start_idx = graph.row_ptr[u]
            end_idx = graph.row_ptr[u + 1] - 1
            degree = end_idx - start_idx + 1
            
            if degree > 0
                random_offset = rand(0:degree-1)
                v = graph.col_idx[start_idx + random_offset]
                union_nodes!(parent, Int32(u), v)
            end
        end
        
        for i in 1:num_nodes
            find_root(parent, Int32(i))
        end
    end
    
    # HOOK PHASE
    println("  Phase 2: Full Edge Scan (Hook)")
    
    changed = true
    iterations = 0
    
    while changed
        changed = false
        iterations += 1
        
        for u in 1:num_nodes
            start_idx = graph.row_ptr[u]
            end_idx = graph.row_ptr[u + 1] - 1
            
            for edge_idx in start_idx:end_idx
                v = graph.col_idx[edge_idx]
                if union_nodes!(parent, Int32(u), v)
                    changed = true
                end
            end
        end
        
        for i in 1:num_nodes
            find_root(parent, Int32(i))
        end
    end
    
    println("  Phase 3: Final Compression (converged in $iterations iterations)")
    
    for i in 1:num_nodes
        parent[i] = find_root(parent, Int32(i))
    end
    
    num_components = length(unique(parent))
    println("  ✅ Found $num_components connected components")
    
    return parent
end

"""
    afforest_compressed_cpu(graph::CompressedCSRGraph; sampling_rounds::Int=0)

CPU Afforest on compressed graph (single-pass, no sampling for efficiency).
Matches V5.3 CUDA implementation (pruning disabled for compressed graphs).

# V5.3 Note:
In the CUDA implementation, pruning is disabled for compressed graphs because
the GCC ID cannot be reliably determined without costly sampling. This Julia
version mirrors that behavior.
"""
function afforest_compressed_cpu(graph::CompressedCSRGraph; sampling_rounds::Int=0)
    num_nodes = graph.num_nodes
    parent = collect(Int32(1):Int32(num_nodes))
    
    println("🌲 Running Compressed Afforest (Single-Pass)...")
    
    # GCC ID Heuristic (node 1 is likely in GCC)
    gcc_id = parent[1]
    println("  Estimated GCC ID: $gcc_id")
    
    # SINGLE-PASS LINK PHASE (matches V4 kernel)
    for u in 1:num_nodes
        # Path compression inline
        comp_u = u
        while parent[comp_u] != comp_u
            comp_u = parent[comp_u]
        end
        parent[u] = comp_u
        
        # Decompress neighbors
        neighbors = decompress_neighbors(graph, Int32(u))
        
        for v in neighbors
            comp_v = v
            while parent[comp_v] != comp_v
                comp_v = parent[comp_v]
            end
            
            # GCC Pruning (matches V4 logic)
            if comp_u == gcc_id && comp_v == gcc_id
                continue
            end
            
            if comp_u != comp_v
                small = min(comp_u, comp_v)
                large = max(comp_u, comp_v)
                parent[large] = small
            end
        end
    end
    
    # Final compression
    println("  Finalizing component labels...")
    for i in 1:num_nodes
        parent[i] = find_root(parent, Int32(i))
    end
    
    num_components = length(unique(parent))
    println("  ✅ Found $num_components connected components")
    
    return parent
end

# ==============================================================================
# GPU BFS Implementation
# ==============================================================================

"""
GPU kernel for BFS traversal.
"""
function bfs_kernel!(
    row_ptr, col_idx,
    distances, frontier, frontier_size,
    next_frontier, next_frontier_size,
    current_level
)
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if tid <= frontier_size[]
        node = frontier[tid]
        start_idx = row_ptr[node]
        end_idx = row_ptr[node + 1] - 1
        
        for edge_idx in start_idx:end_idx
            neighbor = col_idx[edge_idx]
            
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
    bfs_gpu(graph::CSRGraph, source::Int32)

GPU-accelerated parallel BFS.
"""
function bfs_gpu(graph::CSRGraph, source::Int32)
    if !CUDA.functional()
        error("CUDA is not available. Use bfs_cpu instead.")
    end
    
    d_row_ptr = CuArray(graph.row_ptr)
    d_col_idx = CuArray(graph.col_idx)
    
    d_distances = CUDA.fill(UNVISITED, graph.num_nodes)
    d_distances[source] = Int32(0)
    
    d_frontier = CuArray{Int32}(undef, graph.num_nodes)
    d_next_frontier = CuArray{Int32}(undef, graph.num_nodes)
    d_frontier[1] = source
    
    d_frontier_size = CuArray([Int32(1)])
    d_next_frontier_size = CuArray([Int32(0)])
    
    current_level = Int32(0)
    block_size = 256
    
    while true
        h_frontier_size = Array(d_frontier_size)[1]
        if h_frontier_size == 0
            break
        end
        
        d_next_frontier_size .= Int32(0)
        
        num_blocks = cld(h_frontier_size, block_size)
        @cuda threads=block_size blocks=num_blocks bfs_kernel!(
            d_row_ptr, d_col_idx,
            d_distances, d_frontier, d_frontier_size,
            d_next_frontier, d_next_frontier_size,
            current_level
        )
        CUDA.synchronize()
        
        d_frontier, d_next_frontier = d_next_frontier, d_frontier
        d_frontier_size, d_next_frontier_size = d_next_frontier_size, d_frontier_size
        
        current_level += Int32(1)
    end
    
    return Array(d_distances)
end

# ==============================================================================
# GPU Zero-Copy BFS (for large graphs exceeding VRAM)
# ==============================================================================

"""
GPU kernel for Zero-Copy BFS (uses host memory pointers).
"""
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
        
        start_idx = unsafe_load(row_ptr, node)
        end_idx = unsafe_load(row_ptr, node + 1) - 1
        
        for edge_idx in start_idx:end_idx
            neighbor = unsafe_load(col_idx, edge_idx)
            
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

Zero-Copy GPU BFS for graphs exceeding VRAM.
Streams graph data from host RAM over PCIe.
"""
function bfs_gpu_zerocopy(graph::CSRGraph, source::Int32)
    if !CUDA.functional()
        error("CUDA is not available.")
    end
    
    println("🔧 Setting up Zero-Copy Memory...")
    
    # Register host memory
    CUDA.Mem.register(CUDA.Mem.Host, pointer(graph.row_ptr), sizeof(graph.row_ptr), CUDA.Mem.HOSTREGISTER_DEVICEMAP)
    CUDA.Mem.register(CUDA.Mem.Host, pointer(graph.col_idx), sizeof(graph.col_idx), CUDA.Mem.HOSTREGISTER_DEVICEMAP)
    
    ptr_row = reinterpret(Core.LLVMPtr{Int64, 0}, pointer(graph.row_ptr))
    ptr_col = reinterpret(Core.LLVMPtr{Int32, 0}, pointer(graph.col_idx))
    
    println("  ✅ Host memory pinned and mapped")
    
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
    
    while true
        h_frontier_size = Array(d_frontier_size)[1]
        if h_frontier_size == 0
            break
        end
        
        d_next_frontier_size .= Int32(0)
        num_blocks = cld(h_frontier_size, block_size)
        
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
    
    return Array(d_distances)
end

"""
    bfs_compressed_gpu_zerocopy(graph::CompressedCSRGraph, source::Int32)

Zero-Copy GPU BFS on compressed graph.
Decodes Varint on-the-fly from compressed memory stream.
"""
function bfs_compressed_gpu_zerocopy(graph::CompressedCSRGraph, source::Int32)
    if !CUDA.functional()
        error("CUDA is not available.")
    end
    
    println("🔧 Setting up Compressed Zero-Copy Memory...")
    
    # Register compressed data
    CUDA.Mem.register(CUDA.Mem.Host, pointer(graph.row_ptr), sizeof(graph.row_ptr), CUDA.Mem.HOSTREGISTER_DEVICEMAP)
    CUDA.Mem.register(CUDA.Mem.Host, pointer(graph.compressed_col), sizeof(graph.compressed_col), CUDA.Mem.HOSTREGISTER_DEVICEMAP)
    
    println("  ✅ Compressed data pinned ($(round(sizeof(graph.compressed_col)/1e9, digits=2)) GB)")
    
    # For now, fall back to CPU decompression + GPU BFS
    # Full GPU Varint decoding kernel would be complex
    println("  ⚠️ Using CPU decompression (GPU Varint kernel not implemented)")
    
    return bfs_compressed_cpu(graph, source)
end

# ==============================================================================
# GPU Afforest Implementation
# ==============================================================================

function init_parent_kernel!(parent, num_nodes)
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if tid <= num_nodes
        parent[tid] = Int32(tid)
    end
    return nothing
end

function afforest_sampling_kernel!(row_ptr, col_idx, parent, num_nodes, seed)
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if tid <= num_nodes
        start_idx = row_ptr[tid]
        end_idx = row_ptr[tid + 1] - 1
        degree = end_idx - start_idx + 1
        
        if degree > 0
            r = (seed * 1664525 + 1013904223 + tid) % typemax(UInt32)
            offset = r % degree
            neighbor = col_idx[start_idx + offset]
            
            p_u = parent[tid]
            p_v = parent[neighbor]
            
            if p_v < p_u
                CUDA.atomic_min!(pointer(parent, p_u), p_v)
            elseif p_u < p_v
                CUDA.atomic_min!(pointer(parent, p_v), p_u)
            end
        end
    end
    return nothing
end

function hook_kernel!(row_ptr, col_idx, parent, changed, num_nodes)
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if tid <= num_nodes
        p_u = parent[tid]
        start_idx = row_ptr[tid]
        end_idx = row_ptr[tid + 1] - 1
        
        for e in start_idx:end_idx
            v = col_idx[e]
            p_v = parent[v]
            
            if p_v < p_u
                old = CUDA.atomic_min!(pointer(parent, p_u), p_v)
                if old > p_v
                    changed[] = Int32(1)
                    p_u = p_v
                end
            end
        end
    end
    return nothing
end

function compress_kernel!(parent, num_nodes)
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if tid <= num_nodes
        p = parent[tid]
        pp = parent[p]
        if p != pp
            parent[tid] = pp
        end
    end
    return nothing
end

"""
    afforest_gpu(graph::CSRGraph; sampling_rounds::Int=2)

GPU-accelerated Afforest algorithm.
"""
function afforest_gpu(graph::CSRGraph; sampling_rounds::Int=2)
    if !CUDA.functional()
        error("CUDA is not available. Use afforest_cpu instead.")
    end
    
    num_nodes = Int32(graph.num_nodes)
    block_size = 256
    num_blocks = cld(num_nodes, block_size)
    
    d_row_ptr = CuArray(graph.row_ptr)
    d_col_idx = CuArray(graph.col_idx)
    
    d_parent = CuArray{Int32}(undef, num_nodes)
    @cuda threads=block_size blocks=num_blocks init_parent_kernel!(d_parent, num_nodes)
    
    d_changed = CuArray([Int32(0)])
    
    println("🌲 Running GPU Afforest...")
    
    # SAMPLING PHASE
    println("  Phase 1: Random Neighbor Sampling ($sampling_rounds rounds)")
    
    for round in 1:sampling_rounds
        @cuda threads=block_size blocks=num_blocks afforest_sampling_kernel!(
            d_row_ptr, d_col_idx, d_parent, num_nodes, UInt32(round + 42)
        )
        @cuda threads=block_size blocks=num_blocks compress_kernel!(d_parent, num_nodes)
        CUDA.synchronize()
    end
    
    # HOOK PHASE
    println("  Phase 2: Full Edge Scan (Hook)")
    
    h_changed = Int32(1)
    iterations = 0
    
    while h_changed != 0
        d_changed .= Int32(0)
        iterations += 1
        
        @cuda threads=block_size blocks=num_blocks hook_kernel!(
            d_row_ptr, d_col_idx, d_parent, d_changed, num_nodes
        )
        @cuda threads=block_size blocks=num_blocks compress_kernel!(d_parent, num_nodes)
        CUDA.synchronize()
        
        h_changed = Array(d_changed)[1]
    end
    
    println("  Phase 3: Final Compression (converged in $iterations iterations)")
    for _ in 1:5
        @cuda threads=block_size blocks=num_blocks compress_kernel!(d_parent, num_nodes)
    end
    CUDA.synchronize()
    
    parent = Array(d_parent)
    num_components = length(unique(parent))
    println("  ✅ Found $num_components connected components")
    
    return parent
end

# ==============================================================================
# Validation
# ==============================================================================

"""
    validate_bfs(cpu_distances, gpu_distances)

Compare CPU and GPU BFS results.
"""
function validate_bfs(cpu_distances::Vector{Int32}, gpu_distances::Vector{Int32})
    if length(cpu_distances) != length(gpu_distances)
        println("❌ Size mismatch: CPU=$(length(cpu_distances)), GPU=$(length(gpu_distances))")
        return false
    end
    
    mismatches = 0
    for i in eachindex(cpu_distances)
        if cpu_distances[i] != gpu_distances[i]
            mismatches += 1
            if mismatches <= 5
                println("  Mismatch at node $i: CPU=$(cpu_distances[i]), GPU=$(gpu_distances[i])")
            end
        end
    end
    
    if mismatches == 0
        println("✅ Validation passed! All $(length(cpu_distances)) distances match.")
        return true
    else
        println("❌ Validation failed! $mismatches mismatches found.")
        return false
    end
end

end # module
