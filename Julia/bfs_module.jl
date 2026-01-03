# ==============================================================================
# BFS & Afforest Module for Julia
# Parallel Processing Project 3 - AUTH
# ==============================================================================

module BFSModule

using CUDA
using SparseArrays
using Statistics
using Random
using HDF5

export CSRGraph, create_csr_graph, generate_random_graph, print_graph_stats
export load_graph_mat, load_graph_text, load_graph_csrbin
export bfs_cpu, bfs_gpu, afforest_cpu, afforest_gpu
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

# ==============================================================================
# Graph Loading Functions
# ==============================================================================

"""
    load_graph_mat(filename::String)

Load a graph from MATLAB .mat file (HDF5 format).
Expects structure: /Problem/A/ir (col indices) and /Problem/A/jc (row pointers)

This is the format used by:
- com-Friendster.mat
- mawi_201512020330.mat

# Arguments
- `filename`: Path to the .mat file

# Returns
- `CSRGraph`: The loaded CSR graph
"""
function load_graph_mat(filename::String)
    println("📥 Loading MAT file: $filename")
    
    h5open(filename, "r") do file
        # Navigate to /Problem/A
        if !haskey(file, "Problem")
            error("'Problem' group not found in $filename")
        end
        
        problem = file["Problem"]
        if !haskey(problem, "A")
            error("'A' group not found in /Problem")
        end
        
        A = problem["A"]
        
        # Read ir (column indices) and jc (row pointers)
        # In MATLAB sparse format: ir = row indices, jc = column pointers
        # For symmetric graphs treated as CSR: jc -> row_ptr, ir -> col_idx
        
        if !haskey(A, "ir") || !haskey(A, "jc")
            error("'ir' or 'jc' datasets not found in /Problem/A")
        end
        
        println("  📖 Reading jc (row pointers)...")
        jc = read(A["jc"])  # Column pointers (becomes row_ptr for CSR)
        
        println("  📖 Reading ir (column indices)...")
        ir = read(A["ir"])  # Row indices (becomes col_idx for CSR)
        
        # Convert to proper types
        # Note: MATLAB uses 0-indexed, Julia uses 1-indexed
        row_ptr = Vector{Int64}(jc) .+ 1  # Convert to 1-indexed
        col_idx = Vector{Int32}(ir) .+ 1  # Convert to 1-indexed
        
        num_nodes = Int32(length(row_ptr) - 1)
        num_edges = Int64(length(col_idx))
        
        println("  ✅ Loaded: $(num_nodes) nodes, $(num_edges) edges")
        
        return CSRGraph(num_nodes, num_edges, row_ptr, col_idx)
    end
end

"""
    load_graph_text(filename::String)

Load graph from text file format:
First line: num_nodes num_edges
Following lines: src dst (0-indexed or 1-indexed based on content)

# Arguments
- `filename`: Path to the text file

# Returns
- `CSRGraph`: The loaded CSR graph
"""
function load_graph_text(filename::String)
    println("📥 Loading text graph: $filename")
    
    open(filename) do file
        header = readline(file)
        parts = split(header)
        num_nodes = parse(Int, parts[1])
        num_edges_declared = parse(Int, parts[2])
        
        edges = Tuple{Int32, Int32}[]
        
        for line in eachline(file)
            parts = split(line)
            if length(parts) >= 2
                src = parse(Int32, parts[1])
                dst = parse(Int32, parts[2])
                
                # Auto-detect 0-indexed vs 1-indexed
                # If any edge has node 0, assume 0-indexed
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
Format: [num_nodes:Int64][num_edges:Int64][row_ptr:Int64×(n+1)][col_idx:Int32×m]

# Arguments
- `filename`: Path to the .csrbin file

# Returns
- `CSRGraph`: The loaded CSR graph
"""
function load_graph_csrbin(filename::String)
    println("📥 Loading CSR binary: $filename")
    
    open(filename, "r") do file
        # Read header
        num_nodes = read(file, Int64)
        num_edges = read(file, Int64)
        
        println("  📊 Nodes: $num_nodes, Edges: $num_edges")
        
        # Read arrays
        row_ptr = Vector{Int64}(undef, num_nodes + 1)
        col_idx = Vector{Int32}(undef, num_edges)
        
        read!(file, row_ptr)
        read!(file, col_idx)
        
        # Convert to 1-indexed if needed (check if max col_idx < num_nodes)
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

# Arguments
- `num_nodes`: Number of vertices
- `edges`: Vector of (source, destination) tuples (1-indexed)

# Returns
- `CSRGraph`: The constructed CSR graph
"""
function create_csr_graph(num_nodes::Int, edges::Vector{Tuple{Int32, Int32}})
    # Build adjacency list
    adj = [Int32[] for _ in 1:num_nodes]
    for (src, dst) in edges
        if 1 <= src <= num_nodes && 1 <= dst <= num_nodes
            push!(adj[src], dst)
        end
    end
    
    # Count total edges
    num_edges = sum(length.(adj))
    
    # Allocate CSR arrays
    row_ptr = zeros(Int64, num_nodes + 1)
    col_idx = Vector{Int32}(undef, num_edges)
    
    # Fill CSR structure
    edge_idx = 1
    for i in 1:num_nodes
        row_ptr[i] = edge_idx
        sort!(adj[i])  # Sort for cache locality
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

Generate a random graph for testing purposes.

# Arguments
- `num_nodes`: Number of vertices
- `avg_degree`: Average degree (edges per node)
- `seed`: Random seed for reproducibility

# Returns
- `CSRGraph`: A randomly generated CSR graph
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
Used as a reference for correctness validation.

# Arguments
- `graph`: CSRGraph to traverse
- `source`: Starting vertex (1-indexed)

# Returns
- `Vector{Int32}`: Distance from source to each vertex (UNVISITED if unreachable)
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

# ==============================================================================
# CPU Afforest (Union-Find with Sampling)
# ==============================================================================

"""
    find_root(parent::Vector{Int32}, u::Int32)

Find root of node u with path compression.
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
        # Link smaller to larger (simple heuristic)
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

CPU-based Afforest algorithm for connected components.
Uses random neighbor sampling to accelerate convergence.

# Algorithm:
1. **Afforest Sampling Phase**: Each node connects to a random neighbor
   (repeated `sampling_rounds` times)
2. **Hook Phase**: Iteratively connect adjacent nodes until convergence
3. **Compress Phase**: Flatten trees for direct root access

# Arguments
- `graph`: CSRGraph to analyze
- `sampling_rounds`: Number of random sampling rounds (default: 2)

# Returns
- `Vector{Int32}`: Component ID for each node (root of component)
"""
function afforest_cpu(graph::CSRGraph; sampling_rounds::Int=2)
    num_nodes = graph.num_nodes
    parent = collect(Int32(1):Int32(num_nodes))  # Each node is its own parent
    
    println("🌲 Running Afforest (Union-Find with Sampling)...")
    
    # --- AFFOREST SAMPLING PHASE ---
    println("  Phase 1: Random Neighbor Sampling ($sampling_rounds rounds)")
    
    for round in 1:sampling_rounds
        Random.seed!(round + 42)  # Reproducible randomness
        
        for u in 1:num_nodes
            start_idx = graph.row_ptr[u]
            end_idx = graph.row_ptr[u + 1] - 1
            degree = end_idx - start_idx + 1
            
            if degree > 0
                # Pick random neighbor
                random_offset = rand(0:degree-1)
                v = graph.col_idx[start_idx + random_offset]
                
                # Union u and v
                union_nodes!(parent, Int32(u), v)
            end
        end
        
        # Compress paths after each round
        for i in 1:num_nodes
            find_root(parent, Int32(i))
        end
    end
    
    # --- HOOK PHASE ---
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
        
        # Compress paths after each iteration
        for i in 1:num_nodes
            find_root(parent, Int32(i))
        end
    end
    
    println("  Phase 3: Final Compression (converged in $iterations iterations)")
    
    # Final compression - ensure all nodes point directly to root
    for i in 1:num_nodes
        parent[i] = find_root(parent, Int32(i))
    end
    
    # Count components
    num_components = length(unique(parent))
    println("  ✅ Found $num_components connected components")
    
    return parent
end

"""
    connectivity_to_distances(parent::Vector{Int32}, source::Int32)

Convert component labels to BFS-like distances.
Nodes in source's component get distance 1, others get UNVISITED.

# Arguments
- `parent`: Component labels from afforest
- `source`: Source node

# Returns
- `Vector{Int32}`: Distance-like array (1 for connected, UNVISITED otherwise)
"""
function connectivity_to_distances(parent::Vector{Int32}, source::Int32)
    source_root = parent[source]
    distances = fill(UNVISITED, length(parent))
    
    for i in eachindex(parent)
        if parent[i] == source_root
            distances[i] = Int32(1)
        end
    end
    distances[source] = Int32(0)
    
    return distances
end

# ==============================================================================
# GPU BFS Implementation
# ==============================================================================

"""
GPU kernel for BFS traversal.
Each thread processes one node from the current frontier.
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
            
            # Atomic compare-and-swap for thread-safe update
            old_val = CUDA.atomic_cas!(
                pointer(distances, neighbor), 
                UNVISITED, 
                current_level + Int32(1)
            )
            
            if old_val == UNVISITED
                # Atomically add to next frontier
                idx = CUDA.atomic_add!(pointer(next_frontier_size, 1), Int32(1)) + 1
                next_frontier[idx] = neighbor
            end
        end
    end
    
    return nothing
end

"""
    bfs_gpu(graph::CSRGraph, source::Int32)

GPU-accelerated parallel BFS using CUDA.jl.

# Arguments
- `graph`: CSRGraph to traverse
- `source`: Starting vertex (1-indexed)

# Returns
- `Vector{Int32}`: Distance from source to each vertex

# Notes
- Requires CUDA-capable GPU
- Automatically transfers data to/from GPU
"""
function bfs_gpu(graph::CSRGraph, source::Int32)
    if !CUDA.functional()
        error("CUDA is not available. Use bfs_cpu instead.")
    end
    
    # Transfer graph to GPU
    d_row_ptr = CuArray(graph.row_ptr)
    d_col_idx = CuArray(graph.col_idx)
    
    # Initialize distances
    d_distances = CUDA.fill(UNVISITED, graph.num_nodes)
    d_distances[source] = Int32(0)
    
    # Double-buffered frontier
    d_frontier = CuArray{Int32}(undef, graph.num_nodes)
    d_next_frontier = CuArray{Int32}(undef, graph.num_nodes)
    d_frontier[1] = source
    
    # Frontier sizes
    d_frontier_size = CuArray([Int32(1)])
    d_next_frontier_size = CuArray([Int32(0)])
    
    current_level = Int32(0)
    block_size = 256
    
    while true
        h_frontier_size = Array(d_frontier_size)[1]
        if h_frontier_size == 0
            break
        end
        
        # Reset next frontier size
        d_next_frontier_size .= Int32(0)
        
        # Launch kernel
        num_blocks = cld(h_frontier_size, block_size)
        @cuda threads=block_size blocks=num_blocks bfs_kernel!(
            d_row_ptr, d_col_idx,
            d_distances, d_frontier, d_frontier_size,
            d_next_frontier, d_next_frontier_size,
            current_level
        )
        CUDA.synchronize()
        
        # Swap frontiers
        d_frontier, d_next_frontier = d_next_frontier, d_frontier
        d_frontier_size, d_next_frontier_size = d_next_frontier_size, d_frontier_size
        
        current_level += Int32(1)
    end
    
    return Array(d_distances)
end

# ==============================================================================
# GPU Afforest Implementation
# ==============================================================================

"""
GPU kernel to initialize parent array (each node is its own parent).
"""
function init_parent_kernel!(parent, num_nodes)
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if tid <= num_nodes
        parent[tid] = Int32(tid)
    end
    return nothing
end

"""
GPU kernel for Afforest random neighbor sampling.
Each node connects to a random neighbor.
"""
function afforest_sampling_kernel!(row_ptr, col_idx, parent, num_nodes, seed)
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if tid <= num_nodes
        start_idx = row_ptr[tid]
        end_idx = row_ptr[tid + 1] - 1
        degree = end_idx - start_idx + 1
        
        if degree > 0
            # Simple pseudo-random using linear congruential generator
            r = (seed * 1664525 + 1013904223 + tid) % typemax(UInt32)
            offset = r % degree
            neighbor = col_idx[start_idx + offset]
            
            p_u = parent[tid]
            p_v = parent[neighbor]
            
            # Link smaller to larger
            if p_v < p_u
                CUDA.atomic_min!(pointer(parent, p_u), p_v)
            elseif p_u < p_v
                CUDA.atomic_min!(pointer(parent, p_v), p_u)
            end
        end
    end
    return nothing
end

"""
GPU kernel for Hook phase - connect adjacent nodes.
"""
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

"""
GPU kernel for path compression.
"""
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

GPU-accelerated Afforest algorithm for connected components.

# Arguments
- `graph`: CSRGraph to analyze
- `sampling_rounds`: Number of random sampling rounds (default: 2)

# Returns
- `Vector{Int32}`: Component ID for each node (root of component)
"""
function afforest_gpu(graph::CSRGraph; sampling_rounds::Int=2)
    if !CUDA.functional()
        error("CUDA is not available. Use afforest_cpu instead.")
    end
    
    num_nodes = Int32(graph.num_nodes)
    block_size = 256
    num_blocks = cld(num_nodes, block_size)
    
    # Transfer graph to GPU
    d_row_ptr = CuArray(graph.row_ptr)
    d_col_idx = CuArray(graph.col_idx)
    
    # Initialize parent array
    d_parent = CuArray{Int32}(undef, num_nodes)
    @cuda threads=block_size blocks=num_blocks init_parent_kernel!(d_parent, num_nodes)
    
    d_changed = CuArray([Int32(0)])
    
    println("🌲 Running GPU Afforest...")
    
    # --- AFFOREST SAMPLING PHASE ---
    println("  Phase 1: Random Neighbor Sampling ($sampling_rounds rounds)")
    
    for round in 1:sampling_rounds
        @cuda threads=block_size blocks=num_blocks afforest_sampling_kernel!(
            d_row_ptr, d_col_idx, d_parent, num_nodes, UInt32(round + 42)
        )
        @cuda threads=block_size blocks=num_blocks compress_kernel!(d_parent, num_nodes)
        CUDA.synchronize()
    end
    
    # --- HOOK PHASE ---
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
    
    # Final compression rounds
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

Compare CPU and GPU BFS results for correctness.

# Returns
- `Bool`: true if results match, false otherwise
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
            if mismatches <= 5  # Show first 5 mismatches
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
