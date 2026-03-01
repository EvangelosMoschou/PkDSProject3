#!/usr/bin/env julia
# Quick test script for Friendster graph

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

println("=" ^ 50)
println("Julia BFS Test on Friendster (CPU Fallback)")
println("=" ^ 50)

# Load the graph
graph_path = "../Mat Files/com-Friendster.mat.csrbin"

# Check available memory before loading (rough check)
# Sys.free_memory()

println("\n📥 Loading graph: $graph_path")
@time graph = load_graph_csrbin(graph_path)
print_graph_stats(graph)

println("\n⚠️ Skipping GPU BFS (Graph size > GPU VRAM)")
println("🚀 Running CPU BFS from source 1...")
println("(This may take a few minutes...)")

@time distances = bfs_cpu(graph, Int32(1))

# Calculate stats
reachable = count(d -> d != UNVISITED, distances)
max_dist = maximum(d for d in distances if d != UNVISITED)

println("\n=== BFS Result (CPU) ===")
println("Source: 1")
println("Reachable Nodes: $reachable / $(graph.num_nodes) ($(round(100*reachable/graph.num_nodes, digits=2))%)")
println("Max Distance (Diameter): $max_dist")
println("==================")
