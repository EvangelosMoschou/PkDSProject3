import struct
import numpy as np
import os
import sys

def generate_random_graph(filename, num_nodes, num_edges):
    print(f"Generating Random Graph: N={num_nodes}, E={num_edges}")
    
    # 1. Generate Edges
    # We want a connected-ish graph, but pure random is fine for BFS benchmarking.
    # To ensure valid CSR, we generate sorted random edges.
    
    print("Generating random edges...")
    # Generate random sources and destinations
    # We use numpy for speed
    
    # Method: Generate average degree per node, then distribute
    avg_degree = num_edges // num_nodes
    print(f"Average Degree: {avg_degree}")
    
    # Generate row pointers directly?
    # No, let's just generate random degrees that sum to num_edges
    degrees = np.random.poisson(avg_degree, num_nodes).astype(np.int64)
    
    # Adjust to match exactly num_edges
    current_sum = np.sum(degrees)
    diff = num_edges - current_sum
    if diff > 0:
        degrees[0:diff] += 1
    elif diff < 0:
        degrees[0:abs(diff)] -= 1
        
    # Calculate row pointers (prefix sum)
    row_ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(degrees, out=row_ptr[1:])
    row_ptr[0] = 0
    
    total_edges = row_ptr[-1]
    print(f"Actual Edges: {total_edges}")
    
    # Generate column indices
    print("Generating column indices...")
    col_idx = np.random.randint(0, num_nodes, total_edges, dtype=np.int32)
    
    # Write to binary format
    # Format:
    # N (8 bytes)
    # M (8 bytes)
    # row_ptr (N+1 * 8 bytes)
    # col_idx (M * 4 bytes)
    
    print(f"Writing to {filename}...")
    with open(filename, 'wb') as f:
        # Header
        f.write(struct.pack('Q', num_nodes))
        f.write(struct.pack('Q', total_edges))
        
        # Arrays
        # row_ptr is int64 (8 bytes)
        row_ptr.tofile(f)
        
        # col_idx is int32 (4 bytes)
        col_idx.tofile(f)
        
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 generate_graph.py <output_file>")
        sys.exit(1)
        
    output_file = sys.argv[1]
    
    # Parameters for ~2GB total VRAM footprint
    # N = 20,000,000
    # E = 200,000,000
    N = 20_000_000
    E = 200_000_000
    
    generate_random_graph(output_file, N, E)
