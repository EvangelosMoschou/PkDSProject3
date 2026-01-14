import struct
import numpy as np
import os
import sys
import time

def generate_uniform_graph(filename, num_nodes, num_edges):
    print(f"Generating UNIFORM Random Graph: N={num_nodes}, E={num_edges}")
    start_time = time.time()
    
    # 1. Generate Degrees (Poisson)
    print("Generating degrees...")
    avg_degree = num_edges // num_nodes
    degrees = np.random.poisson(avg_degree, num_nodes).astype(np.int64)
    
    # Adjust to match exactly num_edges
    current_sum = np.sum(degrees)
    diff = num_edges - current_sum
    if diff > 0:
        degrees[0:diff] += 1
    elif diff < 0:
        degrees[0:abs(diff)] -= 1
        
    # Find max degree node
    max_deg_idx = np.argmax(degrees)
    print(f"\n>>> SUGGESTED SOURCE NODE: {max_deg_idx} (Degree: {degrees[max_deg_idx]}) <<<\n")
    
    # 2. Row Pointers
    print("Generating row pointers...")
    row_ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(degrees, out=row_ptr[1:])
    total_edges = row_ptr[-1]
    
    # 3. Column Indices (Uniform Random)
    # This ensures low diameter due to "small world" property of random graphs
    print("Generating col indices...")
    col_idx = np.random.randint(0, num_nodes, total_edges, dtype=np.int32)
    col_idx.sort() # Local sort helps CSR slightly but global structure is random
            
    print(f"Generation took {time.time() - start_time:.2f}s")

    # Write to binary
    print(f"Writing to {filename}...")
    with open(filename, 'wb') as f:
        f.write(struct.pack('Q', int(num_nodes)))
        f.write(struct.pack('Q', int(total_edges)))
        row_ptr.tofile(f)
        col_idx.tofile(f)
        
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 generate_graph.py <output_file>")
        sys.exit(1)
        
    output_file = sys.argv[1]
    
    # Optimized Parameters for Speedup
    # N = 40,000,000 
    # E = 400,000,000
    N = 40_000_000
    E = 400_000_000
    
    generate_uniform_graph(output_file, N, E)
