import scipy.io
import os
import struct
import numpy as np

def convert_mat_to_csrbin(mat_path, bin_path):
    print(f"Loading {mat_path}...")
    try:
        data = scipy.io.loadmat(mat_path)
    except Exception as e:
        print(f"Failed to load with scipy: {e}")
        return False
    
    # Try to find the adjacency matrix 'A'
    if 'Problem' in data:
        A = data['Problem'][0,0]['A']
    elif 'A' in data:
        A = data['A']
    else:
        A = None
        for k, v in data.items():
            if scipy.sparse.issparse(v):
                print(f"Found sparse matrix in key '{k}'")
                A = v
                break
        if A is None:
            print("No sparse matrix found in .mat file")
            return False
            
    # Ensure it's CSR
    A = A.tocsr()
    
    # Symmetrize for correct Bottom-Up BFS!
    # A = A | A.T (equivalent for binary adjacency)
    # A = A + A.T and then setting data=1 is safer
    print("Symmetrizing graph...")
    A = A + A.T
    A.data = np.ones_like(A.data)
    A = A.tocsr()
    A.eliminate_zeros()
    
    num_nodes = A.shape[0]
    num_edges = A.nnz
    
    print(f"Nodes: {num_nodes}, Edges: {num_edges}")
    
    # Write to bin
    with open(bin_path, 'wb') as f:
        f.write(struct.pack('QQ', num_nodes, num_edges))
        row_ptr = A.indptr.astype(np.int64)
        f.write(row_ptr.tobytes())
        col_idx = A.indices.astype(np.int32)
        f.write(col_idx.tobytes())
        
    print(f"Success! Saved to {bin_path}")
    return True

if __name__ == "__main__":
    mat_file = "/media/vaggelis/SHARED_DATA/AUTH/7th Semester/Parallel/Project4/Mat Files/road_usa.mat"
    bin_file = "/media/vaggelis/SHARED_DATA/AUTH/7th Semester/Parallel/Project4/data/road_usa.bin"
    convert_mat_to_csrbin(mat_file, bin_file)
