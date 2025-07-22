try:
    from ..random_walk_samplers import SparseRandomWalk
    from ..utils import get_normalized_laplacian
except ImportError:
    # For running directly or when relative imports fail
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from random_walk_samplers import SparseRandomWalk
    from utils import get_normalized_laplacian


def fast_general_grf_kernel(adj_matrix, modulator_vector, walks_per_node=50, p_halt=0.1, max_walk_length=10): 
    """
    Construct graph random features on the normalized graph Laplacian.
    
    Args:
    - adj_matrix: Sparse adjacency matrix of the graph.
    - modulator_vector: Dense vector modulating the random walk features.
    - walks_per_node: Number of random walks per node.
    - p_halt: Probability of halting the random walk.
    - max_walk_length: Maximum length of the random walks.
    
    Returns:
    - Phi: Kernel matrix of shape (num_nodes, num_nodes).
    """
    import scipy.sparse as sp
    
    laplacian = get_normalized_laplacian(adj_matrix)
    random_walk = SparseRandomWalk(laplacian, seed=None)
    feature_matrices = random_walk.get_random_walk_matrices(walks_per_node, p_halt, max_walk_length)
    
    # Stack phi vectors for each node (convert to proper sparse format)
    num_nodes = len(feature_matrices)
    phi_rows = []
    for i in range(num_nodes):
        phi_vector = feature_matrices[i] @ modulator_vector
        phi_rows.append(sp.csr_matrix(phi_vector).reshape(1, -1))
    
    Phi = sp.vstack(phi_rows)
    return Phi @ Phi.T


if __name__ == "__main__":
    import numpy as np
    from scipy.sparse import csr_matrix
    
    # Create a larger 100-node cycle graph
    n_nodes = 100
    row = []
    col = []
    for i in range(n_nodes):
        # Connect each node to the next (and last to first for cycle)
        next_node = (i + 1) % n_nodes
        row.extend([i, next_node])
        col.extend([next_node, i])
    
    data = [1] * len(row)
    adj_matrix = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    
    # Random modulator vector
    np.random.seed(42)
    modulator_vector = np.random.randn(3)  # 5 steps
    
    # Test the function
    print("Testing fast_general_grf_kernel...")
    print(f"Graph adjacency matrix shape: {adj_matrix.shape}")
    print(f"Modulator vector shape: {modulator_vector.shape}")
    print(f"Graph edges: {adj_matrix.nnz}")
    
    kernel_matrix = fast_general_grf_kernel(
        adj_matrix, 
        modulator_vector, 
        walks_per_node=20, 
        p_halt=0.15, 
        max_walk_length=3
    )
    
    print(f"Kernel matrix shape: {kernel_matrix.shape}")
    print(f"Kernel matrix sparsity: {kernel_matrix.nnz / (kernel_matrix.shape[0] * kernel_matrix.shape[1]):.4f}")
    print(f"Sample of kernel matrix (first 5x5):\n{kernel_matrix[:10, :10].toarray()}")
    try:
        print("Positive definite:", np.all(np.linalg.eigvals(kernel_matrix.toarray()) >= -1e-10))
    except:
        print("Could not check positive definiteness (matrix too large)")

