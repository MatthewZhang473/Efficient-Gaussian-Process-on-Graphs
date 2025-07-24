try:
    from ..random_walk_samplers_sparse import SparseRandomWalk
    from ..utils_sparse import get_normalized_laplacian
except ImportError:
    # For running directly or when relative imports fail
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from random_walk_samplers_sparse import SparseRandomWalk
    from utils import get_normalized_laplacian


def fast_general_grf_kernel(adj_matrix, modulator_vector, walks_per_node=50, p_halt=0.1, max_walk_length=10): 
    """
    Construct graph random features on the normalized graph Laplacian.
    
    Args:
    - adj_matrix: Sparse adjacency matrix of the graph.
    - modulator_vector: Dense vector modulating the random walk features (length = max_walk_length).
    - walks_per_node: Number of random walks per node.
    - p_halt: Probability of halting the random walk.
    - max_walk_length: Maximum length of the random walks.
    
    Returns:
    - Sparse kernel matrix of shape (num_nodes, num_nodes).
    """
    import scipy.sparse as sp
    
    laplacian = get_normalized_laplacian(adj_matrix)
    random_walk = SparseRandomWalk(laplacian, seed=None)
    step_matrices = random_walk.get_random_walk_matrices(walks_per_node, p_halt, max_walk_length)
    
    # Compute Phi = sum(f_p * M_p) where f_p are modulator weights
    num_nodes = adj_matrix.shape[0]
    Phi = sp.csr_matrix((num_nodes, num_nodes))
    
    for step, f_p in enumerate(modulator_vector):
        if step < len(step_matrices):
            Phi += f_p * step_matrices[step]
    
    # Keep kernel matrix sparse: K = Phi @ Phi.T
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
    modulator_vector = np.random.randn(3)  # 3 steps to match max_walk_length
    
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
    print(f"Kernel matrix sparsity: {kernel_matrix.nnz / n_nodes**2:.4f}")
    print(f"Sample of kernel matrix (first 10x10):\n{kernel_matrix[:10, :10].toarray()}")

