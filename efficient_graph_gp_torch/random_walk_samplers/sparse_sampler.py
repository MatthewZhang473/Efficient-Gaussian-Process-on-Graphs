import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


class SparseRandomWalk:
    """
    Efficient sparse random walk sampler for weighted graphs.
    
    Computes random walk feature matrices using sparse operations.

    
    Args:
        adjacency_matrix: Sparse adjacency matrix (any scipy.sparse format)
        seed: Random seed for reproducibility (None for random)
    
    Returns:
        List of sparse CSR matrices, one per starting node
    """
    def __init__(self, adjacency_matrix, seed=None):
        self.adjacency = adjacency_matrix.tocsr()
        self.num_nodes = adjacency_matrix.shape[0]
        self.rng = np.random.default_rng(seed)  # None seed automatically uses random entropy
    
    def _get_neighbors_and_weights(self, node_idx):
        row = self.adjacency.getrow(node_idx)
        return row.indices, row.data
    
    def _perform_single_walk(self, start_node_idx, p_halt, max_walk_length):
        walk_data = sp.lil_matrix((self.num_nodes, max_walk_length))
        current_node = start_node_idx
        load = 1.0
        
        for step in range(max_walk_length):
            walk_data[current_node, step] = load
            neighbors, weights = self._get_neighbors_and_weights(current_node)
            degree = len(neighbors)
            
            if degree == 0 or self.rng.random() < p_halt:
                break
            
            next_idx = self.rng.choice(degree)
            current_node = neighbors[next_idx]
            weight = weights[next_idx]
            load *= degree * weight / (1 - p_halt)
        
        return walk_data.tocsr()
    
    def _perform_multiple_walks(self, start_node_idx, num_walks, p_halt, max_walk_length):
        cumulative_matrix = sp.csr_matrix((self.num_nodes, max_walk_length))
        
        for _ in range(num_walks):
            walk_matrix = self._perform_single_walk(start_node_idx, p_halt, max_walk_length)
            cumulative_matrix += walk_matrix
        
        return cumulative_matrix / num_walks
    
    def get_random_walk_matrices(self, num_walks, p_halt, max_walk_length, use_tqdm=False):
        feature_matrices = []
        iterator = tqdm(range(self.num_nodes), desc="Random walks", disable=not use_tqdm)
        
        for start_node_idx in iterator:
            feature_matrix = self._perform_multiple_walks(start_node_idx, num_walks, p_halt, max_walk_length)
            feature_matrices.append(feature_matrix)
        
        return feature_matrices


if __name__ == "__main__":
    # Create a simple CSR adjacency matrix
    rows = [0, 0, 0, 1, 1, 2, 2, 3]
    cols = [1, 2, 3, 0, 2, 0, 3, 0]
    data = [1, 1, 1, 1, 1, 1, 1, 1]
    adjacency = sp.csr_matrix((data, (rows, cols)), shape=(4, 4))
    
    random_walk = SparseRandomWalk(adjacency, seed=None)
    feature_matrices = random_walk.get_random_walk_matrices(
        num_walks=10000, p_halt=0.1, max_walk_length=4, use_tqdm=True
    )
    
    print(f"Generated {len(feature_matrices)} feature matrices")
    print(f"Shape: {feature_matrices[0].shape}")
    print(f"Sparsity: {sum(m.nnz for m in feature_matrices) / sum(m.size for m in feature_matrices):.3f}")
    print(feature_matrices[0].todense())