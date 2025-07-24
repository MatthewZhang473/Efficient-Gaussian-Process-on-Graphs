import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


class SparseRandomWalk:
    """
    Efficient sparse random walk sampler for weighted graphs.
    
    Uses direct COO accumulation to build step matrices without intermediate operations.
    """
    
    def __init__(self, adjacency_matrix, seed=None):
        """Initialize with adjacency matrix and optional random seed."""
        self.adjacency = adjacency_matrix.tocsr()
        self.num_nodes = adjacency_matrix.shape[0]
        self.rng = np.random.default_rng(seed)
        
        # Cache neighbors and weights for efficiency
        self._neighbors = {}
        self._weights = {}
        for node in range(self.num_nodes):
            row = self.adjacency.getrow(node)
            self._neighbors[node] = row.indices
            self._weights[node] = row.data
    
    def get_random_walk_matrices(self, num_walks, p_halt, max_walk_length, use_tqdm=False):
        """
        Generate random walk step matrices.
        
        Args:
            num_walks: Number of walks per starting node
            p_halt: Probability of halting at each step
            max_walk_length: Maximum steps per walk
            use_tqdm: Show progress bar
            
        Returns:
            List of sparse CSR matrices, each num_nodes × num_nodes representing walks at step t.
        """
        # Initialize collectors for each step
        step_data = {step: ([], [], []) for step in range(max_walk_length)}
        
        iterator = tqdm(range(self.num_nodes), desc="Random walks", disable=not use_tqdm)
        
        for start_node in iterator:
            for _ in range(num_walks):
                current_node = start_node
                load = 1.0
                
                for step in range(max_walk_length):
                    # Accumulate visit: [start_node, current_node] = load
                    step_data[step][0].append(start_node)
                    step_data[step][1].append(current_node)
                    step_data[step][2].append(load)
                    
                    # Get neighbors and check termination
                    neighbors = self._neighbors[current_node]
                    degree = len(neighbors)
                    
                    if degree == 0 or self.rng.random() < p_halt:
                        break
                    
                    # Move to next node and update load
                    next_idx = self.rng.choice(degree)
                    weight = self._weights[current_node][next_idx]  # Get weight before moving
                    current_node = neighbors[next_idx]
                    load *= degree * weight / (1 - p_halt)
        
        # Build final matrices
        step_matrices = []
        for step in range(max_walk_length):
            rows, cols, data = step_data[step]
            if len(rows) == 0:
                matrix = sp.csr_matrix((self.num_nodes, self.num_nodes))
            else:
                # COO matrix is efficient for constructing sparse matrices
                coo_matrix = sp.coo_matrix((data, (rows, cols)), shape=(self.num_nodes, self.num_nodes))
                # CSR matrix is efficient for arithmetic operations
                matrix = coo_matrix.tocsr() / num_walks
            step_matrices.append(matrix)
        
        return step_matrices


if __name__ == "__main__":
    # Example usage
    rows = [0, 0, 0, 1, 1, 2, 2, 3]
    cols = [1, 2, 3, 0, 2, 0, 3, 0]
    data = [1, 1, 1, 1, 1, 1, 1, 1]
    adjacency = sp.csr_matrix((data, (rows, cols)), shape=(4, 4))
    
    walker = SparseRandomWalk(adjacency, seed=42)
    step_matrices = walker.get_random_walk_matrices(
        num_walks=1000, p_halt=0.1, max_walk_length=6, use_tqdm=True
    )
    
    print(f"Generated {len(step_matrices)} step matrices")
    print("Example step matrix (t=1):")
    print(step_matrices[3].todense())
    print(f"Shape: {step_matrices[0].shape}")
    print(f"Sparsity: {sum(m.nnz for m in step_matrices) / sum(m.shape[0]**2 for m in step_matrices):.3f}")