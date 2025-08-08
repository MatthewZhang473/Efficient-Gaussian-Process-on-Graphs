import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import os


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
        self.seed = seed or 42
        
        # Cache neighbors and weights for efficiency
        self._neighbors = {}
        self._weights = {}
        for node in range(self.num_nodes):
            row = self.adjacency.getrow(node)
            self._neighbors[node] = row.indices
            self._weights[node] = row.data
    
    @staticmethod
    def _worker_walks(args):
        """Worker function for multiprocessing random walks."""
        nodes, adj_data, num_walks, p_halt, max_walk_length, seed, show_progress, n_processes = args
        
        # Reconstruct adjacency matrix and setup
        data, indices, indptr, shape = adj_data
        adjacency = sp.csr_matrix((data, indices, indptr), shape=shape)
        rng = np.random.default_rng(seed)
        
        # Cache neighbors and weights for ALL nodes (since walks can visit any node)
        neighbors = {}
        weights = {}
        for node in range(shape[0]):
            row = adjacency.getrow(node)
            neighbors[node] = row.indices
            weights[node] = row.data
        
        # Initialize dictionary accumulators instead of coordinate lists
        step_accumulators = [defaultdict(float) for _ in range(max_walk_length)]
        
        # Only show progress for one process (the first one)
        iterator = tqdm(nodes, desc=f"Process 1/{n_processes} - Nodes processed", disable=not show_progress) if show_progress else nodes
        
        for start_node in iterator:
            for _ in range(num_walks):
                current_node = start_node
                load = 1.0
                
                for step in range(max_walk_length):
                    # Accumulate load for (start_node, current_node) pair
                    step_accumulators[step][(start_node, current_node)] += load
                    
                    # Get neighbors and check termination
                    node_neighbors = neighbors[current_node]
                    degree = len(node_neighbors)
                    
                    if degree == 0 or rng.random() < p_halt:
                        break
                    
                    # Move to next node and update load
                    next_idx = rng.choice(degree)
                    weight = weights[current_node][next_idx]  # Get weight before moving
                    current_node = node_neighbors[next_idx]
                    load *= degree * weight / (1 - p_halt)
        
        return step_accumulators

    def get_random_walk_matrices(self, num_walks, p_halt, max_walk_length, use_tqdm=False, n_processes=None):
        """
        Generate random walk step matrices.
        
        Args:
            num_walks: Number of walks per starting node
            p_halt: Probability of halting at each step
            max_walk_length: Maximum steps per walk
            use_tqdm: Show progress bar
            n_processes: Number of processes (default: CPU count)
            
        Returns:
            List of sparse CSR matrices, each num_nodes × num_nodes representing walks at step t.
        """
        if n_processes is None:
            n_processes = os.cpu_count()
        
        # Split nodes across processes
        chunks = np.array_split(range(self.num_nodes), n_processes)
        adj_data = (self.adjacency.data, self.adjacency.indices, self.adjacency.indptr, self.adjacency.shape)
        
        # Prepare worker arguments - only first process shows progress
        args = [
            (chunk.tolist(), adj_data, num_walks, p_halt, max_walk_length, self.seed + i, use_tqdm and i == 0, n_processes)
            for i, chunk in enumerate(chunks)
        ]
        
        # Execute in parallel
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            results = list(executor.map(self._worker_walks, args))
        
        # Merge dictionaries from all processes
        step_accumulators = [defaultdict(float) for _ in range(max_walk_length)]
        for result in results:
            for step in range(max_walk_length):
                for coord_pair, value in result[step].items():
                    step_accumulators[step][coord_pair] += value
        
        # Convert dictionaries to COO matrices
        step_matrices = []
        for step in range(max_walk_length):
            if step_accumulators[step]:
                # Extract coordinates and values
                coord_pairs = list(step_accumulators[step].keys())
                rows = [pair[0] for pair in coord_pairs]
                cols = [pair[1] for pair in coord_pairs]  
                data = [step_accumulators[step][pair] for pair in coord_pairs]
                
                coo_matrix = sp.coo_matrix((data, (rows, cols)), shape=(self.num_nodes, self.num_nodes))
                matrix = coo_matrix.tocsr() / num_walks
            else:
                matrix = sp.csr_matrix((self.num_nodes, self.num_nodes))
                
            step_matrices.append(matrix)
        
        return step_matrices


if __name__ == "__main__":
    # Check available CPU cores
    print(f"Available CPU cores: {os.cpu_count()}")
    
    # Example usage
    rows = [0, 0, 0, 1, 1, 2, 2, 3]
    cols = [1, 2, 3, 0, 2, 0, 3, 0]
    data = [1, 1, 1, 1, 1, 1, 1, 1]
    adjacency = sp.csr_matrix((data, (rows, cols)), shape=(4, 4))
    
    walker = SparseRandomWalk(adjacency, seed=42)
    step_matrices = walker.get_random_walk_matrices(
        num_walks=100000, p_halt=0.1, max_walk_length=6, use_tqdm=True, n_processes=4
    )
    
    print(f"Generated {len(step_matrices)} step matrices")
    print("Example step matrix (t=1):")
    print(step_matrices[3].todense())
    print(f"Shape: {step_matrices[0].shape}")
    print(f"Sparsity: {sum(m.nnz for m in step_matrices) / sum(m.shape[0]**2 for m in step_matrices):.3f}")