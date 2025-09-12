import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ---- Global read-only adjacency matrix (populated once per worker via initializer)
_G_ADJACENCY = None
_G_NUM_NODES = None

def _init_worker(adjacency_matrix, num_nodes):
    """Runs once in each worker: bind globals to parent's adjacency matrix memory (fork: CoW)."""
    global _G_ADJACENCY, _G_NUM_NODES
    _G_ADJACENCY = adjacency_matrix
    _G_NUM_NODES = num_nodes

def _get_neighbors(node):
    """Get neighbors using global adjacency matrix."""
    return np.flatnonzero(_G_ADJACENCY[node])

def _get_edge_weight(node1, node2):
    """Get edge weight using global adjacency matrix."""
    return _G_ADJACENCY[node1, node2]

def _worker_walks(args):
    """Worker: performs walks for assigned nodes using global adjacency matrix."""
    nodes, num_walks, p_halt, max_walk_length, seed, show_progress = args
    rng = np.random.default_rng(seed)

    step_accumulators = [defaultdict(float) for _ in range(max_walk_length)]
    it = tqdm(nodes, desc="Process walks", disable=not show_progress) if show_progress else nodes

    for start_node in it:
        for _ in range(num_walks):
            current_node = start_node
            load = 1.0
            
            for step in range(max_walk_length):
                # Accumulate (start_node, current_node) pair
                step_accumulators[step][(start_node, current_node)] += load
                
                neighbors = _get_neighbors(current_node)
                degree = neighbors.size
                
                if degree == 0 or rng.random() < p_halt:
                    break
                
                # Pick random neighbor
                next_node = rng.choice(neighbors)
                weight = _get_edge_weight(current_node, next_node)
                load *= degree * weight / (1 - p_halt)
                current_node = next_node

    return step_accumulators

class Graph:
    def __init__(self, adjacency_matrix=None):
        if adjacency_matrix is not None:
            self.adjacency_matrix = adjacency_matrix
            self.num_nodes = adjacency_matrix.shape[0]
        else:
            self.adjacency_matrix = None
            self.num_nodes = 0

    def get_neighbors(self, node):
        # Return indices where adjacency_matrix[node] is non-zero
        return np.flatnonzero(self.adjacency_matrix[node])

    def get_num_nodes(self):
        return self.num_nodes

    def get_edge_weight(self, node1, node2):
        return self.adjacency_matrix[node1, node2]


class RandomWalk:
    def __init__(self, graph: Graph, seed=None):
        self.graph = graph
        self.rng = np.random.default_rng(seed)
        self.seed = seed or 42

    def get_random_walk_matrices(self, num_walks, p_halt, max_walk_length, use_tqdm=False, n_processes=None, ablation=False):
        """
        Perform multiple random walks for each node in the graph as a starting point.
        Returns a NumPy array of shape (num_nodes, num_nodes, max_walk_length).
        """
        num_nodes = self.graph.get_num_nodes()
        
        if n_processes is None:
            n_processes = os.cpu_count()
        
        # If single process or small graph, use original sequential implementation
        if n_processes == 1 or num_nodes < n_processes * 2:
            return self._sequential_walks(num_walks, p_halt, max_walk_length, use_tqdm, ablation)
        
        # Split nodes among processes
        chunks = np.array_split(np.arange(num_nodes), n_processes)
        
        # Use fork (Linux) so workers share memory via CoW; also set initializer to bind globals
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=n_processes,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(self.graph.adjacency_matrix, num_nodes),
        ) as executor:

            args = [
                (chunk.tolist(), num_walks, p_halt, max_walk_length, self.seed + i, use_tqdm and i == 0)
                for i, chunk in enumerate(chunks)
            ]

            # Stream results to avoid holding all at once
            futures = [executor.submit(_worker_walks, a) for a in args]
            step_accumulators = [defaultdict(float) for _ in range(max_walk_length)]

            for fut in as_completed(futures):
                result = fut.result()
                for step in range(max_walk_length):
                    for k, v in result[step].items():
                        step_accumulators[step][k] += v

        # Build final 3D array from accumulated results
        return self._build_matrices_from_accumulators(step_accumulators, num_walks, num_nodes, max_walk_length)
    
    def _sequential_walks(self, num_walks, p_halt, max_walk_length, use_tqdm, ablation=False):
        """Original sequential implementation."""
        num_nodes = self.graph.get_num_nodes()
        step_accumulators = [defaultdict(float) for _ in range(max_walk_length)]
        
        iterator = tqdm(range(num_nodes), desc="Random walks", disable=not use_tqdm)
        
        for start_node in iterator:
            for _ in range(num_walks):
                current_node = start_node
                load = 1.0
                
                for step in range(max_walk_length):
                    # Accumulate (start_node, current_node) pair
                    step_accumulators[step][(start_node, current_node)] += load
                    
                    neighbors = self.graph.get_neighbors(current_node)
                    degree = neighbors.size
                    
                    if degree == 0 or self.rng.random() < p_halt:
                        break
                    
                    # Pick random neighbor
                    next_node = self.rng.choice(neighbors)
                    weight = self.graph.get_edge_weight(current_node, next_node)
                    if ablation:
                        load = weight
                    else:
                        load = degree * weight / (1 - p_halt)
                    current_node = next_node
        
        return self._build_matrices_from_accumulators(step_accumulators, num_walks, num_nodes, max_walk_length)
    
    def _build_matrices_from_accumulators(self, step_accumulators, num_walks, num_nodes, max_walk_length):
        """Build final 3D array from accumulated results."""
        feature_matrices = np.zeros((num_nodes, num_nodes, max_walk_length), dtype=float)
        
        for step in range(max_walk_length):
            acc = step_accumulators[step]
            for (start_node, current_node), value in acc.items():
                feature_matrices[start_node, current_node, step] = value / num_walks
        
        return feature_matrices

if __name__ == "__main__":
    import time
    
    print(f"Available CPU cores: {os.cpu_count()}")
    
    # Define a larger adjacency matrix for better multiprocessing demonstration
    np.random.seed(42)
    n_nodes = 100
    # Create a random sparse adjacency matrix
    adjacency_matrix = np.random.rand(n_nodes, n_nodes)
    adjacency_matrix = (adjacency_matrix > 0.95).astype(float)  # Make it sparse
    # Make it symmetric and remove self-loops
    adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2
    np.fill_diagonal(adjacency_matrix, 0)
    
    print(f"Graph: {n_nodes} nodes, {np.sum(adjacency_matrix)} edges")
    print(f"Sparsity: {np.sum(adjacency_matrix > 0) / (n_nodes * n_nodes):.3f}")

    # Create Graph instance
    graph = Graph(adjacency_matrix=adjacency_matrix)

    # Create RandomWalk instance
    random_walk = RandomWalk(graph, seed=42)

    # Parameters for the random walk
    num_walks = 1000
    max_walk_length = 4
    p_halt = 0.5

    # Test sequential processing
    print("\n=== Sequential Processing ===")
    start_time = time.time()
    feature_matrices_seq = random_walk.get_random_walk_matrices(
        num_walks, p_halt, max_walk_length, use_tqdm=True, n_processes=1
    )
    seq_time = time.time() - start_time
    print(f"Sequential time: {seq_time:.2f} seconds")

    # Test multiprocessing
    print("\n=== Multiprocessing ===")
    start_time = time.time()
    feature_matrices_mp = random_walk.get_random_walk_matrices(
        num_walks, p_halt, max_walk_length, use_tqdm=True, n_processes=None
    )
    mp_time = time.time() - start_time
    print(f"Multiprocessing time: {mp_time:.2f} seconds")
    print(f"Speedup: {seq_time / mp_time:.2f}x")

    # Verify results are consistent
    max_diff = np.max(np.abs(feature_matrices_seq - feature_matrices_mp))
    print(f"\nMax difference between sequential and multiprocessing results: {max_diff:.10f}")
    
    # Output the feature matrices shape and some statistics
    print(f"\nFeature matrices shape: {feature_matrices_mp.shape}")
    print(f"Non-zero entries: {np.sum(feature_matrices_mp > 0)}")
    print(f"Density: {np.sum(feature_matrices_mp > 0) / feature_matrices_mp.size:.6f}")
    
    # Show example values for first few nodes
    print(f"\nExample feature matrix for start node 0 (step 0):")
    print(feature_matrices_mp[0, :10, 0])  # First 10 destinations, step 0
