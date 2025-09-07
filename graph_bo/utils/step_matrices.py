import os
import pickle
import hashlib
import scipy.sparse as sp
from typing import List
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)
from efficient_graph_gp_sparse.preprocessor import GraphPreprocessor

def load_or_compute_step_matrices(adj_matrix: sp.csr_matrix,
                                 walks_per_node: int,
                                 p_halt: float, 
                                 max_walk_length: int,
                                 cache_dir: str,
                                 dataset_name: str) -> List[sp.csr_matrix]:
    """
    Load step matrices from cache or compute them using GraphPreprocessor.
    
    Args:
        adj_matrix: Adjacency matrix
        walks_per_node: Number of walks per node
        p_halt: Halt probability
        max_walk_length: Maximum walk length
        cache_dir: Cache directory
        dataset_name: Dataset name for logging
        
    Returns:
        List of scipy CSR step matrices
    """

    # Create cache filename in the specified directory
    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = os.path.join(cache_dir, f"{dataset_name}_step_matrices.pkl")
    
    # Check if cached matrices exist
    load_from_disk = os.path.exists(cache_filename)
    
    if load_from_disk:
        print(f"Loading step matrices from cache: {os.path.basename(cache_filename)}")
    else:
        print(f"Computing step matrices for {dataset_name}...")
    
    # Create GraphPreprocessor instance
    preprocessor = GraphPreprocessor(
        adjacency_matrix=adj_matrix,
        walks_per_node=walks_per_node,
        p_halt=p_halt,
        max_walk_length=max_walk_length,
        random_walk_seed=42,
        load_from_disk=load_from_disk,
        use_tqdm=True,
        cache_filename=cache_filename,
        n_processes=None  # Use all available cores
    )
    
    if not load_from_disk:
        # Compute and save step matrices
        _ = preprocessor.preprocess_graph(save_to_disk=True)
        print(f"Cached step matrices to: {os.path.basename(cache_filename)}")

    # Return the torch tensors
    return preprocessor.step_matrices_torch

def save_step_matrices(step_matrices: List[sp.csr_matrix], filename: str) -> None:
    """Save step matrices to disk using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(step_matrices, f)

def load_step_matrices(filename: str) -> List[sp.csr_matrix]:
    """Load step matrices from disk."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
