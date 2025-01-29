from ..random_walk_samplers import Graph, RandomWalk
from ..modulation_functions import diffusion_modulator
from .utils import get_normalized_laplacian
import numpy as np


def fast_general_grf_kernel(adj_matrix, modulator_vector, walks_per_node=50, p_halt=0.1, max_walk_length=10): 
    """
    Construct graph random features on the normalized graph Laplacian.
    
    The modulator_vector is a numpy array of the same length as the maximum walk length.
    """
    laplacian = get_normalized_laplacian(adj_matrix)
    graph = Graph(laplacian)
    random_walk = RandomWalk(graph, seed=42)
    feature_matrices = random_walk.get_random_walk_matrices(walks_per_node, p_halt, max_walk_length)
    
    # Use broadcasting for 3D matrix-vector multiplication
    Phi = feature_matrices @ modulator_vector  # Shape: (num_nodes, num_nodes)
    
    return Phi @ Phi.T
