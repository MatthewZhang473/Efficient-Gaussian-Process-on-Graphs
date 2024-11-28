from ..random_walk_samplers import Graph, RandomWalk
from ..modulation_functions import diffusion_modulator
from .utils import get_normalized_laplacian
import numpy as np


def fast_diffusion_grf_kernel(adj_matrix, walks_per_node=50, p_halt=0.1, max_walk_length=10, beta=1.0): 
    """
    Construct graph random features on the normalized graph Laplacian.
    """

    laplacian = get_normalized_laplacian(adj_matrix)
    graph = Graph(laplacian)
    random_walk = RandomWalk(graph, seed=42)
    feature_matrices = random_walk.get_random_walk_matrices(walks_per_node, p_halt, max_walk_length)
    
    modulator_vector = np.array([diffusion_modulator(step, beta) for step in range(max_walk_length)])
    num_nodes = adj_matrix.shape[0]
    Phi = np.zeros((num_nodes, num_nodes))
    for node in range(num_nodes):
        Phi[node, :] = feature_matrices[node] @ modulator_vector
    
    return Phi
        
    
    