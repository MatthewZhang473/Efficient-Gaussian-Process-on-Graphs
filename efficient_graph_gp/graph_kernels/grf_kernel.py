from .utils import get_normalized_laplacian
import numpy as np
from math import factorial
import networkx as nx

# A networkx-based graph for utilities
class Graph:
    def __init__(self, adjacency_matrix=None):
        self.graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph) if adjacency_matrix is not None else nx.DiGraph()

    def get_neighbors(self, node):
        return list(self.graph.neighbors(node))

    def get_num_nodes(self):
        return self.graph.number_of_nodes()

    def get_edge_weight(self, node1, node2):
        return self.graph.edges[node1, node2].get('weight', 1.0) if self.graph.has_edge(node1, node2) else 0.0
class RandomWalk:
    def __init__(self, graph: Graph, seed=None):
        self.graph = graph
        if seed is not None:
            np.random.seed(seed)  # Set the random seed for reproducibility

    def _perform_walk(self, start_node, max_steps, modulation_function=None, p_halt=0.1):
        current_node = start_node
        walk_length = 0
        load = 1.0
        feature_vector = np.zeros(self.graph.get_num_nodes())

        while True:
            feature_vector[current_node] += load * (modulation_function(walk_length) if modulation_function else 1)
            walk_length += 1

            neighbors = self.graph.get_neighbors(current_node)
            if not neighbors:
                break

            new_node = np.random.choice(neighbors)
            degree = len(neighbors)
            weight = self.graph.get_edge_weight(current_node, new_node)
            load *= degree / (1 - p_halt) * weight
            current_node = new_node

            if np.random.rand() < p_halt:
                break

        return feature_vector

    def calculate_feature_vector(self, start_node, num_walks, max_steps, modulation_function, p_halt=0.1):
        feature_vector = np.zeros(self.graph.get_num_nodes())
        for _ in range(num_walks):
            feature_vector += self._perform_walk(start_node, max_steps, modulation_function, p_halt)
        return feature_vector / num_walks

class GeneralGraphRandomFeatures:
    def __init__(self, graph: Graph, modulation_function=None, max_walk_length=10, beta=1.0, seed=None):
        self.graph = graph
        self.random_walk = RandomWalk(graph, seed=seed)  # Pass seed to RandomWalk
        self.modulation_function = modulation_function
        self.max_walk_length = max_walk_length
        self.beta = beta

    def generate_features(self, num_walks=50, p_halt=0.1):
        num_nodes = self.graph.get_num_nodes()
        feature_matrix = np.zeros((num_nodes, num_nodes))

        for node in range(num_nodes):
            feature_matrix[node, :] = self.random_walk.calculate_feature_vector(
                start_node=node,
                num_walks=num_walks,
                max_steps=self.max_walk_length,
                modulation_function=lambda length: self.modulation_function(length, self.beta),
                p_halt=p_halt
            )

        return feature_matrix

def grf_kernel(adj_matrix, walks_per_node=50, p_halt=0.1, modulation_function=None, beta=1.0): 
    """
    Construct graph random features on the normalized graph Laplacian.
    """

    laplacian = get_normalized_laplacian(adj_matrix)
    # Use two independent GRFs to avoid biased estimation
    grf_1 = GeneralGraphRandomFeatures(Graph(laplacian), modulation_function, beta=beta, seed=42)  # Hard-coded seed 1
    grf_2 = GeneralGraphRandomFeatures(Graph(laplacian), modulation_function, beta=beta, seed=84)  # Hard-coded seed 2
    feature_matrix_1 = grf_1.generate_features(num_walks=walks_per_node, p_halt=p_halt)
    feature_matrix_2 = grf_2.generate_features(num_walks=walks_per_node, p_halt=p_halt)

    # Return their product
    return feature_matrix_1 @ feature_matrix_2.T
