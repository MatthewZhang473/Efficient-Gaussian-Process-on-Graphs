import numpy as np
from tqdm import tqdm

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
        self.rng = np.random.default_rng(seed)  # Use NumPy's Generator for random numbers

    def _perform_single_walk(self, start_node, p_halt, max_walk_length):
        num_nodes = self.graph.get_num_nodes()
        walk_matrix = np.zeros((num_nodes, max_walk_length), dtype=float)
        
        current_node = start_node
        load = 1.0
        for step in range(max_walk_length):
            walk_matrix[current_node, step] = load
            neighbors = self.graph.get_neighbors(current_node)
            degree = neighbors.size
            if degree == 0 or self.rng.random() < p_halt:
                break
            new_node = self.rng.choice(neighbors)
            weight = self.graph.get_edge_weight(current_node, new_node)
            load *= degree * weight / (1 - p_halt)
            current_node = new_node
        return walk_matrix

    def _perform_multiple_walks(self, start_node, num_walks, p_halt, max_walk_length):
        num_nodes = self.graph.get_num_nodes()
        cumulative_matrix = np.zeros((num_nodes, max_walk_length), dtype=float)
        
        for _ in range(num_walks):
            cumulative_matrix += self._perform_single_walk(start_node, p_halt, max_walk_length)
        return cumulative_matrix / num_walks

    def get_random_walk_matrices(self, num_walks, p_halt, max_walk_length, use_tqdm=False):
        """
        Perform multiple random walks for each node in the graph as a starting point.
        Returns a NumPy array of shape (num_nodes, num_nodes, max_walk_length).
        """
        num_nodes = self.graph.get_num_nodes()
        # Preallocate a 3D NumPy array to store feature matrices for each start node
        feature_matrices = np.zeros((num_nodes, num_nodes, max_walk_length), dtype=float)
    
        iterator = tqdm(range(num_nodes), desc="Random walks", disable=not use_tqdm)
        for start_node in iterator:
            feature_matrix = self._perform_multiple_walks(start_node, num_walks, p_halt, max_walk_length)
            feature_matrices[start_node] = feature_matrix
        return feature_matrices

if __name__ == "__main__":
    # Define the adjacency matrix
    adjacency_matrix = np.array([
        [0, 1, 1, 1],
        [0, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 0, 0]
    ], dtype=float)

    # Create Graph instance
    graph = Graph(adjacency_matrix=adjacency_matrix)

    # Create RandomWalk instance
    random_walk = RandomWalk(graph, seed=42)

    # Parameters for the random walk
    num_walks = 1000
    max_walk_length = 4
    p_halt = 0.5

    # Perform the random walks and get the feature matrices as a NumPy array
    feature_matrices = random_walk.get_random_walk_matrices(num_walks, p_halt, max_walk_length)

    # Output the feature matrices
    print("Feature matrices shape:", feature_matrices.shape)
    for start_node in range(graph.get_num_nodes()):
        print(f"\nFeature matrix for start node {start_node}:")
        print(feature_matrices[start_node])
