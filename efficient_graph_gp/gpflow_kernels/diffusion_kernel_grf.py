import gpflow
import tensorflow as tf
import numpy as np
import networkx as nx


class Graph:
    def __init__(self, adjacency_matrix):
        """
        Wraps a NetworkX graph for graph-based utilities.
        """
        self.graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

    def get_neighbors(self, node):
        return list(self.graph.neighbors(node))

    def get_num_nodes(self):
        return self.graph.number_of_nodes()

    def get_edge_weight(self, node1, node2):
        return self.graph.edges[node1, node2].get("weight", 1.0) if self.graph.has_edge(node1, node2) else 0.0


class RandomWalk:
    def __init__(self, graph: Graph):
        self.graph = graph

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
    def __init__(self, graph: Graph, modulation_function=None, max_walk_length=10):
        self.graph = graph
        self.random_walk = RandomWalk(graph)
        self.modulation_function = modulation_function
        self.max_walk_length = max_walk_length

    def generate_features(self, num_walks=50, p_halt=0.1):
        num_nodes = self.graph.get_num_nodes()
        feature_matrix = np.zeros((num_nodes, num_nodes))

        for node in range(num_nodes):
            feature_matrix[node, :] = self.random_walk.calculate_feature_vector(
                start_node=node,
                num_walks=num_walks,
                max_steps=self.max_walk_length,
                modulation_function=self.modulation_function,
                p_halt=p_halt,
            )

        return feature_matrix

    def estimate_kernel(self, num_walks=50, p_halt=0.1):
        feature_matrix = self.generate_features(num_walks, p_halt)
        return feature_matrix @ feature_matrix.T


class GraphDiffusionGRFKernel(gpflow.kernels.Kernel):
    """
    A GPflow-compatible kernel that uses Graph Random Features (GRF) for diffusion kernel estimation.
    """

    def __init__(self, adjacency_matrix, modulation_function, **kwargs):
        super().__init__(**kwargs)
        self.adjacency_matrix = tf.convert_to_tensor(adjacency_matrix, dtype=tf.float64)
        self.beta = gpflow.Parameter(2.0, transform=gpflow.utilities.positive())  # Learnable hyperparameter
        self.num_walks = 50
        self.p_halt = 0.1
        self.modulation_function = modulation_function

    def K(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        kernel_matrix = self.grf_kernel()
        indices_X1 = tf.cast(tf.reshape(X1, [-1]), dtype=tf.int32)
        indices_X2 = tf.cast(tf.reshape(X2, [-1]), dtype=tf.int32)
        return tf.gather(tf.gather(kernel_matrix, indices_X1, axis=0), indices_X2, axis=1)

    def K_diag(self, X):
        kernel_matrix = self.grf_kernel()
        indices_X = tf.cast(tf.reshape(X, [-1]), dtype=tf.int32)
        return tf.gather(tf.linalg.diag_part(kernel_matrix), indices_X)

    def grf_kernel(self):
        """
        Computes the kernel matrix using the Graph Random Features method.
        """
        laplacian = self.get_normalized_laplacian(self.adjacency_matrix)
        graph = Graph(laplacian.numpy())  # Convert to NumPy for NetworkX compatibility
        grf = GeneralGraphRandomFeatures(graph, self.modulation_function)
        return tf.convert_to_tensor(grf.estimate_kernel(num_walks=self.num_walks, p_halt=self.p_halt), dtype=tf.float64)

    @staticmethod
    def get_normalized_laplacian(adj_matrix):
        """
        Calculates the normalized Laplacian for the graph.
        """
        degrees = tf.reduce_sum(adj_matrix, axis=1)
        safe_degrees = tf.where(degrees > 0, degrees, tf.constant(float('inf'), dtype=adj_matrix.dtype))
        D_inv_sqrt = tf.linalg.diag(1.0 / tf.sqrt(safe_degrees))
        I = tf.eye(tf.shape(adj_matrix)[0], dtype=adj_matrix.dtype)
        return I - tf.matmul(tf.matmul(D_inv_sqrt, adj_matrix), D_inv_sqrt)
