import gpflow
import tensorflow as tf
import numpy as np
import math
import networkx as nx

# Define the diffusion modulation function
def diffusion_modulation_function(length: int, beta: float) -> float:
    numerator = (-beta) ** length
    denominator = (2 ** length) * math.factorial(length)
    return numerator / denominator

class Graph:
    def __init__(self, adjacency_matrix: np.ndarray):
        assert isinstance(adjacency_matrix, np.ndarray), "Adjacency matrix must be a NumPy array."
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "Adjacency matrix must be square."
        self.graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

    def get_neighbors(self, node: int) -> list:
        return list(self.graph.neighbors(node))

    def get_num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    def get_edge_weight(self, node1: int, node2: int) -> float:
        return self.graph.edges[node1, node2].get('weight', 1.0) if self.graph.has_edge(node1, node2) else 0.0


class RandomWalk:
    def __init__(self, graph: Graph, seed: int = None):
        self.graph = graph
        if seed is not None:
            np.random.seed(seed)

    def _perform_walk(
        self, start_node: int, max_steps: int, modulation_function=None, p_halt: float = 0.1
    ) -> np.ndarray:
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

    def calculate_feature_vector(
        self, start_node: int, num_walks: int, max_steps: int, modulation_function, p_halt: float = 0.1
    ) -> np.ndarray:
        feature_vector = np.zeros(self.graph.get_num_nodes())
        for _ in range(num_walks):
            feature_vector += self._perform_walk(start_node, max_steps, modulation_function, p_halt)
        return feature_vector / num_walks


class GeneralGraphRandomFeatures:
    def __init__(
        self,
        graph: Graph,
        modulation_function,
        max_walk_length: int = 10,
        beta: float = 1.0,
        seed: int = None,
    ):
        self.graph = graph
        self.random_walk = RandomWalk(graph, seed=seed)
        self.modulation_function = modulation_function
        self.max_walk_length = max_walk_length
        self.beta = beta

    def generate_features(self, num_walks: int = 50, p_halt: float = 0.1) -> np.ndarray:
        num_nodes = self.graph.get_num_nodes()
        feature_matrix = np.zeros((num_nodes, num_nodes))

        for node in range(num_nodes):
            feature_matrix[node, :] = self.random_walk.calculate_feature_vector(
                start_node=node,
                num_walks=num_walks,
                max_steps=self.max_walk_length,
                modulation_function=lambda length: self.modulation_function(length, self.beta),
                p_halt=p_halt,
            )

        return feature_matrix

class GraphDiffusionGRFKernel(gpflow.kernels.Kernel):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        walks_per_node: int = 50,
        p_halt: float = 0.1,
        max_walk_length: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert isinstance(adjacency_matrix, np.ndarray), "Adjacency matrix must be a NumPy array."
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "Adjacency matrix must be square."

        self.adjacency_matrix = tf.convert_to_tensor(adjacency_matrix, dtype=tf.float64)
        self.walks_per_node = walks_per_node
        self.p_halt = p_halt
        self.max_walk_length = max_walk_length
        self.beta = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())  # Learnable hyperparameter

    def K(self, X1: tf.Tensor, X2: tf.Tensor = None) -> tf.Tensor:
        if X2 is None:
            X2 = X1
        kernel_matrix = self.grf_kernel(self.adjacency_matrix, self.beta)
        indices_X1 = tf.cast(tf.reshape(X1, [-1]), dtype=tf.int32)
        indices_X2 = tf.cast(tf.reshape(X2, [-1]), dtype=tf.int32)
        return tf.gather(tf.gather(kernel_matrix, indices_X1, axis=0), indices_X2, axis=1)

    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        kernel_matrix = self.grf_kernel(self.adjacency_matrix, self.beta)
        indices_X = tf.cast(tf.reshape(X, [-1]), dtype=tf.int32)
        return tf.gather(tf.linalg.diag_part(kernel_matrix), indices_X)

    @staticmethod
    def get_normalized_laplacian(adj_matrix: tf.Tensor) -> tf.Tensor:
        """Calculate the normalized Laplacian of the adjacency matrix."""
        degrees = tf.reduce_sum(adj_matrix, axis=1)
        safe_degrees = tf.where(degrees > 0, degrees, tf.constant(float('inf'), dtype=adj_matrix.dtype))
        D_inv_sqrt = tf.linalg.diag(1.0 / tf.sqrt(safe_degrees))
        I = tf.eye(tf.shape(adj_matrix)[0], dtype=adj_matrix.dtype)
        return I - tf.matmul(tf.matmul(D_inv_sqrt, adj_matrix), D_inv_sqrt)

    def grf_kernel(self, adj_matrix: tf.Tensor, beta: gpflow.Parameter) -> tf.Tensor:
        """Construct the graph random features kernel."""
        # Get normalized Laplacian using TensorFlow operations
        laplacian = self.get_normalized_laplacian(adj_matrix)

        # Convert TensorFlow tensor to NumPy only when required
        laplacian_np = laplacian.numpy() if tf.executing_eagerly() else tf.convert_to_tensor(laplacian)

        graph = Graph(laplacian_np)

        grf_1 = GeneralGraphRandomFeatures(
            graph,
            modulation_function=diffusion_modulation_function,
            max_walk_length=self.max_walk_length,
            beta=beta.numpy() if tf.executing_eagerly() else tf.convert_to_tensor(beta),
            seed=42,
        )
        grf_2 = GeneralGraphRandomFeatures(
            graph,
            modulation_function=diffusion_modulation_function,
            max_walk_length=self.max_walk_length,
            beta=beta.numpy() if tf.executing_eagerly() else tf.convert_to_tensor(beta),
            seed=84,
        )

        # Generate feature matrices
        feature_matrix_1 = grf_1.generate_features(num_walks=self.walks_per_node, p_halt=self.p_halt)
        feature_matrix_2 = grf_2.generate_features(num_walks=self.walks_per_node, p_halt=self.p_halt)

        # Return kernel matrix as a TensorFlow tensor
        return tf.convert_to_tensor(feature_matrix_1 @ feature_matrix_2.T, dtype=tf.float64)
