import gpflow
import tensorflow as tf
import math

def diffusion_modulation_function(length: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
    length = tf.cast(length, tf.float64)
    beta = tf.cast(beta, tf.float64)
    two = tf.constant(2.0, dtype=tf.float64)
    numerator = tf.pow(-beta, length)
    denominator = tf.pow(two, length) * tf.exp(tf.math.lgamma(length + 1.0))
    return numerator / denominator

class Graph:
    def __init__(self, adjacency_matrix: tf.Tensor):
        assert isinstance(adjacency_matrix, tf.Tensor), "Adjacency matrix must be a TensorFlow tensor."
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "Adjacency matrix must be square."
        self.adjacency_matrix = adjacency_matrix
        self.num_nodes = tf.shape(adjacency_matrix)[0]

    def get_neighbors(self, node: tf.Tensor) -> tf.Tensor:
        # Returns the indices of neighbors for a given node
        neighbors = tf.where(self.adjacency_matrix[node, :] != 0)
        neighbors = tf.reshape(neighbors, [-1])
        neighbors = tf.cast(neighbors, dtype=tf.int32)  # Cast to tf.int32
        return neighbors

    def get_num_nodes(self) -> tf.Tensor:
        return self.num_nodes

    def get_edge_weight(self, node1: tf.Tensor, node2: tf.Tensor) -> tf.Tensor:
        node1 = tf.cast(node1, tf.int32)
        node2 = tf.cast(node2, tf.int32)
        return self.adjacency_matrix[node1, node2]


class RandomWalk:
    def __init__(self, graph: Graph, seed: int = None):
        self.graph = graph
        if seed is not None:
            tf.random.set_seed(seed)

    def _perform_walk(
        self, start_node: tf.Tensor, modulation_function, p_halt: tf.Tensor
    ) -> tf.Tensor:
        current_node = start_node
        walk_length = tf.constant(0, dtype=tf.int32)
        load = tf.constant(1.0, dtype=tf.float64)
        num_nodes = self.graph.get_num_nodes()
        feature_vector = tf.zeros([num_nodes], dtype=tf.float64)

        proceed = tf.constant(True, dtype=tf.bool)

        def condition(current_node, walk_length, load, feature_vector, proceed):
            return proceed

        def body(current_node, walk_length, load, feature_vector, proceed):
            modulation = modulation_function(tf.cast(walk_length, tf.float64))
            indices = tf.reshape(current_node, [1])
            updates = tf.reshape(load * modulation, [1])
            feature_vector = tf.tensor_scatter_nd_add(feature_vector, tf.expand_dims(indices, axis=1), updates)
            walk_length += 1

            neighbors = self.graph.get_neighbors(current_node)
            num_neighbors = tf.shape(neighbors)[0]

            def no_neighbors():
                return current_node, walk_length, load, feature_vector, tf.constant(False)

            def has_neighbors():
                random_index = tf.random.uniform([], minval=0, maxval=num_neighbors, dtype=tf.int32)
                new_node = neighbors[random_index]
                weight = self.graph.get_edge_weight(current_node, new_node)
                load_new = load * tf.cast(num_neighbors, tf.float64) / (1.0 - p_halt) * weight
                current_node_new = new_node

                rand_val = tf.random.uniform([], dtype=tf.float64)
                proceed_new = tf.greater_equal(rand_val, p_halt)

                return current_node_new, walk_length, load_new, feature_vector, proceed_new

            current_node, walk_length, load, feature_vector, proceed = tf.cond(
                tf.greater(num_neighbors, 0),
                true_fn=has_neighbors,
                false_fn=no_neighbors
            )

            return current_node, walk_length, load, feature_vector, proceed

        max_iterations = self.graph.num_nodes * 10  # Set a reasonable maximum to prevent infinite loops
        current_node, walk_length, load, feature_vector, proceed = tf.while_loop(
            condition,
            body,
            [current_node, walk_length, load, feature_vector, proceed],
            maximum_iterations=max_iterations
        )

        return feature_vector

    def calculate_feature_vector(
        self, start_node: tf.Tensor, num_walks: int, modulation_function, p_halt: tf.Tensor
    ) -> tf.Tensor:
        num_nodes = self.graph.get_num_nodes()
        feature_vector = tf.zeros([num_nodes], dtype=tf.float64)

        def body(i, feature_vector):
            fv = self._perform_walk(start_node, modulation_function, p_halt)
            feature_vector += fv
            return i + 1, feature_vector

        _, feature_vector = tf.while_loop(
            lambda i, fv: i < num_walks,
            body,
            [tf.constant(0), feature_vector]
        )

        return feature_vector / tf.cast(num_walks, tf.float64)

class GeneralGraphRandomFeatures:
    def __init__(
        self,
        graph: Graph,
        modulation_function,
        max_walk_length: int = 10,
        beta: tf.Tensor = 1.0,
        seed: int = None,
    ):
        self.graph = graph
        self.random_walk = RandomWalk(graph, seed=seed)
        self.modulation_function = modulation_function
        self.max_walk_length = max_walk_length
        self.beta = tf.cast(beta, tf.float64)  # Ensure beta is tf.float64

    def generate_features(self, num_walks: int = 50, p_halt: float = 0.1) -> tf.Tensor:
        num_nodes = self.graph.get_num_nodes()
        feature_matrix = tf.TensorArray(dtype=tf.float64, size=num_nodes)

        def body(node, feature_matrix):
            feature_vector = self.random_walk.calculate_feature_vector(
                start_node=node,
                num_walks=num_walks,
                modulation_function=lambda length: self.modulation_function(length, self.beta),
                p_halt=tf.constant(p_halt, dtype=tf.float64),
            )
            feature_matrix = feature_matrix.write(node, feature_vector)
            return node + 1, feature_matrix

        _, feature_matrix = tf.while_loop(
            lambda node, _: node < num_nodes,
            body,
            [tf.constant(0), feature_matrix]
        )

        feature_matrix = feature_matrix.stack()
        return feature_matrix

class GraphDiffusionGRFKernel(gpflow.kernels.Kernel):
    def __init__(
        self,
        adjacency_matrix: tf.Tensor,
        walks_per_node: int = 50,
        p_halt: float = 0.1,
        max_walk_length: int = 10,
        beta: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert isinstance(adjacency_matrix, tf.Tensor), "Adjacency matrix must be a TensorFlow tensor."
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "Adjacency matrix must be square."

        self.adjacency_matrix = adjacency_matrix
        self.walks_per_node = walks_per_node
        self.p_halt = p_halt
        self.max_walk_length = max_walk_length
        self.beta = gpflow.Parameter(tf.cast(beta, tf.float64), transform=gpflow.utilities.positive())

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
        safe_degrees = tf.where(degrees > 0, degrees, tf.ones_like(degrees))
        D_inv_sqrt = tf.linalg.diag(1.0 / tf.sqrt(safe_degrees))
        I = tf.eye(tf.shape(adj_matrix)[0], dtype=adj_matrix.dtype)
        L = I - tf.matmul(tf.matmul(D_inv_sqrt, adj_matrix), D_inv_sqrt)
        return L

    def grf_kernel(self, adj_matrix: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
        # Get normalized Laplacian using TensorFlow operations
        laplacian = self.get_normalized_laplacian(adj_matrix)
        graph = Graph(laplacian)
        
        # Ensure beta is of type tf.float64
        beta = tf.cast(beta, tf.float64)
        
        grf_1 = GeneralGraphRandomFeatures(
            graph,
            modulation_function=diffusion_modulation_function,
            max_walk_length=self.max_walk_length,
            beta=beta,
            seed=42,
        )
        grf_2 = GeneralGraphRandomFeatures(
            graph,
            modulation_function=diffusion_modulation_function,
            max_walk_length=self.max_walk_length,
            beta=beta,
            seed=84,
        )

        # Generate feature matrices
        feature_matrix_1 = grf_1.generate_features(num_walks=self.walks_per_node, p_halt=self.p_halt)
        feature_matrix_2 = grf_2.generate_features(num_walks=self.walks_per_node, p_halt=self.p_halt)

        # Return kernel matrix
        return tf.matmul(feature_matrix_1, tf.transpose(feature_matrix_2))

