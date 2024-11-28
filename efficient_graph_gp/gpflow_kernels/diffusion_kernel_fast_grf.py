import gpflow
import tensorflow as tf
from ..random_walk_samplers import Graph, RandomWalk
from ..modulation_functions import diffusion_modulator_tf
from ..preprocessing import get_normalized_laplacian


class GraphDiffusionFastGRFKernel(gpflow.kernels.Kernel):
    def __init__(
        self,
        adjacency_matrix: tf.Tensor,
        walks_per_node: int = 50,
        p_halt: float = 0.1,
        max_walk_length: int = 10,
        beta: float = 1.0, # this is the learnable parameter
        **kwargs
    ):
        super().__init__(**kwargs)
        assert isinstance(adjacency_matrix, tf.Tensor), "Adjacency matrix must be a TensorFlow tensor."
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "Adjacency matrix must be square."

        self.adjacency_matrix = adjacency_matrix
        self.walks_per_node = walks_per_node
        self.p_halt = p_halt
        self.max_walk_length = max_walk_length
        self.beta = gpflow.Parameter(tf.cast(beta, tf.float32), transform=gpflow.utilities.positive())
        
        # Precompute random walk feature matrices
        self.laplacian = get_normalized_laplacian(adjacency_matrix)
        graph = Graph(self.laplacian)
        random_walk = RandomWalk(graph, seed=42)
        self.feature_matrices_tf = tf.constant(
            random_walk.get_random_walk_matrices(walks_per_node, p_halt, max_walk_length), dtype=tf.float32
        )

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

    def grf_kernel(self, beta: tf.Tensor) -> tf.Tensor:
        # Compute the modulator vector
        modulator_vector = diffusion_modulator_tf(
            tf.range(self.max_walk_length, dtype=tf.float32), beta
        )
        Phi = tf.linalg.matmul(self.feature_matrices_tf, modulator_vector[:, tf.newaxis])[:, :, 0]
        return tf.matmul(Phi, tf.transpose(Phi))