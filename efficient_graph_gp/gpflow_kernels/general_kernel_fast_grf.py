import gpflow
import numpy as np
import tensorflow as tf
from ..random_walk_samplers import Graph, RandomWalk
from ..modulation_functions import diffusion_modulator_tf
from ..preprocessing import get_normalized_laplacian


class GraphGeneralFastGRFKernel(gpflow.kernels.Kernel):
    def __init__(
        self,
        adjacency_matrix: tf.Tensor,
        walks_per_node: int = 50,
        p_halt: float = 0.1,
        max_walk_length: int = 10,
        random_walk_seed: int = 42,
        modulator_vector: np.ndarray = None,
        step_matrices: np.ndarray = None,  # NEW: pre-computed step matrices
        use_tqdm: bool = False,
        ablation: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "Adjacency matrix must be square."

        self.adjacency_matrix = adjacency_matrix
        self.walks_per_node = walks_per_node
        self.p_halt = p_halt
        self.max_walk_length = max_walk_length
        
        if modulator_vector is None:
            # initialize the modulator vector as a random vector, learnable
            np.random.seed(42)
            initial_modulator_vector = np.random.randn(max_walk_length)
            self.modulator_vector = gpflow.Parameter(tf.cast(initial_modulator_vector, tf.float64))
        else:
            # initialize the modulator vector as a given vector
            if len(modulator_vector) != max_walk_length:
                raise ValueError("The length of the modulator vector must be equal to the max_walk_length.")
            else:
                self.modulator_vector = gpflow.Parameter(tf.cast(modulator_vector, tf.float64))
        
        # Handle step matrices - either load pre-computed or compute new ones
        if step_matrices is not None:
            self.feature_matrices_tf = tf.constant(step_matrices, dtype=tf.float64)
        elif ablation:
            graph = Graph(adjacency_matrix)
            random_walk = RandomWalk(graph, seed=random_walk_seed)
            self.feature_matrices_tf = tf.constant(
                random_walk.get_random_walk_matrices(walks_per_node, p_halt, max_walk_length, use_tqdm=use_tqdm, ablation=ablation), dtype=tf.float64
            )
        else:
            # Precompute random walk feature matrices
            self.laplacian = get_normalized_laplacian(adjacency_matrix)
            graph = Graph(self.laplacian)
            random_walk = RandomWalk(graph, seed=random_walk_seed)
            self.feature_matrices_tf = tf.constant(
                random_walk.get_random_walk_matrices(walks_per_node, p_halt, max_walk_length, use_tqdm=use_tqdm), dtype=tf.float64
            )

    def K(self, X1: tf.Tensor, X2: tf.Tensor = None) -> tf.Tensor:
        if X2 is None:
            X2 = X1
        kernel_matrix = self.grf_kernel(self.modulator_vector)
        indices_X1 = tf.cast(tf.reshape(X1, [-1]), dtype=tf.int32)
        indices_X2 = tf.cast(tf.reshape(X2, [-1]), dtype=tf.int32)
        return tf.gather(tf.gather(kernel_matrix, indices_X1, axis=0), indices_X2, axis=1)

    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        kernel_matrix = self.grf_kernel(self.modulator_vector)
        indices_X = tf.cast(tf.reshape(X, [-1]), dtype=tf.int32)
        return tf.gather(tf.linalg.diag_part(kernel_matrix), indices_X)

    def grf_kernel(self, modulator_vector: tf.Tensor) -> tf.Tensor:
        # Ensure consistency in data types
        Phi = tf.linalg.matmul(self.feature_matrices_tf, modulator_vector[:, tf.newaxis])[:, :, 0]
        return tf.matmul(Phi, tf.transpose(Phi))
