import gpflow
import numpy as np
import tensorflow as tf
from ..random_walk_samplers import Graph
from ..preprocessing import get_normalized_laplacian, get_laplacian

def compute_pstep_walk_matrix(adj_matrix, p_max):
    """
    Given a (possibly weighted) adjacency matrix (adj_matrix) of shape (N, N),
    compute a 3D walk matrix of shape (N, N, p_max+1),
    where walk_matrix[i, j, p] is the sum of weights of all walks of length p
    from node i to node j.

    Indices:
      p = 0 => Identity matrix (length-0 walks)
      p = 1 => A (length-1 walks)
      ...
      p = p_max => A^p_max (length-p_max walks)

    Parameters
    ----------
    adj_matrix : numpy.ndarray of shape (N, N)
        Weighted or unweighted adjacency matrix (float).
    p_max : int
        Maximum path length to consider.

    Returns
    -------
    walk_matrix : numpy.ndarray of shape (N, N, p_max+1)
        3D array where walk_matrix[:, :, p] = A^p.
    """

    N = adj_matrix.shape[0]
    walk_matrix = np.zeros((N, N, p_max), dtype=np.float64)
    walk_matrix[:, :, 0] = np.eye(N, dtype=np.float64)

    current_power = np.eye(N, dtype=np.float64)
    for p in range(1, p_max):
        current_power = current_power @ adj_matrix
        walk_matrix[:, :, p] = current_power

    return walk_matrix


class GraphGeneralPoFMKernel(gpflow.kernels.Kernel):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        max_walk_length: int = 10,
        modulator_vector: np.ndarray = None,
        normalize_laplacian: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "Adjacency matrix must be square."
        self.adjacency_matrix = adjacency_matrix
        self.max_walk_length = max_walk_length
        
        if modulator_vector is None:
            # initialize the modulator vector as a random vector, learnable
            np.random.seed(42)
            initial_modulator_vector = np.random.randn(max_walk_length)
            self.modulator_vector =  gpflow.Parameter(tf.cast(initial_modulator_vector, tf.float64)) #TODO: Check Initialization & Constraint
        else:
            # initialize the modulator vector as a given vector
            if len(modulator_vector) != max_walk_length:
                raise ValueError("The length of the modulator vector must be equal to the max_walk_length.")
            else:
                self.modulator_vector =  gpflow.Parameter(tf.cast(modulator_vector, tf.float64))
        
        # Precompute feature matrices
        self.laplacian = get_normalized_laplacian(self.adjacency_matrix) if normalize_laplacian else get_laplacian(self.adjacency_matrix)
        p_step_walk_matrix = compute_pstep_walk_matrix(self.laplacian, self.max_walk_length)
        self.feature_matrices_tf = tf.constant(
            p_step_walk_matrix, dtype=tf.float64
        )

    def K(self, X1: tf.Tensor, X2: tf.Tensor = None) -> tf.Tensor:
        if X2 is None:
            X2 = X1
        kernel_matrix = self.pofm_kernel(self.modulator_vector)
        indices_X1 = tf.cast(tf.reshape(X1, [-1]), dtype=tf.int32)
        indices_X2 = tf.cast(tf.reshape(X2, [-1]), dtype=tf.int32)
        return tf.gather(tf.gather(kernel_matrix, indices_X1, axis=0), indices_X2, axis=1)

    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        kernel_matrix = self.pofm_kernel(self.modulator_vector)
        indices_X = tf.cast(tf.reshape(X, [-1]), dtype=tf.int32)
        return tf.gather(tf.linalg.diag_part(kernel_matrix), indices_X)

    def pofm_kernel(self, modulator_vector: tf.Tensor) -> tf.Tensor:
        Phi = tf.linalg.matmul(self.feature_matrices_tf, modulator_vector[:, tf.newaxis])[:, :, 0]
        return tf.matmul(Phi, tf.transpose(Phi))