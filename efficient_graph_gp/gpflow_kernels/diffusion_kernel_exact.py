import gpflow
import tensorflow as tf


class GraphDiffusionKernel(gpflow.kernels.Kernel):
    def __init__(self, adjacency_matrix, **kwargs):
        super().__init__(**kwargs)
        self.adjacency_matrix = tf.convert_to_tensor(adjacency_matrix, dtype=tf.float64)
        self.beta = gpflow.Parameter(2.0, transform=gpflow.utilities.positive())  # Learnable hyperparameter

    def K(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        kernel_matrix = self.diffusion_kernel(self.adjacency_matrix, self.beta)
        indices_X1 = tf.cast(tf.reshape(X1, [-1]), dtype=tf.int32)
        indices_X2 = tf.cast(tf.reshape(X2, [-1]), dtype=tf.int32)
        return tf.gather(tf.gather(kernel_matrix, indices_X1, axis=0), indices_X2, axis=1)

    def K_diag(self, X):
        kernel_matrix = self.diffusion_kernel(self.adjacency_matrix, self.beta)
        indices_X = tf.cast(tf.reshape(X, [-1]), dtype=tf.int32)
        return tf.gather(tf.linalg.diag_part(kernel_matrix), indices_X)

    def diffusion_kernel(self, adj_matrix, beta):
        degrees = tf.reduce_sum(adj_matrix, axis=1)
        safe_degrees = tf.where(degrees > 0, degrees, tf.constant(float('inf'), dtype=adj_matrix.dtype))
        # Compute D^(-1/2)
        D_inv_sqrt = tf.linalg.diag(1.0 / tf.sqrt(safe_degrees))
        I = tf.eye(tf.shape(adj_matrix)[0], dtype=adj_matrix.dtype)
        normalized_laplacian = I - tf.matmul(tf.matmul(D_inv_sqrt, adj_matrix), D_inv_sqrt)
        return tf.linalg.expm(-beta * normalized_laplacian)
    
    # def diffusion_kernel(self, adj_matrix, beta):
    #     laplacian = tf.linalg.diag(tf.reduce_sum(adj_matrix, axis=1)) - adj_matrix  # Graph Laplacian
    #     return tf.linalg.expm(-beta * laplacian)