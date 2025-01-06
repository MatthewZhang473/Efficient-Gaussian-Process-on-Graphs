import gpflow
import tensorflow as tf
import numpy as np


class GraphDiffusionKernel(gpflow.kernels.Kernel):
    def __init__(self, adjacency_matrix, beta = None, **kwargs):
        super().__init__(**kwargs)
        self.adjacency_matrix = tf.convert_to_tensor(adjacency_matrix, dtype=tf.float64)
        if beta:
            self.beta = gpflow.Parameter(beta, transform=gpflow.utilities.positive())
        else:
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
        normalized_laplacian = self.get_normalized_laplacian(adj_matrix)
        return tf.linalg.expm(-beta * normalized_laplacian)
    
    @staticmethod
    def get_normalized_laplacian(adj_matrix):
        degrees = tf.reduce_sum(adj_matrix, axis=1)
        safe_degrees = tf.where(degrees > 0, degrees, tf.constant(float('inf'), dtype=adj_matrix.dtype))
        D_inv_sqrt = tf.linalg.diag(1.0 / tf.sqrt(safe_degrees))
        I = tf.eye(tf.shape(adj_matrix)[0], dtype=adj_matrix.dtype)
        return I - tf.matmul(tf.matmul(D_inv_sqrt, adj_matrix), D_inv_sqrt)
    
  
if __name__ == "__main__":

    # Define a simple adjacency matrix for testing
    adjacency_matrix = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]
    ], dtype=np.float64)

    # Define test input data points (indices corresponding to graph nodes)
    X_test = np.array([[0], [1], [2], [3]])

    # Initialize the kernel
    kernel = GraphDiffusionKernel(adjacency_matrix)

    # Compute the full kernel matrix
    kernel_matrix = kernel.K(X_test)

    # Compute the diagonal of the kernel matrix
    kernel_diag = kernel.K_diag(X_test)

    # Print results
    print("Adjacency Matrix:")
    print(adjacency_matrix)

    print("\nKernel Matrix (K):")
    print(kernel_matrix)

    print("\nDiagonal of Kernel Matrix (K_diag):")
    print(kernel_diag)
