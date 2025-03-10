import gpflow
import tensorflow as tf
from math import factorial
import numpy as np
from ..preprocessing import get_normalized_laplacian, get_laplacian

class GraphDiffusionPoFMKernel(gpflow.kernels.Kernel):
    def __init__(self, adjacency_matrix, beta = 2.0, max_expansion=5, sigma_f=1.0, normalize_laplacian=True, **kwargs):
        super().__init__(**kwargs)
        self.adjacency_matrix = tf.convert_to_tensor(adjacency_matrix, dtype=tf.float64)
        self.beta = gpflow.Parameter(beta, transform=gpflow.utilities.positive())  # Learnable hyperparameter
        self.max_expansion = max_expansion  # Maximum expansion for series approximation
        self.sigma_f = gpflow.Parameter(sigma_f, transform=gpflow.utilities.positive())  # Learnable hyperparameter
        self.normalize_laplacian = normalize_laplacian

    def K(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        kernel_matrix = self.compute_diffusion_kernel(self.adjacency_matrix, self.beta, self.max_expansion)
        indices_X1 = tf.cast(tf.reshape(X1, [-1]), dtype=tf.int32)
        indices_X2 = tf.cast(tf.reshape(X2, [-1]), dtype=tf.int32)
        return tf.gather(tf.gather(kernel_matrix, indices_X1, axis=0), indices_X2, axis=1)

    def K_diag(self, X):
        kernel_matrix = self.compute_diffusion_kernel(self.adjacency_matrix, self.beta, self.max_expansion)
        indices_X = tf.cast(tf.reshape(X, [-1]), dtype=tf.int32)
        return tf.gather(tf.linalg.diag_part(kernel_matrix), indices_X)

    def compute_diffusion_kernel(self, adj_matrix, beta, max_expansion):
        
        laplacian = get_normalized_laplacian(adj_matrix) if self.normalize_laplacian else get_laplacian(adj_matrix)
        modulation_coefficients = [(-beta / 2) ** i / factorial(i) for i in range(max_expansion + 1)]
        L_basis = [tf.eye(tf.shape(laplacian)[0], dtype=laplacian.dtype)]
        for _ in range(1, max_expansion + 1):
            L_basis.append(tf.matmul(L_basis[-1], laplacian))
        K_f = sum(coeff * L for coeff, L in zip(modulation_coefficients, L_basis))
        return self.sigma_f**2 * tf.matmul(K_f, tf.transpose(K_f))  # Ensure symmetry of the kernel
    
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
    kernel = GraphDiffusionPoFMKernel(adjacency_matrix, max_expansion=8)

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

