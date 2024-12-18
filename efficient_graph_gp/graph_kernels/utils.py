import numpy as np
import scipy.sparse as sp

#TODO: Test the correctness of the sparse implementation of get_normalized_laplacian

def get_normalized_laplacian(W, sparse=False):
    """
    Compute the normalized Laplacian matrix for a graph.

    Parameters:
    W (ndarray or sparse matrix): Symmetric adjacency matrix of shape (n, n).
    sparse (bool): Flag indicating whether the input graph is sparse. Default is False.

    Returns:
    ndarray or sparse matrix: Normalized Laplacian matrix of shape (n, n).
    """
    if sparse:
        degrees = np.array(W.sum(axis=1)).flatten()
        D_inv_sqrt = sp.diags([1.0 / np.sqrt(degrees[i]) if degrees[i] > 0 else 0.0 for i in range(len(degrees))])
        L = sp.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        degrees = np.sum(W, axis=1)
        D_inv_sqrt = np.zeros_like(W, dtype=float)
        valid_degrees = degrees > 0
        D_inv_sqrt[valid_degrees, valid_degrees] = 1.0 / np.sqrt(degrees[valid_degrees])
        L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt

    return L

def generate_noisy_samples(K, noise_std=0.1):
    """
    Generate noisy samples from a Gaussian process.
    
    Parameters:
    K (ndarray): Covariance matrix of shape (n, n).
    noise_std (float): Standard deviation of the additive noise.
    
    Returns:
    ndarray: Noisy samples of shape (n, 1).
    """
    
    num_nodes = K.shape[0]
    L = np.linalg.cholesky(K + 1e-6 * np.eye(num_nodes))  # Cholesky decomposition
    true_samples = L @ np.random.normal(size=(num_nodes, 1))  # Sample from Gaussian process
    noise = noise_std * np.random.randn(num_nodes, 1)  # Additive noise
    Y_noisy = true_samples + noise  # Noisy observations
    return Y_noisy