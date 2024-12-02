import numpy as np

def get_normalized_laplacian(W):
    """
    Compute the normalized Laplacian matrix for a graph.

    Parameters:
    W (ndarray): Symmetric adjacency matrix of shape (n, n).

    Returns:
    ndarray: Normalized Laplacian matrix of shape (n, n).
    """
    degrees = np.sum(W, axis=1)
    # Handle nodes with zero degree
    safe_degrees = np.where(degrees > 0, degrees, 1.0)
    # Compute D^(-1/2) as a diagonal matrix
    D_inv_sqrt = np.diag(1.0 / np.sqrt(safe_degrees))
    # Compute normalized Laplacian
    L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    return L
