from scipy.linalg import expm
from .utils import get_normalized_laplacian
import numpy as np


def diffusion_kernel(adj_matrix, beta):
    """
    Compute the diffusion kernel matrix for a graph.
    """
    laplacian = get_normalized_laplacian(adj_matrix)
    # laplacian = np.diag(np.sum(adj_matrix, axis=1)) - adj_matrix 
    return expm(-beta * laplacian)  # Matrix exponential


