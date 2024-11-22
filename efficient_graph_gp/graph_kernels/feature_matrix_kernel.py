from .utils import get_normalized_laplacian
import numpy as np
from math import factorial


def feature_matrix_kernel(adj_matrix, max_expansion = 3, kernel_type = 'diffusion', kernel_hyperparameters = {'beta':2}):
    """
    Compute the feature matrix kernel for a graph.
    """
    if kernel_type == 'diffusion':
        beta = kernel_hyperparameters['beta']
        laplacian = get_normalized_laplacian(adj_matrix)
        modulation_coefficients = [(-beta/2)**i / factorial(i)for i in range(max_expansion+1)]
        L_basis = [np.linalg.matrix_power(laplacian, i) for i in range(max_expansion + 1)]
        K_f = sum(coeff * L for coeff, L in zip(modulation_coefficients, L_basis))
        return K_f @ K_f.T # Although K_f is symmetric        
        
    else:
        raise ValueError('Kernel type not recognized')