import torch
import gpytorch
import numpy as np
import sys
import os
import scipy.sparse as sp
from linear_operator.operators import ZeroLinearOperator

# Import sparse implementations
try:
    from ..random_walk_samplers_sparse import SparseRandomWalk
    from ..utils_sparse import get_normalized_laplacian
    from ..utils_sparse.sparse_lo import SparseLinearOperator
except ImportError:
    # For running directly or when relative imports fail
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from random_walk_samplers_sparse import SparseRandomWalk
    from utils_sparse import get_normalized_laplacian
    from utils_sparse.sparse_lo import SparseLinearOperator


class SparseGRFKernel(gpytorch.kernels.Kernel):
    def __init__(
        self,
        adjacency_matrix,  # Accept scipy sparse matrix only
        walks_per_node: int = 50,
        p_halt: float = 0.1,
        max_walk_length: int = 10,
        random_walk_seed: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "Adjacency matrix must be square."

        self.walks_per_node = walks_per_node
        self.p_halt = p_halt
        self.max_walk_length = max_walk_length
        self.num_nodes = adjacency_matrix.shape[0]
        
        # Initialize learnable modulator vector as a parameter
        self.register_parameter(
            name="raw_modulator_vector", 
            parameter=torch.nn.Parameter(torch.randn(max_walk_length))
        )
        
        # Precompute random walk feature matrices and convert to PyTorch sparse tensors
        laplacian = get_normalized_laplacian(adjacency_matrix)
        random_walk = SparseRandomWalk(laplacian, seed=random_walk_seed)
        step_matrices_scipy = random_walk.get_random_walk_matrices(walks_per_node, p_halt, max_walk_length)
        
        # Convert to Torch sparse linear operators
        self.step_matrices = []
        for csr_matrix in step_matrices_scipy:
            sparse_tensor = self._from_scipy_csr(csr_matrix)
            sparse_lo = SparseLinearOperator(sparse_tensor)
            self.step_matrices.append(sparse_lo)

    @property
    def modulator_vector(self):
        return self.raw_modulator_vector

    def forward(self, x1_idx = None, x2_idx = None, diag=False, **params):
        """
        Efficient Implementation of K[x1, x2], where K = Phi @ Phi^T
        """
        
        # Build the combined matrix Phi = sum(modulator_vector[i] * step_matrices[i])
        phi = ZeroLinearOperator(self.step_matrices[0].shape, dtype=torch.float32)
        for step, matrix in enumerate(self.step_matrices):
            phi += self.modulator_vector[step] * matrix
        
        # Handle indexing
        if x1_idx is not None:
            x1_idx = x1_idx.long().flatten()
            phi_x1 = phi[x1_idx]
        else:
            phi_x1 = phi
        if x2_idx is not None:
            x2_idx = x2_idx.long().flatten()
            phi_x2 = phi[x2_idx]
        else:
            phi_x2 = phi
            
        if diag:
            # Return diagonal as tensor
            raise NotImplementedError("Diagonal computation TBC.")
            
        else:
            # Return K[x1, x2] = Phi[x1, :] @ Phi[x2, :]^T
            return phi_x1 @ phi_x2.transpose(-1, -2)
        
    @staticmethod
    def _from_scipy_csr(scipy_csr: sp.csr_matrix) -> torch.sparse_csr_tensor:
        """
        Convert a scipy CSR matrix to a PyTorch sparse CSR tensor.
        """
        if not isinstance(scipy_csr, sp.csr_matrix):
            raise ValueError("Input must be a scipy CSR matrix")
        
        crow_indices = torch.from_numpy(scipy_csr.indptr).long()
        col_indices = torch.from_numpy(scipy_csr.indices).long()
        values = torch.from_numpy(scipy_csr.data).float()
        
        return torch.sparse_csr_tensor(
            crow_indices, col_indices, values,
            (scipy_csr.shape[0], scipy_csr.shape[1]),
            dtype=torch.float32
        )
