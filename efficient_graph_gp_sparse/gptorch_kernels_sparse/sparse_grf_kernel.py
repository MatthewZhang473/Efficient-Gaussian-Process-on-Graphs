import torch
import gpytorch
import numpy as np
import sys
import os
import scipy.sparse as sp
from linear_operator import LinearOperator

# Import sparse implementations
try:
    from ..random_walk_samplers_sparse import SparseRandomWalk
    from ..utils_sparse import get_normalized_laplacian
    from ..utils_sparse.sparse_lo import GRFLinearOperator
except ImportError:
    # For running directly or when relative imports fail
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from random_walk_samplers_sparse import SparseRandomWalk
    from utils_sparse import get_normalized_laplacian
    from utils_sparse.sparse_lo import GRFLinearOperator


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
        
        # Convert to PyTorch sparse tensors once in __init__
        self.step_matrices = []
        for matrix in step_matrices_scipy:
            crow_indices = torch.from_numpy(matrix.indptr).long()
            col_indices = torch.from_numpy(matrix.indices).long()
            values = torch.from_numpy(matrix.data).float()
            
            sparse_tensor = torch.sparse_csr_tensor(
                crow_indices, col_indices, values,
                (self.num_nodes, self.num_nodes),
                dtype=torch.float32
            )
            self.step_matrices.append(sparse_tensor)

    @property
    def modulator_vector(self):
        return self.raw_modulator_vector

    def forward(self, x1, x2, diag=False, **params):
        """
        Forward pass using GRFLinearOperator for efficient computation.
        
        Args:
            x1, x2: Node indices tensors 
            diag: If True, return diagonal as tensor; if False, return LinearOperator
            
        Returns:
            Kernel matrix K[x1, x2] as LinearOperator when diag=False, or diagonal tensor when diag=True
        """
        # Convert x1, x2 to node indices
        x1_idx = x1.long().flatten()
        x2_idx = x2.long().flatten()
        
        # Create GRF LinearOperator that represents K[x1, x2]
        kernel_op = GRFLinearOperator(self.step_matrices, self.modulator_vector, x1_idx, x2_idx)
        
        if diag:
            return kernel_op.diagonal(dim1=-2, dim2=-1)
        else:
            return kernel_op