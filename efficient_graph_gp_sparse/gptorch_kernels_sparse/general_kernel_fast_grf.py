import torch
import gpytorch
import numpy as np
import sys
import os
import scipy.sparse as sp

# Import sparse implementations
try:
    from ..random_walk_samplers_sparse import SparseRandomWalk
    from ..utils_sparse import get_normalized_laplacian
except ImportError:
    # For running directly or when relative imports fail
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from random_walk_samplers_sparse import SparseRandomWalk
    from utils_sparse import get_normalized_laplacian


class GraphGeneralFastGRFKernel(gpytorch.kernels.Kernel):
    def __init__(
        self,
        adjacency_matrix,  # Acceptcscipy sparse matrix only
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
        # Compute Phi = sum(f_p * M_p) using PyTorch sparse operations to maintain gradients
        Phi = None
        for step, matrix in enumerate(self.step_matrices):
            if step < len(self.modulator_vector):
                weighted_matrix = matrix * self.modulator_vector[step]
                if Phi is None:
                    Phi = weighted_matrix
                else:
                    Phi = Phi + weighted_matrix
        
        # Move to correct device
        if Phi is not None:
            Phi = Phi.to(device=self.modulator_vector.device)
        
        # Extract indices
        x1_idx = x1.long().flatten()
        x2_idx = x2.long().flatten()
        
        if diag:
            # Compute diagonal elements: diag(Phi @ Phi.T) = rowwise dot products
            Phi_dense = Phi.to_dense()
            Phi_rows = Phi_dense[x1_idx]
            return torch.sum(Phi_rows * Phi_rows, dim=1)
        else:
            # Compute K[x1, x2] = Phi[x1] @ Phi[x2].T
            Phi_dense = Phi.to_dense()
            Phi_x1 = Phi_dense[x1_idx]
            Phi_x2 = Phi_dense[x2_idx]
            return torch.mm(Phi_x1, Phi_x2.t())