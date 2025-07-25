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
        adjacency_matrix,
        walks_per_node: int = 50,
        p_halt: float = 0.1,
        max_walk_length: int = 10,
        random_walk_seed: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Handle both sparse and dense adjacency matrices
        if sp.issparse(adjacency_matrix):
            adjacency_matrix = adjacency_matrix.tocsr()
            self.num_nodes = adjacency_matrix.shape[0]
        else:
            adjacency_matrix = sp.csr_matrix(adjacency_matrix)
            self.num_nodes = adjacency_matrix.shape[0]
        
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "Adjacency matrix must be square."

        self.walks_per_node = walks_per_node
        self.p_halt = p_halt
        self.max_walk_length = max_walk_length
        
        # Initialize learnable modulator vector as a parameter
        self.register_parameter(
            name="raw_modulator_vector", 
            parameter=torch.nn.Parameter(torch.randn(max_walk_length))
        )
        
        # Precompute random walk feature matrices and store as dense on CPU for conversion
        laplacian = get_normalized_laplacian(adjacency_matrix)
        random_walk = SparseRandomWalk(laplacian, seed=random_walk_seed)
        step_matrices_scipy = random_walk.get_random_walk_matrices(walks_per_node, p_halt, max_walk_length)
        
        # Store step matrices as dense tensors for MPS compatibility
        self.register_buffer('step_matrices_dense', 
                           torch.stack([torch.from_numpy(matrix.toarray()).float() 
                                      for matrix in step_matrices_scipy], dim=0))
        
        print(f"Kernel initialized with {len(step_matrices_scipy)} step matrices")
        print(f"Using device-aware dense tensors for MPS compatibility")

    @property
    def modulator_vector(self):
        return self.raw_modulator_vector

    def _get_device(self):
        """Helper method to get the device of the modulator vector"""
        return self.modulator_vector.device

    def forward(self, x1, x2, diag=False, **params):
        # Get current device
        device = self._get_device()
        
        # Move step matrices to the same device as modulator vector for computation
        step_matrices_device = self.step_matrices_dense.to(device)
        
        # Compute Phi = sum(f_p * M_p) using tensor operations on MPS
        # step_matrices_device: (max_walk_length, num_nodes, num_nodes)
        # modulator_vector: (max_walk_length,)
        weighted_matrices = step_matrices_device * self.modulator_vector.view(-1, 1, 1)
        Phi = torch.sum(weighted_matrices, dim=0)  # Sum over walk steps
        
        # Extract indices
        x1_idx = x1.long().flatten()
        x2_idx = x2.long().flatten()
        
        if diag:
            # Compute diagonal elements: diag(Phi @ Phi.T) = rowwise dot products
            Phi_rows = Phi[x1_idx]
            return torch.sum(Phi_rows * Phi_rows, dim=1)
        else:
            # Compute K[x1, x2] = Phi[x1] @ Phi[x2].T
            Phi_x1 = Phi[x1_idx]
            Phi_x2 = Phi[x2_idx]
            return torch.mm(Phi_x1, Phi_x2.t())