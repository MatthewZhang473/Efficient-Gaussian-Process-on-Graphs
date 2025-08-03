import torch
import gpytorch
from linear_operator.operators import SumLinearOperator, LinearOperator

class SparseGRFKernel(gpytorch.kernels.Kernel):
    def __init__(
        self,
        max_walk_length,
        step_matrices_torch,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Initialize learnable modulator vector as a parameter
        self.register_parameter(
            name="raw_modulator_vector", 
            parameter=torch.nn.Parameter(torch.randn(max_walk_length))
        )
        self.step_matrices = step_matrices_torch
        
    @property
    def modulator_vector(self):
        return self.raw_modulator_vector

    def forward(self, x1_idx = None, x2_idx = None, diag=False, **params):
        """
        Efficient Implementation of K[x1, x2], where K = Phi @ Phi^T
        """
        
        phi = self._get_feature_matrix()
        
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
            # diag(A @ B^T) = sum(A * B, dim=-1)
            return (phi_x1 * phi_x2).sum(dim=-1)
            
        else:
            # Return K[x1, x2] = Phi[x1, :] @ Phi[x2, :]^T
            return phi_x1 @ phi_x2.transpose(-1, -2)
        
    def _get_feature_matrix(self):
        """
        Returns the feature matrix Phi, the ith row is the GRF vector for the ith node.
        
        Ideally this should be lazy-evaluated linear operator.
        """
        # Build the combined matrix Phi = sum(modulator_vector[i] * step_matrices[i])
        # TODO: Check SumLinearOperator(...)
        phi = sum(
            mod_vec * mat for mod_vec, mat in zip(self.modulator_vector, self.step_matrices)
        )
        return phi