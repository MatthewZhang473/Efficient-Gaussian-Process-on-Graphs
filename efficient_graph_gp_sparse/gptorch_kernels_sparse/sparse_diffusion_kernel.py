import torch
import gpytorch
from gpytorch.constraints import Positive


def diffusion_modulator_torch(length: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Compute diffusion modulator vector: (-beta)^length / (2^length * Gamma(length + 1))

    Args:
        length: Tensor of walk lengths (typically 0 to max_walk_length-1)
        beta: Learnable diffusion parameter

    Returns:
        Modulator vector of same shape as length
    """
    # Make sure dtype and device follow beta
    length = length.to(dtype=beta.dtype, device=beta.device)

    numerator = torch.pow(-beta, length)
    denominator = torch.pow(torch.tensor(2.0, dtype=beta.dtype, device=beta.device), length)
    denominator = denominator * torch.exp(torch.lgamma(length + 1.0))

    return numerator / denominator


class SparseDiffusionKernel(gpytorch.kernels.Kernel):
    def __init__(self, max_walk_length, step_matrices_torch, **kwargs):
        super().__init__(**kwargs)

        # Register parameters with positivity constraints
        self.register_parameter(
            name="raw_beta",
            parameter=torch.nn.Parameter(torch.tensor(1.0))
        )
        self.register_constraint("raw_beta", Positive())

        self.register_parameter(
            name="raw_sigma_f",
            parameter=torch.nn.Parameter(torch.tensor(1.0))
        )
        self.register_constraint("raw_sigma_f", Positive())

        self.step_matrices = step_matrices_torch
        self.max_walk_length = max_walk_length

    @property
    def beta(self):
        return self.raw_beta_constraint.transform(self.raw_beta)

    @property
    def sigma_f(self):
        return self.raw_sigma_f_constraint.transform(self.raw_sigma_f)

    @property
    def modulator_vector(self):
        """Compute the diffusion modulator vector using the diffusion formula"""
        walk_lengths = torch.arange(
            self.max_walk_length,
            dtype=self.raw_beta.dtype,
            device=self.raw_beta.device
        )
        return self.sigma_f * diffusion_modulator_torch(walk_lengths, self.beta)

    def forward(self, x1_idx=None, x2_idx=None, diag=False, **params):
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
            return (phi_x1 * phi_x2).sum(dim=-1)
        else:
            return (phi_x1 @ phi_x2.transpose(-1, -2))

    def _get_feature_matrix(self):
        """
        Returns the feature matrix Phi, the ith row is the GRF vector for the ith node.
        """
        # Dense sum of weighted step matrices
        phi = sum(
            mod_vec * mat for mod_vec, mat in zip(self.modulator_vector, self.step_matrices)
        )
        return phi
