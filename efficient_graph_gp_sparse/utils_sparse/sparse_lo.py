import torch
from linear_operator.operators import LinearOperator
from linear_operator.operators.linear_operator_representation_tree import LinearOperatorRepresentationTree
from typing import Union, Optional, Tuple, List


class GRFLinearOperator(LinearOperator):
    """
    A LinearOperator for GRF kernels that extracts submatrices K[x1, x2].
    
    Represents K[x1, x2] where K = Phi @ Phi^T and x1, x2 are node indices.
    """
    
    def __init__(self, step_matrices: List[torch.Tensor], modulator_vector: torch.Tensor, 
                 x1_indices: torch.Tensor, x2_indices: torch.Tensor):
        self.step_matrices = step_matrices
        self.modulator_vector = modulator_vector
        self.num_nodes = step_matrices[0].shape[0]
        self.x1_indices = x1_indices.long().flatten()
        self.x2_indices = x2_indices.long().flatten()
        
        # Store for reconstruction
        self._step_matrices = step_matrices
        self._x1_indices = self.x1_indices.clone()
        self._x2_indices = self.x2_indices.clone()
        self._num_nodes = self.num_nodes
        
        # Only pass the modulator vector to avoid reconstruction issues
        super().__init__(modulator_vector)
    
    def _matmul(self, rhs: torch.Tensor) -> torch.Tensor:
        """Efficient matrix-vector multiplication: K[x1, x2] @ rhs"""
        batch_shape = rhs.shape[:-2]
        num_cols = rhs.shape[-1]
        n_x2 = len(self.x2_indices)
        
        # Flatten batch dimensions
        rhs_flat = rhs.view(-1, n_x2, num_cols)
        batch_size = rhs_flat.shape[0]
        
        results = []
        for b in range(batch_size):
            rhs_b = rhs_flat[b]
            
            # Expand to full node space
            rhs_full = torch.zeros(self.num_nodes, num_cols, device=rhs.device)
            rhs_full[self.x2_indices] = rhs_b
            
            # Compute Phi^T @ rhs_full
            phi_t_rhs = torch.zeros_like(rhs_full)
            for step, matrix in enumerate(self.step_matrices):
                if step < len(self.modulator_vector):
                    weight = self.modulator_vector[step]
                    phi_t_rhs += weight * torch.sparse.mm(matrix.transpose(-2, -1), rhs_full)
            
            # Compute Phi @ (Phi^T @ rhs_full)
            result_full = torch.zeros_like(phi_t_rhs)
            for step, matrix in enumerate(self.step_matrices):
                if step < len(self.modulator_vector):
                    weight = self.modulator_vector[step]
                    result_full += weight * torch.sparse.mm(matrix, phi_t_rhs)
            
            # Extract rows for x1_indices
            results.append(result_full[self.x1_indices])
        
        result = torch.stack(results, dim=0)
        return result.view(*batch_shape, len(self.x1_indices), num_cols)
    
    def _size(self) -> torch.Size:
        batch_shape = self.modulator_vector.shape[:-1]
        return batch_shape + torch.Size([len(self.x1_indices), len(self.x2_indices)])
    
    def _transpose_nonbatch(self) -> "GRFLinearOperator":
        return GRFLinearOperator(
            self.step_matrices, self.modulator_vector, 
            self.x2_indices, self.x1_indices
        )
    
    def _diagonal(self) -> torch.Tensor:
        """Extract diagonal elements efficiently"""
        if not torch.equal(self.x1_indices, self.x2_indices):
            raise RuntimeError("Diagonal only defined for square matrices with identical indices")
        
        batch_shape = self.modulator_vector.shape[:-1]
        diagonal = torch.zeros(*batch_shape, len(self.x1_indices), device=self.modulator_vector.device)
        
        for i, node_idx in enumerate(self.x1_indices):
            # Create unit vector for this node
            e_i = torch.zeros(self.num_nodes, 1, device=self.modulator_vector.device)
            e_i[node_idx, 0] = 1.0
            
            # Compute Phi^T @ e_i
            phi_t_ei = torch.zeros_like(e_i)
            for step, matrix in enumerate(self.step_matrices):
                if step < len(self.modulator_vector):
                    weight = self.modulator_vector[step]
                    phi_t_ei += weight * torch.sparse.mm(matrix.transpose(-2, -1), e_i)
            
            # Compute Phi @ (Phi^T @ e_i)
            phi_ei = torch.zeros_like(phi_t_ei)
            for step, matrix in enumerate(self.step_matrices):
                if step < len(self.modulator_vector):
                    weight = self.modulator_vector[step]
                    phi_ei += weight * torch.sparse.mm(matrix, e_i)
            
            # Diagonal element is e_i^T @ Phi @ (Phi^T @ e_i)
            diagonal[..., i] = (phi_ei.squeeze(-1) * phi_t_ei.squeeze(-1)).sum()
        
        return diagonal
    
    def _bilinear_derivative(self, left_vecs: torch.Tensor, right_vecs: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Compute derivatives for backpropagation through the modulator vector.
        """
        derivatives = []
        
        for step, matrix in enumerate(self.step_matrices):
            if step < len(self.modulator_vector):
                # Expand left_vecs and right_vecs to full node space
                left_full = torch.zeros(self.num_nodes, left_vecs.shape[-1], device=left_vecs.device)
                right_full = torch.zeros(self.num_nodes, right_vecs.shape[-1], device=right_vecs.device)
                left_full[self.x1_indices] = left_vecs
                right_full[self.x2_indices] = right_vecs
                
                # Compute derivative w.r.t. modulator_vector[step]
                ps_right = torch.sparse.mm(matrix, right_full)
                pst_left = torch.sparse.mm(matrix.transpose(-2, -1), left_full)
                
                # Two terms: left^T @ P_s @ P_s^T @ right + left^T @ P_s^T @ P_s @ right
                term1 = (left_full * ps_right).sum()
                term2 = (pst_left * torch.sparse.mm(matrix, right_full)).sum()
                
                derivatives.append(term1 + term2)
            else:
                derivatives.append(torch.tensor(0.0, device=self.modulator_vector.device))
        
        modulator_grad = torch.stack(derivatives)
        return (modulator_grad,)
    
    def _clone_with_different_tensors(self, *tensors):
        """Handle cloning with different tensors"""
        if len(tensors) == 1:
            # Create new instance with the new modulator_vector
            return GRFLinearOperator(
                self._step_matrices, 
                tensors[0], 
                self._x1_indices, 
                self._x2_indices
            )
        else:
            raise NotImplementedError(f"Unexpected number of tensors for cloning: {len(tensors)}")
    
    def representation(self):
        """Return only the differentiable tensor"""
        return (self.modulator_vector,)
    
    def representation_tree(self):
        """Return a proper representation tree for reconstruction"""
        return LinearOperatorRepresentationTree(
            self.__class__,
            # Pass the stored references for reconstruction
            step_matrices=self._step_matrices,
            x1_indices=self._x1_indices,
            x2_indices=self._x2_indices,
            num_nodes=self._num_nodes
        )
    
    @classmethod  
    def _make_linear_operator(cls, modulator_vector, **kwargs):
        """Class method to create the LinearOperator from representation"""
        return cls(
            kwargs['step_matrices'],
            modulator_vector,
            kwargs['x1_indices'], 
            kwargs['x2_indices']
        )
    
    def to_dense(self) -> torch.Tensor:
        """Override to_dense to avoid reconstruction issues"""
        # Create identity matrix and use our _matmul directly
        I = torch.eye(len(self.x2_indices), device=self.modulator_vector.device)
        return self._matmul(I)