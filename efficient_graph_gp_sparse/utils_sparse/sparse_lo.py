import torch
from linear_operator.operators import LinearOperator

class SparseLinearOperator(LinearOperator):
    """
    A LinearOperator that wraps a sparse CSR tensor and performs
    sparse matrix @ dense tensor operations efficiently.
    """

    def __init__(self, sparse_csr_tensor):
        if not sparse_csr_tensor.is_sparse_csr:
            raise ValueError("Input tensor must be a sparse CSR tensor")
        self.sparse_csr_tensor = sparse_csr_tensor
        super().__init__(sparse_csr_tensor)

    def _matmul(self, rhs):
        # Use tensor.matmul for CSR tensors
        return self.sparse_csr_tensor.matmul(rhs)

    def _size(self):
        return self.sparse_csr_tensor.size()

    def _transpose_nonbatch(self):
        # CSR → COO
        coo = self.sparse_csr_tensor.to_sparse_coo()
        idx = coo._indices()
        # Swap row and column
        trans_idx = torch.stack([idx[1], idx[0]], dim=0)
        # Build new COO
        trans_shape = (
            self.sparse_csr_tensor.size(1),
            self.sparse_csr_tensor.size(0),
        )
        trans_coo = torch.sparse_coo_tensor(
            trans_idx, coo._values(), trans_shape
        )
        # Convert back to CSR
        trans_csr = trans_coo.to_sparse_csr()
        return SparseLinearOperator(trans_csr)

    

if __name__ == "__main__":
    print("Testing SparseLinearOperator...")
    
    # Create a simple sparse CSR tensor
    # Example: 3x3 matrix with some non-zero elements
    indices = torch.tensor([[0, 0, 1, 2], [0, 2, 1, 0]])  # row, col indices
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    size = (3, 3)
    sparse_coo = torch.sparse_coo_tensor(indices, values, size)
    sparse_csr = sparse_coo.to_sparse_csr()
    
    print(f"Original sparse CSR tensor:\n{sparse_csr}")
    print(f"Dense version:\n{sparse_csr.to_dense()}")
    
    # Create the SparseLinearOperator
    sparse_op = SparseLinearOperator(sparse_csr)
    
    # Test basic properties
    print(f"\nOperator size: {sparse_op.size()}")
    print(f"Operator shape: {sparse_op.shape}")
    
    # Test matrix multiplication with dense tensor
    rhs = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    print(f"\nRight-hand side tensor:\n{rhs}")
    
    result = sparse_op @ rhs
    print(f"\nSparse @ dense result:\n{result}")
    
    # Compare with dense matrix multiplication
    dense_result = sparse_csr.to_dense() @ rhs
    print(f"Dense @ dense result (for comparison):\n{dense_result}")
    print(f"Results match: {torch.allclose(result, dense_result)}")
    
    # Test transpose
    sparse_op_t = sparse_op.t()
    print(f"\nTransposed operator size: {sparse_op_t.size()}")
    
    # Test transpose multiplication
    lhs = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    result_t = lhs @ sparse_op
    print(f"\nDense @ sparse result:\n{result_t}")
    
    print("\nAll tests completed successfully!")

