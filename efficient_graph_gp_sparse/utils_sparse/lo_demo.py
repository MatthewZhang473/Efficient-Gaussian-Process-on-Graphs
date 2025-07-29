import torch
import time
from linear_operator.operators import (
    DiagLinearOperator, 
    DenseLinearOperator, 
    KroneckerProductLinearOperator,
    AddedDiagLinearOperator
)

def simple_linear_operator_example():
    """
    Simple example demonstrating LinearOperator usage
    """
    print("=== LinearOperator Example ===\n")
    
    # Example 1: Diagonal Linear Operators
    print("1. Diagonal Linear Operators")
    print("-" * 30)
    
    # Create diagonal values
    diag_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Dense representation (inefficient)
    dense_matrix = torch.diag(diag_values)
    print(f"Dense matrix shape: {dense_matrix.shape}")
    print(f"Dense matrix storage: {dense_matrix.numel()} elements")
    
    # LinearOperator representation (efficient)
    diag_lo = DiagLinearOperator(diag_values)
    print(f"DiagLinearOperator shape: {diag_lo.shape}")
    print(f"DiagLinearOperator storage: {diag_values.numel()} elements")
    print(f"Memory reduction: {dense_matrix.numel() / diag_values.numel():.1f}x\n")
    
    # Example 2: Matrix Operations
    print("2. Matrix Operations")
    print("-" * 20)
    
    # Create another diagonal operator
    diag2_values = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5])
    diag2_lo = DiagLinearOperator(diag2_values)
    
    # Addition (structure is preserved)
    sum_lo = diag_lo + diag2_lo
    print(f"Addition result type: {type(sum_lo).__name__}")
    print(f"Sum diagonal: {sum_lo.diagonal()}")
    
    # Matrix multiplication
    rhs = torch.randn(5, 3)
    result = diag_lo @ rhs
    print(f"Matrix multiplication result shape: {result.shape}")
    
    # Verify against dense computation
    dense_result = dense_matrix @ rhs
    print(f"Results match: {torch.allclose(result, dense_result)}\n")
    
    # Example 3: Linear System Solving
    print("3. Linear System Solving")
    print("-" * 25)
    
    # Create a positive definite diagonal system
    diag_pd = DiagLinearOperator(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    b = torch.randn(5)
    
    # Solve using LinearOperator (efficient O(n) for diagonal)
    start_time = time.time()
    x_lo = torch.linalg.solve(diag_pd, b)
    lo_time = time.time() - start_time
    
    # Solve using dense matrix (inefficient O(n^3))
    dense_pd = diag_pd.to_dense()
    start_time = time.time()
    x_dense = torch.linalg.solve(dense_pd, b)
    dense_time = time.time() - start_time
    
    print(f"LinearOperator solve time: {lo_time:.6f}s")
    print(f"Dense solve time: {dense_time:.6f}s")
    print(f"Results match: {torch.allclose(x_lo, x_dense)}")
    if dense_time > 0:
        print(f"Speedup: {dense_time / lo_time:.1f}x\n")
    
    # Example 4: Kronecker Product (Advanced)
    print("4. Kronecker Product Example")
    print("-" * 30)
    
    # Create smaller matrices for Kronecker product
    A = torch.randn(3, 3)
    B = torch.randn(2, 2)
    
    # Dense Kronecker product
    dense_kron = torch.kron(A, B)
    print(f"Dense Kronecker shape: {dense_kron.shape}")
    print(f"Dense storage: {dense_kron.numel()} elements")
    
    # LinearOperator Kronecker product
    A_lo = DenseLinearOperator(A)
    B_lo = DenseLinearOperator(B)
    kron_lo = KroneckerProductLinearOperator(A_lo, B_lo)
    print(f"Kronecker LinearOperator shape: {kron_lo.shape}")
    print(f"Actual storage: {A.numel() + B.numel()} elements")
    print(f"Memory reduction: {dense_kron.numel() / (A.numel() + B.numel()):.1f}x")
    
    # Verify operations work
    test_vec = torch.randn(6)  # 3*2 = 6
    result_dense = dense_kron @ test_vec
    result_lo = kron_lo @ test_vec
    print(f"Kronecker operations match: {torch.allclose(result_dense, result_lo)}\n")
    
    # Example 5: Added Diagonal
    print("5. Low-rank + Diagonal Structure")
    print("-" * 35)
    
    # Create a low-rank matrix + diagonal
    low_rank = torch.randn(5, 2)  # rank-2 matrix
    diag_component = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    
    # Dense representation
    dense_lr_diag = low_rank @ low_rank.T + torch.diag(diag_component)
    
    # LinearOperator representation
    lr_lo = DenseLinearOperator(low_rank @ low_rank.T)
    lr_diag_lo = lr_lo.add_diagonal(diag_component)
    
    print(f"Structure type: {type(lr_diag_lo).__name__}")
    print(f"Dense storage: {dense_lr_diag.numel()} elements")
    print(f"Structured storage: ~{low_rank.numel() + diag_component.numel()} elements")
    
    # Test matrix-vector multiplication
    test_vec = torch.randn(5)
    dense_result = dense_lr_diag @ test_vec
    lo_result = lr_diag_lo @ test_vec
    print(f"Results match: {torch.allclose(dense_result, lo_result, atol=1e-6)}")

def performance_comparison_example():
    """
    Demonstrate performance benefits with larger matrices
    """
    print("\n=== Performance Comparison ===\n")
    
    n = 1000
    print(f"Comparing performance with n={n}")
    
    # Create large diagonal matrix
    diag_vals = torch.randn(n).abs() + 0.1  # ensure positive definite
    
    # Dense version
    dense_matrix = torch.diag(diag_vals)
    
    # LinearOperator version  
    diag_lo = DiagLinearOperator(diag_vals)
    
    # Test vector
    b = torch.randn(n)
    
    print(f"Memory usage:")
    print(f"  Dense: {dense_matrix.numel() * 4 / 1024**2:.1f} MB")
    print(f"  LinearOperator: {diag_vals.numel() * 4 / 1024**2:.1f} MB")
    print(f"  Memory reduction: {dense_matrix.numel() / diag_vals.numel():.0f}x")
    
    # Time matrix-vector multiplication
    print(f"\nMatrix-vector multiplication:")
    
    # Dense timing
    start = time.time()
    for _ in range(100):
        _ = dense_matrix @ b
    dense_time = time.time() - start
    
    # LinearOperator timing
    start = time.time()
    for _ in range(100):
        _ = diag_lo @ b
    lo_time = time.time() - start
    
    print(f"  Dense: {dense_time:.4f}s")
    print(f"  LinearOperator: {lo_time:.4f}s")
    print(f"  Speedup: {dense_time / lo_time:.1f}x")

if __name__ == "__main__":
    simple_linear_operator_example()
    performance_comparison_example()
