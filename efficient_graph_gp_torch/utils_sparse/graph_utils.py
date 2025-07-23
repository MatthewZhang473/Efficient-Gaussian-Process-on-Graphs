import numpy as np
import scipy.sparse as sp


def get_normalized_laplacian(adj_matrix):
    """
    Compute normalized Laplacian matrix: L_norm = D^(-1/2) * L * D^(-1/2)
    where L = D - A is the graph Laplacian.
    
    Args:
        adj_matrix: Sparse adjacency matrix (scipy.sparse format)
    
    Returns:
        Normalized Laplacian as sparse CSR matrix
    """
    A = adj_matrix.tocsr()
    degrees = np.array(A.sum(axis=1)).flatten()
    
    # Compute D^(-1/2), handling zero degrees
    with np.errstate(divide='ignore'):
        d_inv_sqrt = 1.0 / np.sqrt(degrees)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    
    # Create diagonal matrices
    D = sp.diags(degrees, format='csr')
    D_inv_sqrt = sp.diags(d_inv_sqrt, format='csr')
    
    # Compute normalized Laplacian: D^(-1/2) * (D - A) * D^(-1/2)
    L = D - A
    return D_inv_sqrt @ L @ D_inv_sqrt

if __name__ == "__main__":
    print("=== Normalized Laplacian Example ===\n")
    
    # Create sparse adjacency matrix directly
    print("Step 1: Create sparse adjacency matrix")
    rows = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    cols = [1, 2, 0, 2, 3, 0, 1, 3, 1, 2]
    data = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    sparse_adj = sp.csr_matrix((data, (rows, cols)), shape=(4, 4))
    
    print(f"Sparse adjacency matrix (CSR):")
    print(f"Shape: {sparse_adj.shape}, Non-zeros: {sparse_adj.nnz}")
    print(f"Sparsity: {sparse_adj.nnz / sparse_adj.size * 100:.1f}% non-zero")
    print(f"Dense representation:\n{sparse_adj.toarray()}\n")
    
    # Compute normalized Laplacian (sparse processing)
    print("Step 2: Compute normalized Laplacian (sparse operations)")
    L_norm_sparse = get_normalized_laplacian(sparse_adj)
    print(f"Normalized Laplacian (sparse):")
    print(f"Shape: {L_norm_sparse.shape}, Non-zeros: {L_norm_sparse.nnz}")
    print(f"Type: {type(L_norm_sparse)}\n")
    
    # Convert to dense for verification
    print("Step 3: Dense output for verification")
    L_norm_dense = L_norm_sparse.toarray()
    print("Normalized Laplacian (dense output):")
    print(L_norm_dense)
    
    # Verify properties
    print("\nStep 4: Verify Laplacian properties")
    row_sums = L_norm_dense.sum(axis=1)
    print(f"Row sums (should be ~0): {row_sums}")
    print(f"Matrix is symmetric: {np.allclose(L_norm_dense, L_norm_dense.T)}")
    print(f"Positive semi-definite: {np.all(np.linalg.eigvals(L_norm_dense) >= -1e-10)}")
