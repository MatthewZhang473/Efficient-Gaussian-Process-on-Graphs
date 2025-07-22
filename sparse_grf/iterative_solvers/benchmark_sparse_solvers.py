"""
Benchmark sparse linear solvers for SPD systems.

Compares direct methods (dense, sparse) vs iterative methods (CG, PCG, AMG)
on randomly generated graph Laplacian-like matrices.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator, spsolve
import numpy.linalg as LA
import pandas as pd
import time
import os
from tqdm import tqdm
import warnings

# Optional dependencies
try:
    from sksparse.cholmod import cholesky
    HAS_CHOLMOD = True
except ImportError:
    HAS_CHOLMOD = False

try:
    import pyamg
    HAS_PYAMG = True
except ImportError:
    HAS_PYAMG = False


def generate_test_matrix(N, d_avg=10, sigma=0.1, seed=None):
    """Generate a sparse SPD test matrix M = K + sigma²I."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate sparse random matrix with average degree d_avg
    Phi = sp.random(N, N, d_avg/N, format='csr', random_state=seed)
    K = Phi @ Phi.T  # SPD kernel matrix
    M = K + sigma**2 * sp.eye(N, format='csr')
    return M


def solve_dense(M, b):
    """Dense direct solver using LU decomposition."""
    A = M.toarray() if sp.isspmatrix(M) else M
    return np.linalg.solve(A, b)


def solve_sparse_direct(M, b):
    """Sparse direct solver using SuperLU."""
    return spsolve(M, b)


def solve_cg(M, b, atol=1e-6, callback=None):
    """Unpreconditioned conjugate gradient."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='invalid value encountered')
        return cg(M, b, atol=atol, callback=callback)


def solve_jacobi_cg(M, b, atol=1e-6, callback=None):
    """Jacobi-preconditioned conjugate gradient."""
    D_inv = 1.0 / M.diagonal()
    P = LinearOperator(M.shape, matvec=lambda x: D_inv * x)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='invalid value encountered')
        return cg(M, b, M=P, atol=atol, callback=callback)


def solve_ic_cg(M, b, atol=1e-6, callback=None):
    """Incomplete Cholesky preconditioned CG."""
    if not HAS_CHOLMOD:
        raise ImportError("sksparse.cholmod not available")
    
    factor = cholesky(M)
    P = LinearOperator(M.shape, matvec=factor.solve_A)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='invalid value encountered')
        return cg(M, b, M=P, atol=atol, callback=callback)


def solve_amg(M, b, atol=1e-6):
    """Algebraic multigrid solver."""
    if not HAS_PYAMG:
        raise ImportError("pyamg not available")
    
    ml = pyamg.ruge_stuben_solver(M)
    residuals = []
    x = ml.solve(b, tol=atol, residuals=residuals)
    return x, len(residuals) - 1


def benchmark_solver(method, M, b, tol_frac=1e-6):
    """
    Benchmark a single solver method.
    
    Returns:
        tuple: (time, residual_norm, niter)
    """
    b_norm = LA.norm(b)
    atol = tol_frac * b_norm
    
    # Iteration counter for CG methods
    counter = {'niter': 0}
    def callback(xk):
        counter['niter'] += 1
    
    start_time = time.perf_counter()
    
    try:
        if method == 'dense':
            x = solve_dense(M, b)
            niter = None
        elif method == 'spsolve':
            x = solve_sparse_direct(M, b)
            niter = None
        elif method == 'cg':
            x, info = solve_cg(M, b, atol=atol, callback=callback)
            niter = counter['niter']
        elif method == 'jacobi':
            x, info = solve_jacobi_cg(M, b, atol=atol, callback=callback)
            niter = counter['niter']
        elif method == 'ic':
            x, info = solve_ic_cg(M, b, atol=atol, callback=callback)
            niter = counter['niter']
        elif method == 'amg':
            x, niter = solve_amg(M, b, atol=atol)
        else:
            return np.nan, np.nan, None
            
    except Exception as e:
        print(f"Error in {method}: {e}")
        return np.nan, np.nan, None
    
    elapsed_time = time.perf_counter() - start_time
    residual_norm = LA.norm(M.dot(x) - b) / b_norm
    
    return elapsed_time, residual_norm, niter


def run_benchmark(sizes, seeds, methods, d_avg=10, sigma=0.1):
    """Run the complete benchmark across all parameters."""
    records = []
    
    for N in tqdm(sizes, desc='Problem sizes'):
        for seed in tqdm(seeds, desc=f'Seeds (N={N})', leave=False):
            # Generate test problem
            M = generate_test_matrix(N, d_avg=d_avg, sigma=sigma, seed=seed)
            b = np.random.rand(N)
            
            # Benchmark each method
            for method in methods:
                time_taken, res_norm, niter = benchmark_solver(method, M, b)
                records.append({
                    'N': N,
                    'seed': seed,
                    'method': method,
                    'time': time_taken,
                    'residual_norm': res_norm,
                    'niter': niter
                })
    
    return pd.DataFrame(records)


def save_results(df, output_dir):
    """Save benchmark results with timestamp."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Save raw data
    raw_file = os.path.join(output_dir, f'benchmark_sparse_{timestamp}.csv')
    df.to_csv(raw_file, index=False)
    
    # Save summary statistics
    summary = df.groupby(['N', 'method']).agg({
        'time': 'mean',
        'residual_norm': 'mean',
        'niter': 'mean'
    }).reset_index()
    
    summary_file = os.path.join(output_dir, f'benchmark_sparse_summary_{timestamp}.csv')
    summary.to_csv(summary_file, index=False)
    
    print(f"Results saved to {raw_file}")
    print(f"Summary saved to {summary_file}")


def main():
    """Main benchmark execution."""
    # Configuration
    CONFIG = {
        'sizes': [2**i for i in range(4, 15)],
        'seeds': list(range(5)),
        'sigma': 0.1,
        'd_avg': 10,
        'methods': ['dense', 'spsolve', 'cg', 'jacobi']
    }
    
    # Add optional methods if available
    if HAS_CHOLMOD:
        CONFIG['methods'].append('ic')
    if HAS_PYAMG:
        CONFIG['methods'].append('amg')
    
    print(f"Running benchmark with methods: {CONFIG['methods']}")
    
    # Run benchmark
    df = run_benchmark(**CONFIG)
    
    # Save results
    output_dir = os.path.dirname(__file__)
    save_results(df, output_dir)


if __name__ == '__main__':
    main()
