import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator, spsolve
import time, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# try optional solvers
try:
    from sksparse.cholmod import cholesky
    has_ic = True
except ImportError:
    has_ic = False

try:
    import pyamg
    has_amg = True
except ImportError:
    has_amg = False

def time_solve(method, M, b, tol=1e-6):
    start = time.perf_counter()
    if method == 'dense':
        A = M.toarray() if sp.isspmatrix(M) else M
        _ = np.linalg.solve(A, b)
    elif method == 'spsolve':
        _ = spsolve(M, b)
    elif method == 'cg':
        _, _ = cg(M, b, atol=tol)
    elif method == 'jacobi':
        D_inv = 1.0 / M.diagonal()
        P = LinearOperator(M.shape, matvec=lambda x: D_inv * x)
        _, _ = cg(M, b, M=P, atol=tol)
    elif method == 'ic' and has_ic:
        factor = cholesky(M)
        P = LinearOperator(M.shape, matvec=factor.solve_A)
        _, _ = cg(M, b, M=P, atol=tol)
    elif method == 'amg' and has_amg:
        ml = pyamg.ruge_stuben_solver(M)
        _ = ml.solve(b, tol=tol)
    else:
        return np.nan
    return time.perf_counter() - start

if __name__ == '__main__':
    sizes  = [2**i for i in range(10, 14)]    # 1k to 16k
    seeds  = [0, 1, 2]
    sigma  = 0.1
    d_avg  = 3
    methods = ['dense', 'spsolve', 'cg', 'jacobi']
    if has_ic:  methods.append('ic')
    if has_amg: methods.append('amg')

    records = []
    for N in tqdm(sizes, desc='sizes'):
        for seed in tqdm(seeds, desc=f'seeds N={N}', leave=False):
            np.random.seed(seed)
            Phi = sp.random(N, N, d_avg/N, format='csr', random_state=seed)
            K   = Phi @ Phi.T
            M   = K + sigma**2 * sp.eye(N, format='csr')
            b   = np.random.rand(N)
            for m in methods:
                t = time_solve(m, M, b)
                records.append({'N': N, 'method': m, 'time': t})

    df     = pd.DataFrame(records)
    df_avg = df.groupby(['N','method'])['time'].mean().reset_index()
    out    = os.path.dirname(__file__)
    df_avg.to_csv(os.path.join(out, 'benchmark_sparse.csv'), index=False)

    sns.set(style='whitegrid')
    plt.figure()
    sns.lineplot(data=df_avg, x='N', y='time', hue='method', marker='o')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Problem size N'); plt.ylabel('Time (s)')
    plt.title('Sparse Solvers Benchmark')
    plt.savefig(os.path.join(out, 'benchmark_sparse.png'))
    plt.show()
