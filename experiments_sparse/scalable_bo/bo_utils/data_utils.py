import os
import pickle
import numpy as np
import networkx as nx
from datetime import datetime
import sys
import scipy.sparse as sp

# Add the correct path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

from efficient_graph_gp_sparse.preprocessor import GraphPreprocessor
from efficient_graph_gp_sparse.utils_sparse import SparseLinearOperator

import numpy as np
import scipy.sparse as sp
import networkx as nx  # only used for small graphs

def generate_grid_data(n_nodes, beta_sample=1.0, noise_std=0.1, seed=42):
    """
    Central-maximum synthetic surface on a sqrt(n_nodes) x sqrt(n_nodes) grid.

    API compatibility with your previous function:
      returns dict with keys: 'A_sparse', 'G', 'y_true', 'X', 'Y'

    Notes:
      - For very large graphs, 'G' is set to None to avoid NetworkX overhead.
      - Requires n_nodes to be a perfect square.
    """
    # ---- grid shape ----
    s = int(np.sqrt(n_nodes))
    if s * s != n_nodes:
        raise ValueError("n_nodes must be a perfect square (got {}).".format(n_nodes))
    ny = nx_ = s  # (ny, nx)

    rng = np.random.default_rng(seed)

    # ---- coordinates in [0,1]^2 ----
    x = np.linspace(0, 1, nx_, dtype=np.float64)
    y = np.linspace(0, 1, ny,  dtype=np.float64)
    Xg, Yg = np.meshgrid(x, y)  # shape (ny, nx)

    # ---- smooth base + central bump (global maximum at center) ----
    Z_base = 1.2 * np.sin(2*np.pi*Xg) + 0.6 * np.cos(3*np.pi*Yg)

    cx, cy = 0.5, 0.5        # center of the peak
    lsx, lsy = 0.06, 0.06    # widths (symmetric)
    bump = 3 * np.exp(-0.5 * (((Xg - cx)/lsx)**2 + ((Yg - cy)/lsy)**2))

    Z = beta_sample * (Z_base + bump)         # true field
    y_true = Z.reshape(-1)
    y_observed = y_true + rng.normal(0.0, noise_std, size=y_true.shape)

    # ---- sparse 4-neighbour adjacency via Kronecker products (CSR) ----
    ex = np.ones(nx_)
    ey = np.ones(ny)
    Tx = sp.diags([ex[:-1], ex[:-1]], offsets=[-1, 1], shape=(nx_, nx_), format="csr")
    Ty = sp.diags([ey[:-1], ey[:-1]], offsets=[-1, 1], shape=(ny, ny),  format="csr")
    A_sparse = sp.kron(sp.eye(ny, format="csr"), Tx, format="csr") + sp.kron(Ty, sp.eye(nx_, format="csr"), format="csr")

    # ---- optional NetworkX graph for small cases; None for large ----
    # (keeps the 'G' key for API compatibility)
    if n_nodes <= 40000:  # ~200x200; adjust if you like
        G = nx.grid_2d_graph(ny, nx_)
    else:
        G = None

    return {
        'A_sparse': A_sparse,                         # scipy.sparse CSR
        'G': G,                                       # networkx.Graph or None
        'y_true': y_true,                             # shape (N,)
        'X': np.arange(n_nodes).reshape(-1, 1).astype(np.float64),
        'Y': y_observed.reshape(-1, 1),               # shape (N,1)
    }

def get_cached_data(config):
    filename = f"grid_n{config.N_NODES}_beta{config.DATA_PARAMS['beta_sample']}_noise{config.DATA_PARAMS['noise_std']}_seed{config.DATA_SEED}.pkl"
    filepath = os.path.join(config.DATA_DIR, filename)
    
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    # Remove kernel_std from params since generate_grid_data doesn't use it
    data_params = {k: v for k, v in config.DATA_PARAMS.items() if k != 'kernel_std'}
    data = generate_grid_data(config.N_NODES, seed=config.DATA_SEED, **data_params)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    return data

def get_step_matrices(data, config):
    filename = f"step_matrices_sparse_n{config.N_NODES}_seed{config.DATA_SEED}_w{config.WALKS_PER_NODE}_p{config.P_HALT}_l{config.MAX_WALK_LENGTH}.pkl"
    filepath = os.path.join(config.STEP_MATRICES_DIR, filename)
    
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)['step_matrices_torch']
    
    pp = GraphPreprocessor(
        adjacency_matrix=data['A_sparse'],
        walks_per_node=config.WALKS_PER_NODE,
        p_halt=config.P_HALT,
        max_walk_length=config.MAX_WALK_LENGTH,
        random_walk_seed=config.DATA_SEED,
        load_from_disk=False,
        use_tqdm=True,
        n_processes=16
    )
    
    pp.preprocess_graph(save_to_disk=False)
    step_matrices = pp.step_matrices_scipy
    
    save_data = {'step_matrices_torch': step_matrices, 'metadata': {'n_nodes': config.N_NODES, 'timestamp': datetime.now().isoformat()}}
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)
    
    return step_matrices

def convert_to_device(step_matrices, device):
    result = []
    for mat in step_matrices:
        tensor = GraphPreprocessor.from_scipy_csr(mat).to(device)
        result.append(SparseLinearOperator(tensor))
    return result
