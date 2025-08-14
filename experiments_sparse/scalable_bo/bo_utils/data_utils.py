import os
import pickle
import numpy as np
import networkx as nx
from datetime import datetime
import sys

# Add the correct path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

from efficient_graph_gp_sparse.preprocessor import GraphPreprocessor
from efficient_graph_gp_sparse.utils_sparse import SparseLinearOperator

def generate_grid_data(n_nodes, beta_sample=1.0, noise_std=0.1, seed=42):
    np.random.seed(seed)
    G = nx.grid_2d_graph(int(np.sqrt(n_nodes)), int(np.sqrt(n_nodes)))
    A = nx.adjacency_matrix(G).tocsr()
    
    x = np.linspace(0, 1, int(np.sqrt(n_nodes)))
    y = np.linspace(0, 1, int(np.sqrt(n_nodes)))
    X, Y = np.meshgrid(x, y)
    Z = beta_sample * (2*np.sin(2*np.pi*X) + 0.5*np.cos(4*np.pi*Y) + 0.3*np.sin(2*np.pi*X))
    y_true = Z.flatten()
    y_observed = y_true + np.random.normal(0, noise_std, n_nodes)
    
    return {
        'A_sparse': A,
        'G': G,
        'y_true': y_true,
        'X': np.arange(n_nodes).reshape(-1, 1).astype(np.float64),
        'Y': y_observed.reshape(-1, 1)
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
        n_processes=4
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
