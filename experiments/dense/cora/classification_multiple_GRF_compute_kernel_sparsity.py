import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # use cuda:0

import tensorflow as tf
import numpy as np
import scipy.special
from sklearn.metrics import accuracy_score
import gpflow
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import networkx as nx
from gpflow.utilities import print_summary
import tensorflow_probability as tfp
import seaborn as sns
from tqdm import tqdm

import sys
# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Add the cora utilities path
cora_utils_path = os.path.join(project_root, 'data', 'cora')
sys.path.insert(0, cora_utils_path)

# Also add the parent directory of cora_utils if it's structured differently
parent_cora_path = os.path.dirname(cora_utils_path)
sys.path.insert(0, parent_cora_path)

import pandas as pd
import pickle

from efficient_graph_gp.graph_kernels import get_normalized_laplacian
from efficient_graph_gp.gpflow_kernels import (
    GraphDiffusionFastGRFKernel, GraphDiffusionPoFMKernel, GraphDiffusionKernel,
    GraphGeneralPoFMKernel, GraphGeneralFastGRFKernel
)
from utils import compute_fro
from cora_utils.preprocessing import load_PEMS, load_cora

# -------------------------------
# GPU + dtype config (float64 everywhere)
# -------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

gpflow.config.set_default_float(np.float64)   # GPflow default; avoid dtype mismatches
dtype = gpflow.config.default_float()         # == np.float64

# -------------------------------
# Load & Preprocess Dataset
# -------------------------------
np.random.seed(1)
num_eigenpairs = 500
dataset = 'cora'
cls_number = 7
train_num = int(2485 * 0.8)
test_num  = int(2485 * 0.2)
M = train_num // 2

G, data_train, data_test = load_cora(num_train=train_num, num_test=test_num)
adjacency_matrix = nx.to_numpy_array(G).astype(dtype)
x_train, y_train = data_train
x_test,  y_test  = data_test

# Cast features to float64 (labels remain int)
x_train = x_train.astype(dtype)
x_test  = x_test.astype(dtype)

# -------------------------------
# Build GP Model & Train
# -------------------------------
SEEDS = [0,1,2,3,4]
MAX_WALK_LENGTH = 4
seed_accuracies = []
NUM_EPOCHES = 1024
WALKS_PER_NODE = 100

for seed in SEEDS:
    print(f"\n=== Running seed {seed} ===")
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # 1) Sample M inducing points from training nodes
    indices = np.random.choice(train_num, M, replace=False)
    Z = x_train[indices].copy().astype(dtype)

    # 2) Graph Diffusion kernel
    graph_kernel = GraphGeneralFastGRFKernel(
        adjacency_matrix=adjacency_matrix,
        walks_per_node=WALKS_PER_NODE,
        p_halt=0.1,
        max_walk_length=MAX_WALK_LENGTH,
        use_tqdm=True
    )
    
    # 3) Compute the kernel sparsity
    modulation_function = graph_kernel.modulator_vector
    kernel = graph_kernel.grf_kernel(modulation_function).cpu().numpy()
    nnz = np.count_nonzero(kernel)
    total_elements = kernel.shape[0] * kernel.shape[1]
    density = nnz / total_elements if total_elements > 0 else 0
    print(f"Kernel density (sparsity): {density:.4f}")
