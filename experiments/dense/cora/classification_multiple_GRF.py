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
SEEDS = [0, 1, 2, 3, 4]
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

    # 3) SVGP model
    likelihood = gpflow.likelihoods.MultiClass(num_classes=cls_number)
    model = gpflow.models.SVGP(
        kernel=graph_kernel,
        likelihood=likelihood,
        inducing_variable=Z,
        num_latent_gps=cls_number,
        whiten=True,
    )

    # 4) Full-batch dataset (prefetch to overlap CPU/GPU)
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(train_num, seed=seed)
        .batch(train_num)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )
    train_iter = iter(train_ds)

    # 5) Optimizer
    optimizer = tf.optimizers.Adam(learning_rate=0.003)

    # 6) Compiled train step (runs on GPU where possible; math stays float64)
    @tf.function
    def train_step(Xb, Yb):
        with tf.GradientTape() as tape:
            loss = model.training_loss((Xb, Yb))  # negative ELBO
        grads = tape.gradient(loss, model.trainable_variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
        optimizer.apply_gradients(grads_and_vars)
        return loss

    # 7) Train
    print("Training SVGP (GRF)…")
    for step in tqdm(range(NUM_EPOCHES), desc=f"Seed {seed} Training", unit="step"):
        Xb, Yb = next(train_iter)
        _ = train_step(Xb, Yb)

    # 8) Hyperparameter summary
    print(f"\nModel hyperparameters (seed={seed}):")
    print_summary(model)

    # 9) Evaluate
    y_pred_mean, _ = model.predict_y(x_test)
    y_pred = np.argmax(y_pred_mean.numpy(), axis=1).ravel()
    acc = accuracy_score(y_test.ravel(), y_pred)
    print(f"Seed {seed} Test Accuracy: {acc*100:.2f}%")
    seed_accuracies.append((seed, acc))

# -------------------------------
# Summary over seeds
# -------------------------------
print("\n=== Summary over all seeds ===")
for seed, acc in seed_accuracies:
    print(f"Seed {seed}: Accuracy = {acc*100:.2f}%")
mean_acc = np.mean([acc for _, acc in seed_accuracies])
std_acc  = np.std([acc for _, acc in seed_accuracies])
print(f"\nMean accuracy over seeds: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
