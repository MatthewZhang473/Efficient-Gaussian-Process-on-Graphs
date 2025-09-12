# ablation_study.py
import sys, os, gc, warnings, subprocess
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
from efficient_graph_gp.graph_kernels import diffusion_kernel, generate_noisy_samples
from efficient_graph_gp.gpflow_kernels import GraphDiffusionKernel, GraphGeneralFastGRFKernel
import tensorflow as tf
import numpy as np
import gpflow, networkx as nx, pandas as pd
from tqdm import tqdm


warnings.filterwarnings('ignore')

# ----------------------------
# CONFIGURATION
# ----------------------------
CONFIG = {
    "mesh_size": 25,
    "beta_sample": 10,
    "noise_std": 0.1,
    "training_fraction": 0.1,
    "seeds": [100, 101, 102],              # BO seeds
    "wpns": [2**i for i in range(1, 8)],   # walks per node configs
    "ablation_flags": [False, True],       # ablation vs non-ablation
}

# ----------------------------
# GPU memory handling
# ----------------------------
def clear_gpu_memory():
    tf.keras.backend.clear_session()
    gc.collect()

# ----------------------------
# GP inference
# ----------------------------
def gp_inference(X, Y, X_new, graph_kernel):
    model = gpflow.models.GPR(data=(X, Y), kernel=graph_kernel, mean_function=None)
    gpflow.optimizers.Scipy().minimize(model.training_loss, model.trainable_variables)
    mean, _ = model.predict_f(X_new)
    lml = model.log_marginal_likelihood().numpy()
    return model, mean, lml

# ----------------------------
# Dataset generation
# ----------------------------
def generate_dataset(mesh_size, beta_sample, noise_std, seed):
    num_nodes = mesh_size**2
    adjacency_matrix = nx.adjacency_matrix(nx.grid_2d_graph(mesh_size, mesh_size)).todense()
    K_true = diffusion_kernel(adjacency_matrix, beta_sample)
    Y = generate_noisy_samples(K_true, noise_std=0, seed=seed)
    Y_noisy = generate_noisy_samples(K_true, noise_std=noise_std, seed=seed)
    X = np.arange(num_nodes, dtype=np.float64).reshape(-1, 1)
    return adjacency_matrix, X, Y, Y_noisy

# ----------------------------
# Experiment run
# ----------------------------
def run_experiment(adjacency_matrix, X, Y, Y_noisy, seed):
    np.random.seed(seed)
    num_nodes = len(Y)
    n_train = int(num_nodes * CONFIG["training_fraction"])
    train_idx = np.random.choice(num_nodes, n_train, replace=False)
    test_idx = np.setdiff1d(np.arange(num_nodes), train_idx)

    X_train = tf.convert_to_tensor(X[train_idx])
    Y_train = tf.convert_to_tensor(Y_noisy[train_idx].reshape(-1, 1))
    X_full = tf.convert_to_tensor(X)

    results = []

    # Diffusion kernel
    try:
        graph_kernel = GraphDiffusionKernel(adjacency_matrix)
        model, mean, lml = gp_inference(X_train, Y_train, X_full, graph_kernel)
        mse = np.mean((Y[test_idx] - mean.numpy()[test_idx].flatten())**2)
        results.append({'seed': seed, 'model': 'Diffusion', 'wpn': None, 'lml': lml, 'mse': mse})
    except Exception as e:
        results.append({'seed': seed, 'model': 'Diffusion', 'wpn': None, 'lml': np.nan, 'mse': np.nan, 'error': str(e)})
    finally:
        clear_gpu_memory()

    # GRF kernels
    for wpn in CONFIG["wpns"]:
        for ablation in CONFIG["ablation_flags"]:
            try:
                graph_kernel = GraphGeneralFastGRFKernel(
                    adjacency_matrix,
                    walks_per_node=wpn,
                    p_halt=0.01,
                    max_walk_length=10,
                    use_tqdm=False,
                    ablation=ablation
                )
                model, mean, lml = gp_inference(X_train, Y_train, X_full, graph_kernel)
                mse = np.mean((Y[test_idx] - mean.numpy()[test_idx].flatten())**2)
                results.append({
                    'seed': seed,
                    'model': 'GRF-ablation' if ablation else 'GRF',
                    'wpn': wpn,
                    'lml': lml,
                    'mse': mse
                })
            except Exception as e:
                results.append({
                    'seed': seed,
                    'model': 'GRF-ablation' if ablation else 'GRF',
                    'wpn': wpn,
                    'lml': np.nan,
                    'mse': np.nan,
                    'error': str(e)
                })
            finally:
                clear_gpu_memory()

    return results

# ----------------------------
# Parent orchestration
# ----------------------------
def parent_main():
    all_dfs = []
    for seed in tqdm(CONFIG["seeds"], desc="Seeds"):
        print(f"\nRunning seed {seed} in subprocess...")
        subprocess.run([sys.executable, __file__, str(seed)], check=True)
        df = pd.read_csv(f"results_seed_{seed}.csv")
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv("ablation_results.csv", index=False)
    print("All results saved to ablation_results.csv")

# ----------------------------
# Child execution
# ----------------------------
def child_main(seed: int):
    adjacency_matrix, X, Y, Y_noisy = generate_dataset(
        CONFIG["mesh_size"], CONFIG["beta_sample"], CONFIG["noise_std"], seed=42
    )
    results = run_experiment(adjacency_matrix, X, Y, Y_noisy, seed)
    df = pd.DataFrame(results)
    df.to_csv(f"results_seed_{seed}.csv", index=False)

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) == 1:
        parent_main()
    else:
        child_main(int(sys.argv[1]))
