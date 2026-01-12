# traffic_experiment.py
import sys, os, warnings, subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ----------------------------
# CONFIGURATION
# ----------------------------
CONFIG = {
    "dataset": "PeMS-Bay-new",
    "num_train": 250,
    "num_eigenpairs": 500,
    "max_walk_length": 4,
    "seeds": [1111 + i for i in range(5)],
    "walks_per_nodes": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    "nlpd_mode": "predict_y",
    "run_svgp": True,
    "run_pofm": False,
    "run_diffusion": False,
    "run_grf": False,
    "run_agrf": False,
    "run_gnn": False,
}

RESULTS_DIR = "experiments_dense/traffic_dataset/results"

# Configuration for SVGP
SVGP_CONFIG = {
    "num_inducing": 150,
    "num_iterations": 1000,
    "learning_rate": 0.01,
}

GNN_CONFIG = {
    "hidden_dims": [64, 32, 16],
    "dropout_rate": 0.1,
    "num_epochs": 500,
    "learning_rate": 1e-3,
}


# ============================================================
# ==============  PARENT ORCHESTRATION =======================
# ============================================================
def parent_main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_dfs = []

    for seed in tqdm(CONFIG["seeds"], desc="Seeds"):
        if CONFIG["run_svgp"]:
            print(f"\n[Parent] Running SVGP for seed={seed}")
            subprocess.run([sys.executable, "-u", __file__, str(seed), "svgp"], check=True)

        if CONFIG["run_pofm"]:
            print(f"\n[Parent] Running PoFM for seed={seed}")
            subprocess.run([sys.executable, "-u", __file__, str(seed), "pofm"], check=True)

        if CONFIG["run_diffusion"]:
            print(f"[Parent] Running Diffusion for seed={seed}")
            subprocess.run([sys.executable, "-u", __file__, str(seed), "diffusion"], check=True)

        if CONFIG["run_grf"]:
            print(f"[Parent] Running GRF for seed={seed}")
            subprocess.run([sys.executable, "-u", __file__, str(seed), "grf"], check=True)

        if CONFIG["run_agrf"]:
            print(f"[Parent] Running A-GRF for seed={seed}")
            subprocess.run([sys.executable, "-u", __file__, str(seed), "agrf"], check=True)

        if CONFIG["run_gnn"]:
            print(f"[Parent] Running GCN for seed={seed}")
            subprocess.run([sys.executable, "-u", __file__, str(seed), "gnn"], check=True)

        # Combine per-seed results (each model saved separately)
        dfs = []
        tags = []
        if CONFIG["run_svgp"]:
            tags.append("svgp")
        if CONFIG["run_pofm"]:
            tags.append("pofm")
        if CONFIG["run_diffusion"]:
            tags.append("diffusion")
        if CONFIG["run_grf"]:
            tags.append("grf")
        if CONFIG["run_agrf"]:
            tags.append("agrf")
        if CONFIG["run_gnn"]:
            tags.append("gnn")
        
        for tag in tags:
            path = f"{RESULTS_DIR}/results_seed_{seed}_{tag}.csv"
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))
        if dfs:
            df_seed = pd.concat(dfs, ignore_index=True)
            df_seed.to_csv(f"{RESULTS_DIR}/results_seed_{seed}.csv", index=False)
            all_dfs.append(df_seed)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv(f"{RESULTS_DIR}/traffic_experiment_results.csv", index=False)
        print(f"\n[Parent] All results saved to {RESULTS_DIR}/traffic_experiment_results.csv")
    else:
        print("\n[Parent] No result files found.")


# ============================================================
# ==============  CHILD EXECUTION ============================
# ============================================================
def child_main(seed: int, experiment_type: str):
    # Delay heavy imports to child to avoid early TF init in parent
    import gc
    import tensorflow as tf
    import tensorflow_probability as tfp
    import gpflow
    import networkx as nx

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.append(project_root)

    from efficient_graph_gp.gpflow_kernels import (
        GraphDiffusionFastGRFKernel,
        GraphDiffusionPoFMKernel,
        GraphDiffusionKernel,
        GraphGeneralFastGRFKernel,
    )
    from traffic_utils.preprocessing import load_PEMS

    def clear_gpu():
        tf.keras.backend.clear_session()
        gc.collect()

    # ---------- Helpers ----------
    def train_gpr(X, Y, kernel, noise_var_center=None):
        """Fit a gpflow GPR with optional LogNormal prior/initialization on noise."""
        model = gpflow.models.GPR(
            data=(X, Y),
            kernel=kernel,
            mean_function=None,
            noise_variance=0.01,
        )
        if noise_var_center is not None:
            model.likelihood.variance.prior = tfp.distributions.LogNormal(
                loc=np.log(noise_var_center), scale=0.5
            )
            model.likelihood.variance.assign(noise_var_center)
        gpflow.optimizers.Scipy().minimize(model.training_loss, model.trainable_variables)
        return model

    def train_svgp(X, Y, kernel, num_inducing, num_iterations, learning_rate):
        """Train SVGP model using Adam optimizer."""
        N = X.shape[0]
        Z = X[:num_inducing, :].numpy().copy()  # Initialize inducing locations

        model = gpflow.models.SVGP(
            kernel=kernel,
            likelihood=gpflow.likelihoods.Gaussian(),
            inducing_variable=Z,
            num_data=N
        )

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def optimization_step():
            with tf.GradientTape() as tape:
                loss = model.training_loss((X, Y))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        for i in range(num_iterations):
            loss = optimization_step()
            if i % 100 == 0:
                print(f"    Iteration {i}: Loss = {loss.numpy():.4f}")

        return model

    def eval_fullcov(model, x_test, y_test, orig_std, jitter=1e-6):
        lml = float(model.log_marginal_likelihood().numpy())
        mean_f, cov_f = model.predict_f(x_test, full_cov=True)  # mean_f [N,1], cov_f [N,N]
        mean_vec = tf.reshape(mean_f, [-1])
        N = tf.shape(mean_vec)[0]
        cov_y = cov_f + tf.linalg.diag(tf.fill([N], model.likelihood.variance))
        cov_y = cov_y + jitter * tf.eye(N, dtype=cov_y.dtype)
        rmse = float(orig_std * tf.sqrt(tf.reduce_mean((y_test[:, 0] - mean_vec) ** 2)))
        dist = tfp.distributions.MultivariateNormalFullCovariance(loc=mean_vec, covariance_matrix=cov_y)
        nlpd = float(-dist.log_prob(tf.reshape(y_test, [-1])))
        return lml, rmse, nlpd

    def eval_predict_y(model, x_test, y_test, orig_std):
        lml = float(model.log_marginal_likelihood().numpy())
        mean_y, var_y = model.predict_y(x_test)  # includes obs noise
        mean_y = tf.reshape(mean_y, [-1])
        var_y = tf.reshape(var_y, [-1])
        rmse = float(orig_std * tf.sqrt(tf.reduce_mean((y_test[:, 0] - mean_y) ** 2)))
        nlpd = -tf.reduce_sum(
            tfp.distributions.Normal(loc=mean_y, scale=tf.sqrt(var_y)).log_prob(y_test[:, 0])
        ).numpy()
        return lml, rmse, nlpd

    def evaluate(model, x_test, y_test, orig_std):
        return eval_fullcov(model, x_test, y_test, orig_std) if CONFIG["nlpd_mode"] == "fullcov" \
            else eval_predict_y(model, x_test, y_test, orig_std)

    def load_pofm_hypers(seed_int):
        """Read PoFM-learned hypers from CSV. Returns dict or None."""
        path = f"{RESULTS_DIR}/results_seed_{seed_int}_pofm.csv"
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        row = df[df["model"] == "PoFM"]
        if row.empty:
            if len(df) == 0:
                return None
            row = df.iloc[[0]]
        return {
            "beta": float(row["beta"].values[0]) if "beta" in row else None,
            "sigma_f": float(row["sigma_f"].values[0]) if "sigma_f" in row else None,
            "variance": float(row["variance"].values[0]) if "variance" in row else None,
        }

    def get_noise_center_from_pofm(seed_int, default=None):
        hypers = load_pofm_hypers(seed_int)
        return hypers["variance"] if (hypers and hypers["variance"] is not None) else default

    # ---------- Data ----------
    np.random.seed(seed)
    tf.random.set_seed(seed)

    G, data_train, data_test, data = load_PEMS(num_train=CONFIG["num_train"])
    x_train, y_train = data_train
    x_test, y_test = data_test
    x, y = data

    # Standardize targets
    orig_mean, orig_std = np.mean(y_train), np.std(y_train)
    y_train = (y_train - orig_mean) / orig_std
    y_test = (y_test - orig_mean) / orig_std

    X_train = tf.convert_to_tensor(x_train)
    Y_train = tf.convert_to_tensor(y_train)
    adjacency = nx.to_numpy_array(G)

    results = []

    # ---------- SVGP ----------
    if experiment_type == "svgp":
        print(f"[Child] Seed {seed}: SVGP")
        try:
            kern = GraphDiffusionKernel(adjacency, normalize_laplacian=True)
            model = train_svgp(
                X_train, Y_train, kern,
                num_inducing=SVGP_CONFIG["num_inducing"],
                num_iterations=SVGP_CONFIG["num_iterations"],
                learning_rate=SVGP_CONFIG["learning_rate"]
            )
            # Use predict_y for consistent NLPD calculation
            mean_y, var_y = model.predict_y(x_test)
            mean_y = tf.reshape(mean_y, [-1])
            var_y = tf.reshape(var_y, [-1])
            rmse = float(orig_std * tf.sqrt(tf.reduce_mean((y_test[:, 0] - mean_y) ** 2)))
            nlpd = -tf.reduce_sum(
                tfp.distributions.Normal(loc=mean_y, scale=tf.sqrt(var_y)).log_prob(y_test[:, 0])
            ).numpy()
            
            results.append({
                "seed": seed, "model": "SVGP", "walks_per_node": None,
                "lml": None,  # SVGP doesn't have true LML
                "rmse": rmse, "nlpd": nlpd,
                "beta": float(model.kernel.beta.numpy()),
                "sigma_f": float(model.kernel.sigma_f.numpy()),
                "variance": float(model.likelihood.variance.numpy()),
            })
        except Exception as e:
            print(f"    SVGP failed: {e}")
            results.append({
                "seed": seed, "model": "SVGP", "walks_per_node": None,
                "lml": np.nan, "rmse": np.nan, "nlpd": np.nan, "error": str(e),
            })
        finally:
            clear_gpu()
        pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/results_seed_{seed}_svgp.csv", index=False)
        return

    # ---------- PoFM ----------
    if experiment_type == "pofm":
        print(f"[Child] Seed {seed}: PoFM")
        try:
            kern = GraphDiffusionPoFMKernel(
                adjacency,
                max_expansion=CONFIG["max_walk_length"],
                normalize_laplacian=True,
            )
            model = train_gpr(X_train, Y_train, kern)
            lml, rmse, nlpd = evaluate(model, x_test, y_test, orig_std)
            results.append({
                "seed": seed, "model": "PoFM", "walks_per_node": None,
                "lml": lml, "rmse": rmse, "nlpd": nlpd,
                "beta": float(model.kernel.beta.numpy()),
                "sigma_f": float(model.kernel.sigma_f.numpy()),
                "variance": float(model.likelihood.variance.numpy()),
            })
        except Exception as e:
            print(f"    PoFM failed: {e}")
            results.append({
                "seed": seed, "model": "PoFM", "walks_per_node": None,
                "lml": np.nan, "rmse": np.nan, "nlpd": np.nan, "error": str(e),
            })
        finally:
            clear_gpu()
        pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/results_seed_{seed}_pofm.csv", index=False)
        return

    # ---------- Diffusion (NO re-optimization) ----------
    if experiment_type == "diffusion":
        print(f"[Child] Seed {seed}: Diffusion (no training; use PoFM hypers)")
        try:
            hypers = load_pofm_hypers(seed)
            if not hypers or any(v is None for v in [hypers.get("beta"), hypers.get("sigma_f"), hypers.get("variance")]):
                raise RuntimeError("PoFM hypers not found; run pofm first.")

            kern = GraphDiffusionKernel(adjacency, normalize_laplacian=True)
            model = gpflow.models.GPR(
                data=(X_train, Y_train),
                kernel=kern,
                mean_function=None,
                noise_variance=hypers["variance"],
            )
            # Warm-set exact values and DO NOT optimize
            model.kernel.beta.assign(hypers["beta"])
            model.kernel.sigma_f.assign(hypers["sigma_f"])
            model.likelihood.variance.assign(hypers["variance"])

            # Direct evaluation with the fixed hypers
            lml, rmse, nlpd = evaluate(model, x_test, y_test, orig_std)
            results.append({
                "seed": seed, "model": "Diffusion", "walks_per_node": None,
                "lml": lml, "rmse": rmse, "nlpd": nlpd,
                "beta": float(model.kernel.beta.numpy()),
                "sigma_f": float(model.kernel.sigma_f.numpy()),
                "variance": float(model.likelihood.variance.numpy()),
            })
        except Exception as e:
            print(f"    Diffusion failed: {e}")
            results.append({
                "seed": seed, "model": "Diffusion", "walks_per_node": None,
                "lml": np.nan, "rmse": np.nan, "nlpd": np.nan, "error": str(e),
            })
        finally:
            clear_gpu()
        pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/results_seed_{seed}_diffusion.csv", index=False)
        return

    # ---------- GRF ----------
    if experiment_type == "grf":
        print(f"[Child] Seed {seed}: GRF loop")
        noise_center = get_noise_center_from_pofm(seed, default=float(np.var(y_train)))
        for wpn in tqdm(CONFIG["walks_per_nodes"], desc=f"Seed {seed} - GRF"):
            try:
                kern = GraphDiffusionFastGRFKernel(
                    adjacency,
                    walks_per_node=wpn,
                    p_halt=0.1,
                    max_walk_length=CONFIG["max_walk_length"] + 1,
                    normalize_laplacian=True,
                )
                model = train_gpr(X_train, Y_train, kern, noise_var_center=noise_center)
                lml, rmse, nlpd = evaluate(model, x_test, y_test, orig_std)
                results.append({
                    "seed": seed, "model": "GRF", "walks_per_node": wpn,
                    "lml": lml, "rmse": rmse, "nlpd": nlpd,
                    "beta": float(model.kernel.beta.numpy()),
                    "sigma_f": float(model.kernel.sigma_f.numpy()),
                    "variance": float(model.likelihood.variance.numpy()),
                })
            except Exception as e:
                results.append({
                    "seed": seed, "model": "GRF", "walks_per_node": wpn,
                    "lml": np.nan, "rmse": np.nan, "nlpd": np.nan, "error": str(e),
                })
            finally:
                clear_gpu()
        pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/results_seed_{seed}_grf.csv", index=False)
        return

    # ---------- A-GRF ----------
    if experiment_type == "agrf":
        print(f"[Child] Seed {seed}: A-GRF loop")
        noise_center = get_noise_center_from_pofm(seed, default=float(np.var(y_train)))
        for wpn in tqdm(CONFIG["walks_per_nodes"], desc=f"Seed {seed} - A-GRF"):
            try:
                kern = GraphGeneralFastGRFKernel(
                    adjacency,
                    walks_per_node=wpn,
                    p_halt=0.1,
                    max_walk_length=CONFIG["max_walk_length"],
                )
                model = train_gpr(X_train, Y_train, kern, noise_var_center=noise_center)
                lml, rmse, nlpd = evaluate(model, x_test, y_test, orig_std)
                results.append({
                    "seed": seed, "model": "A-GRF", "walks_per_node": wpn,
                    "lml": lml, "rmse": rmse, "nlpd": nlpd,
                    "variance": float(model.likelihood.variance.numpy()),
                })
            except Exception as e:
                results.append({
                    "seed": seed, "model": "A-GRF", "walks_per_node": wpn,
                    "lml": np.nan, "rmse": np.nan, "nlpd": np.nan, "error": str(e),
                })
            finally:
                clear_gpu()
        pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/results_seed_{seed}_agrf.csv", index=False)
        return

    # ---------- GNN ----------
    if experiment_type == "gnn":
        print(f"[Child] Seed {seed}: Deterministic GCN baseline")
        try:
            adjacency = nx.to_numpy_array(G, dtype=np.float32)
            num_nodes = adjacency.shape[0]

            def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
                adj_with_self = adj + np.eye(adj.shape[0], dtype=adj.dtype)
                degree = np.sum(adj_with_self, axis=1)
                degree[degree == 0] = 1.0
                inv_sqrt = degree ** -0.5
                adj_norm = adj_with_self * inv_sqrt[:, None]
                return adj_norm * inv_sqrt[None, :]

            def build_full_tensor(template: np.ndarray, values: np.ndarray) -> np.ndarray:
                full = np.zeros((num_nodes, template.shape[1]), dtype=np.float32)
                full[template[:, 0].astype(int)] = values.astype(np.float32)
                return full

            adjacency_norm = tf.constant(normalize_adjacency(adjacency), dtype=tf.float32)
            X_full = build_full_tensor(x, x)
            Y_full_train = np.zeros((num_nodes, 1), dtype=np.float32)
            Y_full_test = np.zeros((num_nodes, 1), dtype=np.float32)
            train_nodes = x_train[:, 0].astype(int)
            test_nodes = x_test[:, 0].astype(int)
            Y_full_train[train_nodes] = y_train
            Y_full_test[test_nodes] = y_test

            X_full_tensor = tf.constant(X_full, dtype=tf.float32)
            Y_train_tensor = tf.constant(Y_full_train, dtype=tf.float32)
            Y_test_tensor = tf.constant(Y_full_test, dtype=tf.float32)
            train_mask = np.zeros(num_nodes, dtype=np.float32)
            train_mask[train_nodes] = 1.0
            train_mask_tensor = tf.constant(train_mask[:, None], dtype=tf.float32)
            test_nodes_tensor = tf.constant(test_nodes, dtype=tf.int32)

            class GraphConvLayer(tf.keras.layers.Layer):
                def __init__(self, units, activation="relu", use_bias=True):
                    super().__init__()
                    self.units = units
                    self.activation = tf.keras.activations.get(activation)
                    self.use_bias = use_bias

                def build(self, input_shape):
                    self.kernel = self.add_weight(
                        name="kernel",
                        shape=(input_shape[-1], self.units),
                        initializer="glorot_uniform",
                    )
                    if self.use_bias:
                        self.bias = self.add_weight(
                            name="bias", shape=(self.units,), initializer="zeros"
                        )

                def call(self, inputs, adjacency_matrix):
                    support = tf.matmul(inputs, self.kernel)
                    output = tf.matmul(adjacency_matrix, support)
                    if self.use_bias:
                        output = output + self.bias
                    return self.activation(output) if self.activation else output

            class GNNRegressor(tf.keras.Model):
                def __init__(self, hidden_dims, dropout_rate):
                    super().__init__()
                    self.conv_layers = [GraphConvLayer(dim) for dim in hidden_dims]
                    self.dropouts = [tf.keras.layers.Dropout(dropout_rate) for _ in hidden_dims]
                    self.output_layer = tf.keras.layers.Dense(1)

                def call(self, inputs, adjacency_matrix, training=False):
                    x_local = inputs
                    for conv, drop in zip(self.conv_layers, self.dropouts):
                        x_local = conv(x_local, adjacency_matrix)
                        if training:
                            x_local = drop(x_local, training=training)
                    return self.output_layer(x_local)

            def masked_mse(y_true, y_pred, mask):
                squared_error = tf.square(y_true - y_pred)
                masked_error = tf.reduce_sum(mask * squared_error)
                denom = tf.reduce_sum(mask) + 1e-10
                return masked_error / denom

            def train_gnn(model):
                optimizer = tf.keras.optimizers.Adam(learning_rate=GNN_CONFIG["learning_rate"])
                losses = []
                for epoch in tqdm(range(GNN_CONFIG["num_epochs"]), desc="Training GCN"):
                    with tf.GradientTape() as tape:
                        preds = model(X_full_tensor, adjacency_norm, training=True)
                        loss = masked_mse(Y_train_tensor, preds, train_mask_tensor)
                    grads = tape.gradient(loss, model.trainable_variables)
                    if any(tf.reduce_any(tf.math.is_nan(g)) for g in grads if g is not None):
                        print(f"    NaN gradients at epoch {epoch}, aborting GCN training.")
                        break
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    losses.append(float(loss.numpy()))
                    if epoch % 50 == 0:
                        print(f"    Epoch {epoch}: Loss={loss.numpy():.6f}")
                return losses

            def evaluate_gnn(model):
                preds = model(X_full_tensor, adjacency_norm, training=False)
                pred_test = tf.gather(preds, test_nodes_tensor)
                y_test_sel = tf.gather(Y_test_tensor, test_nodes_tensor)
                rmse = float(
                    orig_std * tf.sqrt(tf.reduce_mean(tf.square(y_test_sel - pred_test))).numpy()
                )
                print(f"    Test RMSE={rmse:.4f}")
                return rmse

            gnn_model = GNNRegressor(GNN_CONFIG["hidden_dims"], GNN_CONFIG["dropout_rate"])
            train_gnn(gnn_model)
            rmse = evaluate_gnn(gnn_model)
            num_params = int(
                np.sum([np.prod(v.shape.as_list()) for v in gnn_model.trainable_variables])
            )
            results.append(
                {
                    "seed": seed,
                    "model": "GCN",
                    "walks_per_node": None,
                    "lml": None,
                    "rmse": rmse,
                    "nlpd": None,
                    "beta": np.nan,
                    "sigma_f": np.nan,
                    "variance": np.nan,
                    "num_params": num_params,
                }
            )
        except Exception as e:
            print(f"    GCN failed: {e}")
            results.append(
                {
                    "seed": seed,
                    "model": "GCN",
                    "walks_per_node": None,
                    "lml": np.nan,
                    "rmse": np.nan,
                    "nlpd": np.nan,
                    "beta": np.nan,
                    "sigma_f": np.nan,
                    "variance": np.nan,
                    "num_params": np.nan,
                    "error": str(e),
                }
            )
        finally:
            clear_gpu()
        pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/results_seed_{seed}_gnn.csv", index=False)
        return


# ============================================================
# ==============  ENTRY POINT ================================
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) == 1:
        parent_main()
    else:
        seed = int(sys.argv[1])
        experiment_type = sys.argv[2]
        child_main(seed, experiment_type)
