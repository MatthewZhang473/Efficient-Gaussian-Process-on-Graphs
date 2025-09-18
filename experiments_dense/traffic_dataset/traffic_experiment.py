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
    # NLPD mode: "predict_y" (recommended) or "fullcov"
    "nlpd_mode": "predict_y",
}

RESULTS_DIR = "experiments_dense/traffic_dataset/results"


# ============================================================
# ==============  PARENT ORCHESTRATION =======================
# ============================================================
def parent_main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_dfs = []

    for seed in tqdm(CONFIG["seeds"], desc="Seeds"):
        print(f"\n[Parent] Running PoFM for seed={seed}")
        subprocess.run([sys.executable, "-u", __file__, str(seed), "pofm"], check=True)

        print(f"[Parent] Running Diffusion for seed={seed}")
        subprocess.run([sys.executable, "-u", __file__, str(seed), "diffusion"], check=True)

        print(f"[Parent] Running GRF for seed={seed}")
        subprocess.run([sys.executable, "-u", __file__, str(seed), "grf"], check=True)

        print(f"[Parent] Running A-GRF for seed={seed}")
        subprocess.run([sys.executable, "-u", __file__, str(seed), "agrf"], check=True)

        # Combine per-seed results (each model saved separately)
        dfs = []
        for tag in ["pofm", "diffusion", "grf", "agrf"]:
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
