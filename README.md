# Efficient Gaussian Processes on Graphs (GRFs)

Reference implementation for the paper *Graph Random Features for Scalable Gaussian Processes*. The repo contains the dense (`efficient_graph_gp`) and sparse (`efficient_graph_gp_sparse`) Graph Random Features (GRF) kernels, plus the experiment scripts used in the paper.

## Install

All dependencies (core + experiments) are pinned in `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Quickstart

Dense GRF kernel on a toy graph:
```python
import numpy as np
from efficient_graph_gp.graph_kernels.fast_grf_kernel_general import fast_general_grf_kernel

adj = np.array([[0,1,1,0],
                [1,0,0,1],
                [1,0,0,1],
                [0,1,1,0]], dtype=float)
modulator = [1.0, 0.5, 0.25]  # f_l terms
K = fast_general_grf_kernel(adj_matrix=adj,
                            modulator_vector=modulator,
                            walks_per_node=50,
                            p_halt=0.1,
                            max_walk_length=len(modulator))
print(K.shape)  # (4, 4)
```

Sparse GRF kernel (CSR adjacency, suitable for CG-based inference):
```python
import numpy as np
import scipy.sparse as sp
from efficient_graph_gp_sparse.graph_kernels_sparse.fast_grf_kernel_general import fast_general_grf_kernel

rows, cols = zip(*[(0,1),(1,0),(1,2),(2,1),(2,3),(3,2),(3,0),(0,3)])
adj_csr = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(4,4))
modulator = [1.0, 0.5, 0.25]
K_sparse = fast_general_grf_kernel(adj_matrix=adj_csr,
                                   modulator_vector=modulator,
                                   walks_per_node=50,
                                   p_halt=0.1,
                                   max_walk_length=len(modulator))
print(K_sparse.shape)  # (4, 4)
```

## Reproducibility notes
- Core logic lives in `efficient_graph_gp/` (dense, GPflow-friendly) and `efficient_graph_gp_sparse/` (sparse, GPyTorch-friendly). These mirror the Graph Random Features constructions and complexity guarantees described in the paper.
- Paper experiments are organized under `experiments/`:
  - `dense/traffic_dataset`: San Jose traffic regression.
  - `dense/cora`: Cora citation classification.
  - `dense/ablation`: ablation of random-walk kernel construction.
  - `sparse/scaling_exp`: scaling benchmarks (O(N^{3/2}) conjugate gradients).
  - `sparse/scalable_bo`: Bayesian optimisation on synthetic/social/wind graphs (uses configs/results in this folder).
  - `sparse/social_networks`: SNAP social graph assets for BO.
- `graph_bo/` contains the BO scripts/configs used in the paper; see its README for details.
- Large datasets are not bundled; use the data-loading scripts in those folders to fetch sources (PEMS traffic, SNAP social graphs, ERA5 wind, etc.).
- Tests: run `pytest -q` for smoke coverage on the GRF kernels and samplers.

## Project structure (high level)
- `efficient_graph_gp/`: dense GRF kernels, modulation functions, random-walk samplers.
- `efficient_graph_gp_sparse/`: sparse GRF kernels, CSR samplers, sparse models/utilities.
- `graph_bo/`: Bayesian optimisation scripts/configs on large graphs (as in the paper).
- `experiments_dense/`, `experiments_sparse/`: notebooks and scripts for regression/classification and scaling studies.
- `archive/`: historical experiments and scratch work.

## Citation
If you use this code, please cite the accompanying paper *Graph Random Features for Scalable Gaussian Processes* (preprint).
