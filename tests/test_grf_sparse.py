import numpy as np

from efficient_graph_gp_sparse.graph_kernels_sparse.fast_grf_kernel_general import (
    fast_general_grf_kernel,
)
from efficient_graph_gp_sparse.random_walk_samplers_sparse.sparse_sampler import SparseRandomWalk


def test_sparse_random_walk_shapes(toy_cycle_csr):
    rw = SparseRandomWalk(toy_cycle_csr, seed=0)
    mats = rw.get_random_walk_matrices(num_walks=5, p_halt=0.2, max_walk_length=3, n_processes=1)
    assert len(mats) == 3
    for m in mats:
        assert m.shape == (4, 4)
    # diagonal occupancy on step 0 should be positive
    assert np.allclose(mats[0].diagonal(), 1.0, atol=1e-6)


def test_fast_general_grf_kernel_sparse_psd(toy_cycle_csr):
    modulator = np.array([1.0, 0.5, 0.25])
    k = fast_general_grf_kernel(
        adj_matrix=toy_cycle_csr,
        modulator_vector=modulator,
        walks_per_node=10,
        p_halt=0.2,
        max_walk_length=3,
    )
    k_dense = k.toarray()
    assert np.allclose(k_dense, k_dense.T, atol=1e-8)
    eigvals = np.linalg.eigvalsh(k_dense)
    assert eigvals.min() >= -1e-8
