import numpy as np

from efficient_graph_gp.graph_kernels.fast_grf_kernel_general import fast_general_grf_kernel
from efficient_graph_gp.random_walk_samplers.sampler import RandomWalk, Graph


def test_random_walk_shapes(toy_cycle_adj):
    graph = Graph(toy_cycle_adj)
    rw = RandomWalk(graph, seed=0)
    mats = rw.get_random_walk_matrices(num_walks=5, p_halt=0.2, max_walk_length=3, n_processes=1)
    assert mats.shape == (4, 4, 3)
    # first step should include identity because walks start at node
    assert np.allclose(np.diag(mats[:, :, 0]), 1.0, atol=1e-6)


def test_fast_general_grf_kernel_psd(toy_cycle_adj):
    modulator = np.array([1.0, 0.5, 0.25])
    k = fast_general_grf_kernel(
        adj_matrix=toy_cycle_adj,
        modulator_vector=modulator,
        walks_per_node=10,
        p_halt=0.2,
        max_walk_length=3,
    )
    # symmetry
    assert np.allclose(k, k.T, atol=1e-8)
    # eigenvalues should be non-negative up to small numerical noise
    eigvals = np.linalg.eigvalsh(k)
    assert eigvals.min() >= -1e-8
