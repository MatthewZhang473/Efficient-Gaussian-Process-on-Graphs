import numpy as np
import pytest


@pytest.fixture
def toy_cycle_adj() -> np.ndarray:
    """Undirected 4-node cycle adjacency (dense)."""
    adj = np.zeros((4, 4))
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


@pytest.fixture
def toy_cycle_csr(toy_cycle_adj):
    """Sparse CSR version of the 4-node cycle."""
    import scipy.sparse as sp

    return sp.csr_matrix(toy_cycle_adj)
