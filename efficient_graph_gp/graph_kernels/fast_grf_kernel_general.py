"""Fast GRF kernel construction on the (normalized) graph Laplacian."""

from typing import Sequence

import numpy as np

from ..random_walk_samplers import Graph, RandomWalk
from .utils import get_normalized_laplacian


def fast_general_grf_kernel(
    adj_matrix: np.ndarray,
    modulator_vector: Sequence[float],
    walks_per_node: int = 50,
    p_halt: float = 0.1,
    max_walk_length: int = 10,
) -> np.ndarray:
    """
    Construct a GRF kernel estimate K ≈ ΦΦᵀ using importance-sampled random walks.

    Args:
        adj_matrix: Dense adjacency matrix (N x N).
        modulator_vector: Length ``max_walk_length`` sequence of modulation weights f_l.
        walks_per_node: Number of walks launched from each node.
        p_halt: Per-step halting probability.
        max_walk_length: Maximum walk length (determines modulation vector length).

    Returns:
        Dense kernel estimate (N x N) computed as ΦΦᵀ.
    """
    laplacian = get_normalized_laplacian(adj_matrix)
    graph = Graph(laplacian)
    random_walk = RandomWalk(graph, seed=42)
    feature_matrices = random_walk.get_random_walk_matrices(
        walks_per_node, p_halt, max_walk_length
    )

    Phi = feature_matrices @ np.asarray(modulator_vector)  # (N, N)
    return Phi @ Phi.T
