"""Random-walk sampler operating on sparse CSR adjacency for GRF features."""

import os
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

# ---- Global read-only CSR arrays (populated once per worker via initializer)
_G_INDPTR = None
_G_INDICES = None
_G_DATA = None
_G_NUM_NODES = None

def _init_worker(indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, num_nodes: int) -> None:
    """Runs once in each worker: bind globals to parent's CSR memory (fork: CoW)."""
    global _G_INDPTR, _G_INDICES, _G_DATA, _G_NUM_NODES
    _G_INDPTR = indptr
    _G_INDICES = indices
    _G_DATA = data
    _G_NUM_NODES = num_nodes

def _worker_walks(
    args: Tuple[Sequence[int], int, float, int, int, bool]
) -> List[Dict[Tuple[int, int], float]]:
    """Worker: now reads neighbors directly from global CSR arrays."""
    nodes, num_walks, p_halt, max_walk_length, seed, show_progress = args
    rng = np.random.default_rng(seed)

    step_accumulators = [defaultdict(float) for _ in range(max_walk_length)]
    it = tqdm(nodes, desc="Process walks", disable=not show_progress) if show_progress else nodes

    for start_node in it:
        for _ in range(num_walks):
            current_node = start_node
            load = 1.0
            for step in range(max_walk_length):
                # accumulate (start, current)
                step_accumulators[step][(start_node, current_node)] += load

                s = _G_INDPTR[current_node]
                e = _G_INDPTR[current_node + 1]
                degree = e - s
                if degree == 0 or rng.random() < p_halt:
                    break

                # pick neighbor and advance
                next_idx = rng.integers(degree)
                weight = _G_DATA[s + next_idx]
                current_node = _G_INDICES[s + next_idx]
                load *= degree * weight / (1 - p_halt)

    return step_accumulators


class SparseRandomWalk:
    """Sparse random-walk generator on CSR adjacency matrices."""

    def __init__(self, adjacency_matrix: sp.spmatrix, seed: Optional[int] = None) -> None:
        self.adjacency = adjacency_matrix.tocsr()
        self.num_nodes = self.adjacency.shape[0]
        self.seed = seed or 42

        # Keep direct views; no extra copies
        self.indptr = self.adjacency.indptr           # int32
        self.indices = self.adjacency.indices         # int32
        self.data = self.adjacency.data.astype(float, copy=False)

    def get_random_walk_matrices(
        self,
        num_walks: int,
        p_halt: float,
        max_walk_length: int,
        use_tqdm: bool = False,
        n_processes: Optional[int] = None,
    ) -> List[sp.csr_matrix]:
        """
        Perform multiple random walks from every node and collect per-step occupancy matrices.

        Returns:
            List of CSR matrices, one per step, where entry (i, j) counts expected visits to j
            after ``step`` steps when starting from i (normalised by num_walks).
        """
        if n_processes is None:
            n_processes = os.cpu_count()

        chunks = np.array_split(np.arange(self.num_nodes), n_processes)

        # Use fork (Linux) so workers share memory via CoW; also set initializer to bind globals
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=n_processes,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(self.indptr, self.indices, self.data, self.num_nodes),
        ) as executor:

            args = [
                (chunk.tolist(), num_walks, p_halt, max_walk_length, self.seed + i, use_tqdm and i == 0)
                for i, chunk in enumerate(chunks)
            ]

            # Stream results to avoid holding all at once
            futures = [executor.submit(_worker_walks, a) for a in args]
            step_accumulators = [defaultdict(float) for _ in range(max_walk_length)]

            for fut in as_completed(futures):
                result = fut.result()
                for step in range(max_walk_length):
                    for k, v in result[step].items():
                        step_accumulators[step][k] += v

        # Build matrices (unchanged)
        mats = []
        for step in range(max_walk_length):
            acc = step_accumulators[step]
            if not acc:
                mats.append(sp.csr_matrix((self.num_nodes, self.num_nodes)))
                continue

            # Avoid extra Python list churn
            keys = list(acc.keys())
            rows = np.fromiter((r for r, _ in keys), dtype=np.int32, count=len(keys))
            cols = np.fromiter((c for _, c in keys), dtype=np.int32, count=len(keys))
            vals = np.fromiter((acc[k] for k in keys), dtype=float, count=len(keys))

            mats.append(sp.csr_matrix((vals, (rows, cols)), shape=(self.num_nodes, self.num_nodes)) / num_walks)

        return mats

if __name__ == "__main__":
    # Check available CPU cores
    print(f"Available CPU cores: {os.cpu_count()}")
    
    # Example usage
    rows = [0, 0, 0, 1, 1, 2, 2, 3]
    cols = [1, 2, 3, 0, 2, 0, 3, 0]
    data = [1, 1, 1, 1, 1, 1, 1, 1]
    adjacency = sp.csr_matrix((data, (rows, cols)), shape=(4, 4))
    
    walker = SparseRandomWalk(adjacency, seed=42)
    step_matrices = walker.get_random_walk_matrices(
        num_walks=100000, p_halt=0.1, max_walk_length=6, use_tqdm=True, n_processes=2
    )
    
    print(f"Generated {len(step_matrices)} step matrices")
    print("Example step matrix (t=1):")
    print(step_matrices[3].todense())
    print(f"Shape: {step_matrices[0].shape}")
    print(f"Sparsity: {sum(m.nnz for m in step_matrices) / sum(m.shape[0]**2 for m in step_matrices):.3f}")
