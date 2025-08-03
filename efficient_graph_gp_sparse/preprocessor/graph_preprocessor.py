from ..random_walk_samplers_sparse import SparseRandomWalk
from ..utils_sparse import get_normalized_laplacian, SparseLinearOperator
import scipy.sparse as sp
import torch
from typing import List


class GraphPreprocessor:
    """
    Graph Preprocessor for Graph GP models.

    This class preprocesses a graph by computing step matrices using random walks
    on the graph's normalized Laplacian. These step matrices are used in Gaussian
    Process models on graphs.

    Attributes:
        adj_matrix (sp.csr_matrix): The adjacency matrix of the graph.
        walks_per_node (int): Number of random walks per node.
        p_halt (float): Probability of halting the random walk.
        max_walk_length (int): Maximum length of the random walks.
        random_walk_seed (int): Seed for reproducibility.
        step_matrices_scipy (list): Step matrices in scipy CSR format.
        step_matrices_torch (list): Step matrices as PyTorch sparse linear operators.
    """

    def __init__(self, adjacency_matrix: sp.csr_matrix, 
                 walks_per_node: int = 10,
                 p_halt: float = 0.5,
                 max_walk_length: int = 10,
                 random_walk_seed: int = 42,
                 load_from_disk: bool = False,
                 use_tqdm: bool = True
                 ) -> None:
        """
        Initialize the GraphPreprocessor.

        Args:
            adjacency_matrix (sp.csr_matrix): The adjacency matrix of the graph.
            walks_per_node (int): Number of random walks per node.
            p_halt (float): Probability of halting the random walk.
            max_walk_length (int): Maximum length of the random walks.
            random_walk_seed (int): Seed for reproducibility.
            use_tqdm (bool): Whether to use tqdm for progress bars.
            load_from_disk (bool): Whether to load precomputed step matrices from disk.
        """
        # Validate adjacency matrix
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square.")

        self.adj_matrix = adjacency_matrix
        self.walks_per_node = walks_per_node
        self.p_halt = p_halt
        self.max_walk_length = max_walk_length
        self.random_walk_seed = random_walk_seed
        self.use_tqdm = use_tqdm

        if load_from_disk:
            # Placeholder for future implementation
            raise NotImplementedError("Loading from disk is not yet implemented.")

    def preprocess_graph(self) -> List[SparseLinearOperator]:
        """
        Preprocess the graph by computing step matrices.

        Returns:
            List[SparseLinearOperator]: A list of step matrices as PyTorch sparse linear operators.
        """
        # Compute the normalized Laplacian
        laplacian = get_normalized_laplacian(self.adj_matrix)

        # Perform random walks
        random_walk = SparseRandomWalk(laplacian, seed=self.random_walk_seed)
        self.step_matrices_scipy = random_walk.get_random_walk_matrices(
            self.walks_per_node, self.p_halt, self.max_walk_length,
            use_tqdm=self.use_tqdm
        )

        # Convert scipy CSR matrices to PyTorch sparse linear operators
        self.step_matrices_torch = [
            SparseLinearOperator(self.from_scipy_csr(csr_matrix))
            for csr_matrix in self.step_matrices_scipy
        ]

        return self.step_matrices_torch

    @staticmethod
    def from_scipy_csr(scipy_csr: sp.csr_matrix) -> torch.sparse_csr_tensor:
        """
        Convert a scipy CSR matrix to a PyTorch sparse CSR tensor.

        Args:
            scipy_csr (sp.csr_matrix): The scipy CSR matrix to convert.

        Returns:
            torch.sparse_csr_tensor: The converted PyTorch sparse CSR tensor.
        """
        if not isinstance(scipy_csr, sp.csr_matrix):
            raise ValueError("Input must be a scipy CSR matrix.")

        crow_indices = torch.from_numpy(scipy_csr.indptr).long()
        col_indices = torch.from_numpy(scipy_csr.indices).long()
        values = torch.from_numpy(scipy_csr.data).float()

        return torch.sparse_csr_tensor(
            crow_indices, col_indices, values,
            (scipy_csr.shape[0], scipy_csr.shape[1]),
            dtype=torch.float32
        )