import os
import numpy as np
import pandas as pd
import pickle
import pandas as pd
import networkx as nx
import gzip
from scipy.sparse import csr_matrix
from typing import Tuple, Dict, Any
import warnings

class GraphDataLoader:
    """Database class for loading and caching graph datasets."""
    
    def __init__(self, data_root="../../experiments_sparse/social_networks", cache_dir=None):
        # Set default cache directory relative to this module's location
        if cache_dir is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            cache_dir = os.path.join(module_dir, "processed_data")
        
        self.data_root = data_root
        self.cache_dir = cache_dir
        self._cache = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Cache directory: {os.path.abspath(cache_dir)}")
        
        # Dataset configurations
        self.dataset_configs = {
            'facebook': {
                'edges_file': 'facebook_large/musae_facebook_edges.csv',
                'targets_file': 'facebook_large/musae_facebook_target.csv',
                'loader': self._load_facebook
            },
            'youtube': {
                'graph_file': 'com-youtube.ungraph.txt.gz',
                'loader': self._load_youtube
            },
            'twitch': {
                'edges_file': 'large_twitch_edges.csv',
                'features_file': 'large_twitch_features.csv',
                'loader': self._load_twitch
            },
            'enron': {
                'graph_file': 'email-Enron.txt.gz',
                'loader': self._load_enron
            }
        }
    
    def __call__(self, dataset_name: str, force_reload: bool = False) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """
        Load dataset with caching.
        
        Args:
            dataset_name: Name of the dataset
            force_reload: Force reload from source files
            
        Returns:
            Tuple of (adjacency_matrix, node_indices, node_degrees)
        """
        if dataset_name not in self.dataset_configs:
            available = list(self.dataset_configs.keys())
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
        
        # Check memory cache first
        if not force_reload and dataset_name in self._cache:
            return self._cache[dataset_name]
        
        # Check disk cache
        cache_path = os.path.join(self.cache_dir, f"{dataset_name}.pkl")
        if not force_reload and os.path.exists(cache_path):
            print(f"Loading {dataset_name} from cache...")
            data = self._load_from_cache(cache_path)
            self._cache[dataset_name] = data
            return data
        
        # Load from source and cache
        print(f"Loading {dataset_name} from source files...")
        data = self._load_from_source(dataset_name)
        
        # Save to disk cache
        self._save_to_cache(data, cache_path, dataset_name)
        
        # Save to memory cache
        self._cache[dataset_name] = data
        
        return data
    
    def _load_from_source(self, dataset_name: str) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """Load dataset from source files."""
        config = self.dataset_configs[dataset_name]
        dataset_path = os.path.join(self.data_root, dataset_name)
        
        # Check if dataset directory exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        
        # Call the appropriate loader
        return config['loader'](dataset_path)
    
    def _load_facebook(self, dataset_path: str) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """Load Facebook dataset."""
        edges_path = os.path.join(dataset_path, "facebook_large/musae_facebook_edges.csv")
        
        if not os.path.exists(edges_path):
            raise FileNotFoundError(f"Facebook edges file not found: {edges_path}")
        
        # Load edges
        edges_df = pd.read_csv(edges_path)
        
        # Create NetworkX graph
        G = nx.from_pandas_edgelist(edges_df, source='id_1', target='id_2')
        
        # Convert to adjacency matrix and enforce CSR format
        adjacency_matrix = nx.adjacency_matrix(G).tocsr()
        
        # Get node indices and degrees
        X = np.array(list(G.nodes()))
        y = np.array([G.degree(node) for node in X])
        
        return adjacency_matrix, X, y
    
    def _load_youtube(self, dataset_path: str) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """Load YouTube dataset."""
        graph_path = os.path.join(dataset_path, "com-youtube.ungraph.txt.gz")
        
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"YouTube graph file not found: {graph_path}")
        
        G = nx.Graph()
        
        with gzip.open(graph_path, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                u, v = map(int, line.strip().split())
                G.add_edge(u, v)
        
        # Convert to adjacency matrix
        adjacency_matrix = nx.adjacency_matrix(G).tocsr()
        
        # Get node indices and degrees
        X = np.array(list(G.nodes()))
        y = np.array([G.degree(node) for node in X])
        
        return adjacency_matrix, X, y
    
    def _load_twitch(self, dataset_path: str) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """Load Twitch dataset."""
        edges_path = os.path.join(dataset_path, "large_twitch_edges.csv")
        
        if not os.path.exists(edges_path):
            raise FileNotFoundError(f"Twitch edges file not found: {edges_path}")
        
        # Load edges
        edges_df = pd.read_csv(edges_path)
        
        # Create NetworkX graph
        G = nx.from_pandas_edgelist(edges_df, source='numeric_id_1', target='numeric_id_2')
        
        # Convert to adjacency matrix
        adjacency_matrix = nx.adjacency_matrix(G).tocsr()
        
        # Get node indices and degrees
        X = np.array(list(G.nodes()))
        y = np.array([G.degree(node) for node in X])
        
        return adjacency_matrix, X, y
    
    def _load_enron(self, dataset_path: str) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """Load Enron dataset."""
        graph_path = os.path.join(dataset_path, "email-Enron.txt.gz")
        
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Enron graph file not found: {graph_path}")
        
        G = nx.Graph()
        
        with gzip.open(graph_path, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                u, v = map(int, line.strip().split())
                G.add_edge(u, v)
        
        # Convert to adjacency matrix
        adjacency_matrix = nx.adjacency_matrix(G).tocsr()
        
        # Get node indices and degrees
        X = np.array(list(G.nodes()))
        y = np.array([G.degree(node) for node in X])
        
        return adjacency_matrix, X, y
    
    def _load_from_cache(self, cache_path: str) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """Load dataset from cache file."""
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        return data['adjacency_matrix'], data['node_indices'], data['node_degrees']
    
    def _save_to_cache(self, data: Tuple[csr_matrix, np.ndarray, np.ndarray], cache_path: str, dataset_name: str):
        """Save dataset to cache file."""
        adjacency_matrix, X, y = data
        
        cache_data = {
            'adjacency_matrix': adjacency_matrix,
            'node_indices': X,
            'node_degrees': y,
            'num_nodes': len(X),
            'num_edges': adjacency_matrix.nnz // 2,
            'density': adjacency_matrix.nnz / (adjacency_matrix.shape[0] * adjacency_matrix.shape[1]),
            'dataset_name': dataset_name
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Cached {dataset_name}:")
        print(f"  Nodes: {cache_data['num_nodes']}")
        print(f"  Edges: {cache_data['num_edges']}")
        print(f"  Density: {cache_data['density']:.6f}")
        print(f"  Degree range: {y.min()} to {y.max()}")
    
    def list_available_datasets(self) -> list:
        """Return list of available datasets."""
        return list(self.dataset_configs.keys())
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a dataset without loading it."""
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        cache_path = os.path.join(self.cache_dir, f"{dataset_name}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return {
                'num_nodes': data['num_nodes'],
                'num_edges': data['num_edges'],
                'density': data['density'],
                'cached': True
            }
        else:
            return {'cached': False}
    
    def clear_cache(self, dataset_name: str = None):
        """Clear cache for specific dataset or all datasets."""
        if dataset_name:
            # Clear specific dataset
            cache_path = os.path.join(self.cache_dir, f"{dataset_name}.pkl")
            if os.path.exists(cache_path):
                os.remove(cache_path)
            if dataset_name in self._cache:
                del self._cache[dataset_name]
            print(f"Cleared cache for {dataset_name}")
        else:
            # Clear all cache
            if os.path.exists(self.cache_dir):
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.pkl'):
                        os.remove(os.path.join(self.cache_dir, file))
            self._cache.clear()
            print("Cleared all cache")

# Create global instance
graph_data_loader = GraphDataLoader()
