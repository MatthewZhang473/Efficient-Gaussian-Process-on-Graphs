import numpy as np
import scipy.sparse as sp
import networkx as nx
import time
import sys
import os
import psutil
import gc

# Add the parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from efficient_graph_gp.random_walk_samplers.sampler import RandomWalk as DenseRandomWalk, Graph as DenseGraph
from random_walk_samplers.sparse_sampler import SparseRandomWalk

def create_test_graph(n_nodes=50, avg_degree=4, seed=42):
    """Create a test graph and return both NetworkX and adjacency matrix formats."""
    np.random.seed(seed)
    p = avg_degree / (n_nodes - 1)
    G = nx.erdos_renyi_graph(n_nodes, p, seed=seed)
    
    # # Ensure connected
    # if not nx.is_connected(G):
    #     components = list(nx.connected_components(G))
    #     for i in range(len(components) - 1):
    #         node1 = list(components[i])[0]
    #         node2 = list(components[i + 1])[0]
    #         G.add_edge(node1, node2)
    
    # Get adjacency matrix
    adj_matrix = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).toarray().astype(float)
    return G, adj_matrix

def compare_outputs(dense_result, sparse_result, tolerance=1e-10):
    """Compare outputs from dense and sparse implementations."""
    if dense_result is None or sparse_result is None:
        return False, "One of the results is None"
    
    # Convert sparse result to dense format for comparison
    if isinstance(sparse_result, list):
        # Sparse returns list of CSR matrices
        n_nodes = len(sparse_result)
        max_walk_length = sparse_result[0].shape[1]
        sparse_dense = np.zeros((n_nodes, n_nodes, max_walk_length))
        for i, matrix in enumerate(sparse_result):
            sparse_dense[i] = matrix.toarray()
    else:
        sparse_dense = sparse_result
    
    # Check shapes
    if dense_result.shape != sparse_dense.shape:
        return False, f"Shape mismatch: dense {dense_result.shape} vs sparse {sparse_dense.shape}"
    
    # Check values
    max_diff = np.max(np.abs(dense_result - sparse_dense))
    if max_diff > tolerance:
        return False, f"Value mismatch: max difference {max_diff:.2e} > tolerance {tolerance:.2e}"
    
    return True, f"Match within tolerance (max diff: {max_diff:.2e})"

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def run_timing_test(dense_sampler, sparse_sampler, num_walks, p_halt, max_walk_length, num_runs=3):
    """Run timing tests for both samplers with improved memory tracking."""
    dense_times = []
    sparse_times = []
    dense_memory_peak = []
    sparse_memory_peak = []
    dense_result = None
    sparse_result = None
    
    for run in range(num_runs):
        # Force multiple garbage collection cycles
        for _ in range(3):
            gc.collect()
        
        # Stabilize memory baseline
        time.sleep(0.1)
        
        # Dense implementation with peak memory tracking
        mem_baseline = get_memory_usage()
        start_time = time.time()
        dense_result = dense_sampler.get_random_walk_matrices(num_walks, p_halt, max_walk_length)
        dense_time = time.time() - start_time
        mem_peak = get_memory_usage()
        dense_memory_delta = max(0, mem_peak - mem_baseline)  # Ensure non-negative
        
        dense_times.append(dense_time)
        dense_memory_peak.append(dense_memory_delta)
        
        # Clear and stabilize before sparse test
        if run < num_runs - 1:
            del dense_result
        for _ in range(3):
            gc.collect()
        time.sleep(0.1)
        
        # Sparse implementation with peak memory tracking
        mem_baseline = get_memory_usage()
        start_time = time.time()
        sparse_result = sparse_sampler.get_random_walk_matrices(num_walks, p_halt, max_walk_length)
        sparse_time = time.time() - start_time
        mem_peak = get_memory_usage()
        sparse_memory_delta = max(0, mem_peak - mem_baseline)  # Ensure non-negative
        
        sparse_times.append(sparse_time)
        sparse_memory_peak.append(sparse_memory_delta)
        
        if run < num_runs - 1:
            del sparse_result
    
    return {
        'dense_times': dense_times,
        'sparse_times': sparse_times,
        'dense_memory': dense_memory_peak,
        'sparse_memory': sparse_memory_peak,
        'dense_result': dense_result,
        'sparse_result': sparse_result
    }

def main():
    print("=== Random Walk Sampler Comparison ===\n")
    print(f"Python process PID: {os.getpid()}")
    print(f"Initial memory usage: {get_memory_usage():.2f} MB\n")
    
    # Test parameters
    test_configs = [
        {'n_nodes': 100, 'avg_degree': 10, 'num_walks': 100, 'max_walk_length': 4},
    ]
    
    p_halt = 0.1
    seed = 42
    
    for i, config in enumerate(test_configs):
        print(f"Test {i+1}: {config['n_nodes']} nodes, avg_degree={config['avg_degree']}")
        print("-" * 50)
        
        # Create test graph
        G, adj_matrix = create_test_graph(
            config['n_nodes'], 
            config['avg_degree'], 
            seed=seed
        )
        
        # Create samplers
        dense_graph = DenseGraph(adj_matrix)
        dense_sampler = DenseRandomWalk(dense_graph, seed=seed)
        
        sparse_adj = sp.csr_matrix(adj_matrix)
        sparse_sampler = SparseRandomWalk(sparse_adj, seed=seed)
        
        # Run timing test with memory tracking
        timing_results = run_timing_test(
            dense_sampler, 
            sparse_sampler,
            config['num_walks'],
            p_halt,
            config['max_walk_length'],
            num_runs=3
        )
        
        # Check correctness
        is_correct, message = compare_outputs(
            timing_results['dense_result'],
            timing_results['sparse_result']
        )
        
        # Debug output - remove these lines after testing
        if timing_results['dense_result'] is not None:
            print("Dense result shape:", timing_results['dense_result'].shape)
            print("Dense sample:\n", timing_results['dense_result'][0])
        
        if timing_results['sparse_result'] is not None:
            print("Sparse result length:", len(timing_results['sparse_result']))
            print("Sparse sample:\n", timing_results['sparse_result'][0].todense())

        # Report results with memory information
        dense_avg = np.mean(timing_results['dense_times'])
        sparse_avg = np.mean(timing_results['sparse_times'])
        speedup = dense_avg / sparse_avg
        
        dense_mem_avg = np.mean(timing_results['dense_memory'])
        sparse_mem_avg = np.mean(timing_results['sparse_memory'])
        
        print(f"Correctness: {'✓ PASS' if is_correct else '✗ FAIL'} - {message}")
        print(f"Dense time:  {dense_avg:.4f}s ± {np.std(timing_results['dense_times']):.4f}s")
        print(f"Sparse time: {sparse_avg:.4f}s ± {np.std(timing_results['sparse_times']):.4f}s")
        print(f"Speedup:     {speedup:.2f}x {'(sparse faster)' if speedup > 1 else '(dense faster)'}")
        
        # Physical memory usage
        print(f"Dense memory peak:   {dense_mem_avg:.2f} MB ± {np.std(timing_results['dense_memory']):.2f} MB")
        print(f"Sparse memory peak:  {sparse_mem_avg:.2f} MB ± {np.std(timing_results['sparse_memory']):.2f} MB")
        
        if sparse_mem_avg > 0:
            mem_ratio = dense_mem_avg / sparse_mem_avg
            print(f"Memory efficiency:   {mem_ratio:.2f}x {'(sparse uses less)' if mem_ratio > 1 else '(dense uses less)'}")
        
        # Object memory usage estimate
        if timing_results['dense_result'] is not None:
            dense_obj_memory = timing_results['dense_result'].nbytes / 1024**2
            sparse_nnz = sum(m.nnz for m in timing_results['sparse_result'])
            sparse_obj_memory = sparse_nnz * 8 / 1024**2  # Approximate MB
            
            print(f"Dense object size:   {dense_obj_memory:.2f} MB")
            print(f"Sparse object size:  {sparse_obj_memory:.2f} MB (est.)")
        
        print(f"Final memory usage:  {get_memory_usage():.2f} MB\n")
    
    print("=== Summary ===")
    print("Physical memory measurements show actual RAM usage during execution.")
    print("Memory delta = peak memory during execution - memory before execution")
    print("If all tests show '✓ PASS', the sparse implementation is correct.")
    print("Sparse implementation should be faster and use less memory for large sparse graphs.")

if __name__ == "__main__":
    main()
