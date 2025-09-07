import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'graph_bo'))

from graph_bo.data import graph_data_loader
from graph_bo.utils import (
    RandomSearch, SparseGRF, BFS, BayesianOptimizer,
    setup_gpytorch_settings, save_results, print_summary, 
    print_dataset_info, print_config, get_device,
    cleanup_gpu_memory, load_or_compute_step_matrices
)

class SocialBOConfig:
    """Configuration for social network BO experiments"""
    
    def __init__(self):
        # BO parameters
        self.NUM_BO_ITERATIONS = 10
        self.INITIAL_POINTS = 10
        self.BATCH_SIZE = 1
        self.NUM_BO_RUNS = 5
        
        # GRF kernel parameters
        self.WALKS_PER_NODE = 100
        self.P_HALT = 0.1
        self.MAX_WALK_LENGTH = 5
        
        # Training parameters
        self.LEARNING_RATE = 0.01
        self.TRAIN_EPOCHS = 50
        self.GP_RETRAIN_INTERVAL = 10
        
        # Random seeds
        self.BO_SEEDS = [100 + i * 10 for i in range(self.NUM_BO_RUNS)]
        
        # Directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.STEP_MATRICES_DIR = os.path.join(base_dir, '..', 'data', 'step_matrices')
        self.RESULTS_DIR = os.path.join(base_dir, '..', 'results')
        
        # Create directories
        os.makedirs(self.STEP_MATRICES_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)

def run_experiment(dataset_name, algorithms, config):
    """Run BO experiment on a single dataset"""
    print(f"\n{'='*60}")
    print(f"🎯 Running BO experiments on {dataset_name.upper()} dataset")
    print(f"{'='*60}")
    
    # Load dataset
    A, X, y = graph_data_loader(dataset_name)
    n_nodes = len(X)
    
    print_dataset_info(dataset_name, A, X, y)
    
    # Normalize targets for BO (higher degree = better)
    y_normalized = (y - y.mean()) / y.std()
    gt_best_value = float(y_normalized.max())
    
    # Setup device
    device = get_device()
    print(f"  Device: {device}")
    
    # Get step matrices for GRF if needed
    step_matrices_torch = None
    if 'sparse_grf' in algorithms:
        step_matrices_torch = load_or_compute_step_matrices(
            A, config.WALKS_PER_NODE, config.P_HALT, 
            config.MAX_WALK_LENGTH, config.STEP_MATRICES_DIR, dataset_name
        )
        print(f"✅ Step matrices ready")
    
    all_results = []
    
    for algo_name in algorithms:
        print(f"\n🔬 Running {algo_name} with {len(config.BO_SEEDS)} seeds...")
        
        for bo_seed_idx, bo_seed in enumerate(config.BO_SEEDS):
            print(f"   Seed {bo_seed_idx + 1}/{len(config.BO_SEEDS)} (seed={bo_seed})")
            
            # Create algorithm
            if algo_name == 'random_search':
                algorithm = RandomSearch(n_nodes, device)
            elif algo_name == 'bfs':
                algorithm = BFS(A, n_nodes, device)
            elif algo_name == 'sparse_grf':
                algorithm = SparseGRF(
                    n_nodes, device, step_matrices_torch,
                    config.MAX_WALK_LENGTH, config.LEARNING_RATE,
                    config.TRAIN_EPOCHS, config.GP_RETRAIN_INTERVAL
                )
            else:
                raise ValueError(f"Unknown algorithm: {algo_name}")
            
            # Reset algorithm state
            if hasattr(algorithm, 'reset_cache'):
                algorithm.reset_cache()
            
            # Run BO
            optimizer = BayesianOptimizer(
                algorithm, y_normalized, 
                config.INITIAL_POINTS, config.BATCH_SIZE
            )
            
            results = optimizer.run_optimization(
                config.NUM_BO_ITERATIONS, 
                seed=bo_seed, 
                algorithm_name=algo_name.replace('_', ' ').title()
            )
            
            # Add metadata to results
            for result in results:
                result.update({
                    'algorithm': algo_name,
                    'dataset': dataset_name,
                    'bo_seed': bo_seed,
                    'bo_run': bo_seed_idx + 1,
                    'ground_truth_best': gt_best_value,
                    'n_nodes': n_nodes,
                    'n_edges': A.nnz // 2,
                    'density': A.nnz / (A.shape[0] * A.shape[1])
                })
            
            all_results.extend(results)
            
            # Cleanup
            del algorithm
            cleanup_gpu_memory()
    
    return pd.DataFrame(all_results)

def main():
    parser = argparse.ArgumentParser(description='Run Bayesian Optimization on social networks')
    parser.add_argument('--datasets', nargs='+', 
                       choices=['facebook', 'youtube', 'twitch', 'enron'], 
                       default=['facebook'],
                       help='Datasets to run experiments on')
    parser.add_argument('--algorithms', nargs='+',
                       choices=['random_search', 'bfs', 'sparse_grf'],
                       default=['random_search', 'bfs', 'sparse_grf'],
                       help='BO algorithms to compare')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of BO iterations')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of BO runs per algorithm')
    parser.add_argument('--initial-points', type=int, default=10,
                       help='Number of initial random points')
    
    args = parser.parse_args()
    
    # Setup
    setup_gpytorch_settings()
    np.random.seed(42)
    torch.manual_seed(42)
    
    config = SocialBOConfig()
    config.NUM_BO_ITERATIONS = args.iterations
    config.NUM_BO_RUNS = args.runs
    config.INITIAL_POINTS = args.initial_points
    config.BO_SEEDS = [100 + i * 10 for i in range(config.NUM_BO_RUNS)]
    
    config_dict = {
        'datasets': args.datasets,
        'algorithms': args.algorithms,
        'iterations': config.NUM_BO_ITERATIONS,
        'runs': config.NUM_BO_RUNS,
        'initial_points': config.INITIAL_POINTS
    }
    print_config(config_dict)
    
    # Run experiments
    all_results = []
    for dataset in args.datasets:
        try:
            results_df = run_experiment(dataset, args.algorithms, config)
            all_results.append(results_df)
        except Exception as e:
            print(f"❌ Error with dataset {dataset}: {e}")
    
    if all_results:
        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Save and summarize
        dataset_suffix = "_".join(args.datasets) if len(args.datasets) > 1 else args.datasets[0]
        save_results(combined_results, config.RESULTS_DIR, suffix=f"_{dataset_suffix}")
        print_summary(combined_results)
        
        # Print final experiment summary
        print(f"\n{'='*60}")
        print("📊 EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        
        final_results = combined_results[combined_results['iteration'] == combined_results['iteration'].max()]
        
        for dataset in combined_results['dataset'].unique():
            print(f"\n{dataset.upper()}:")
            dataset_final = final_results[final_results['dataset'] == dataset]
            
            for algo in dataset_final['algorithm'].unique():
                algo_final = dataset_final[dataset_final['algorithm'] == algo]
                mean_best = algo_final['best_value'].mean()
                std_best = algo_final['best_value'].std()
                mean_regret = algo_final['regret'].mean()
                print(f"  {algo:15s}: best = {mean_best:.4f} ± {std_best:.4f}, regret = {mean_regret:.4f}")
        
        print(f"\n✅ All experiments completed!")
        print(f"   Total experiments: {len(combined_results)}")
    else:
        print("❌ No experiments completed successfully")

if __name__ == "__main__":
    main()