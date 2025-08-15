#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import torch

from bo_utils import (
    setup_gpytorch_settings, create_directories,
    get_cached_data, get_step_matrices, convert_to_device,
    RandomSearch, SparseGRF, BayesianOptimizer,
    save_results, print_summary
)

class Config:
    """Bayesian Optimization experimental configuration"""
    
    def __init__(self):
        # Dataset parameters
        self.N_NODES = int(1e6)
        self.NOISE_STD = 0.1
        
        # Kernel parameters
        self.WALKS_PER_NODE = 1000
        self.P_HALT = 0.1
        self.MAX_WALK_LENGTH = 5
        
        # Training parameters
        self.LEARNING_RATE = 0.01
        self.TRAIN_EPOCHS = 20
        
        # BO parameters
        self.NUM_BO_ITERATIONS = 50
        self.INITIAL_POINTS = int(1e-3 * self.N_NODES)
        self.BATCH_SIZE = int(1e-3 * self.N_NODES)
        self.GP_RETRAIN_INTERVAL = 20
        
        # Random Seeds
        self.DATA_SEED = 42
        self.NUM_BO_RUNS = 3
        self.BO_SEEDS = [100 + i * 10 for i in range(self.NUM_BO_RUNS)]
        
        # Data synthesis parameters - remove kernel_std since it's not used
        self.DATA_PARAMS = {'beta_sample': 1.0, 'noise_std': self.NOISE_STD}
        
        # Setup directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(base_dir, 'synthetic_data')
        self.STEP_MATRICES_DIR = os.path.join(base_dir, 'step_matrices')
        self.RESULTS_DIR = os.path.join(base_dir, 'results')
        
        create_directories(self.DATA_DIR, self.STEP_MATRICES_DIR, self.RESULTS_DIR)

def run_experiment(config, data, step_matrices_device, output_device):
    print(f"🎯 Running BO experiments...")
    print(f"   Fixed data/RW seed: {config.DATA_SEED}")
    print(f"   BO seeds: {config.BO_SEEDS}")
    
    all_results = []
    gt_best_value = float(data['Y'][np.argmax(data['Y'])])
    print(f"   Ground truth best: {gt_best_value:.4f}")
    
    def create_algorithms():
        return {
            'random_search': RandomSearch(config.N_NODES, output_device),
            'sparse_grf': SparseGRF(
                config.N_NODES, output_device, step_matrices_device, 
                config.MAX_WALK_LENGTH, config.LEARNING_RATE,
                config.TRAIN_EPOCHS, config.GP_RETRAIN_INTERVAL
            )
        }
    
    for algo_name in ['random_search', 'sparse_grf']:
        print(f"\n🔬 Running {algo_name} with {len(config.BO_SEEDS)} seeds...")
        
        for bo_seed_idx, bo_seed in enumerate(config.BO_SEEDS):
            print(f"   BO seed {bo_seed_idx + 1}/{len(config.BO_SEEDS)} (seed={bo_seed})")
            
            algorithms = create_algorithms()
            algorithm = algorithms[algo_name]
            
            if hasattr(algorithm, 'reset_cache'):
                algorithm.reset_cache()
            
            optimizer = BayesianOptimizer(algorithm, data['Y'].flatten(), config.INITIAL_POINTS, config.BATCH_SIZE)
            results = optimizer.run_optimization(config.NUM_BO_ITERATIONS, seed=bo_seed, algorithm_name=algo_name.replace('_', ' ').title())
            
            for result in results:
                result.update({
                    'algorithm': algo_name,
                    'bo_seed': bo_seed,
                    'bo_run': bo_seed_idx + 1,
                    'data_seed': config.DATA_SEED,
                    'ground_truth_best': gt_best_value,
                    'n_nodes': config.N_NODES,
                    'batch_size': config.BATCH_SIZE,
                    'retrain_interval': config.GP_RETRAIN_INTERVAL
                })
            
            all_results.extend(results)
            
            del algorithm, algorithms
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return pd.DataFrame(all_results)

def main():
    # Setup
    setup_gpytorch_settings()
    np.random.seed(42)
    torch.manual_seed(42)
    
    output_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {output_device}")
    
    config = Config()
    print(f"📋 Configuration: {config.N_NODES} nodes, {config.NUM_BO_ITERATIONS} iterations, {config.NUM_BO_RUNS} runs")
    
    # Load data and step matrices
    data = get_cached_data(config)
    print(f"✅ Data loaded: {len(data['Y'])} nodes")
    
    step_matrices_scipy = get_step_matrices(data, config)
    step_matrices_device = convert_to_device(step_matrices_scipy, output_device)
    print(f"✅ Step matrices ready")
    
    # Run experiment
    results_df = run_experiment(config, data, step_matrices_device, output_device)
    print(f"\n✅ Experiment completed: {len(results_df)} total experiments")
    
    # Visualize and save results
    save_results(results_df, config)
    print_summary(results_df, config)

if __name__ == "__main__":
    main()