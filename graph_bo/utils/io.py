import os
import pandas as pd
from typing import Dict, Any

def save_results(results_df: pd.DataFrame, results_dir: str, suffix: str = "") -> str:
    """
    Save experiment results to CSV files.
    
    Args:
        results_df: DataFrame containing experiment results
        results_dir: Directory to save results
        suffix: Optional suffix for filename
        
    Returns:
        Path to saved results file
    """
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bo_results_{timestamp}{suffix}.csv"
    filepath = os.path.join(results_dir, filename)
    
    # Save main results
    results_df.to_csv(filepath, index=False)
    print(f"💾 Results saved to: {filepath}")
    
    # Save summary statistics
    summary = compute_summary_stats(results_df)
    summary_file = filepath.replace('.csv', '_summary.csv')
    summary.to_csv(summary_file)
    print(f"📊 Summary saved to: {summary_file}")
    
    return filepath

def compute_summary_stats(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for experiment results."""
    return results_df.groupby(['dataset', 'algorithm']).agg({
        'best_value': ['mean', 'std', 'max'],
        'regret': ['mean', 'std', 'min'],
        'iteration': 'count'
    }).round(4)

def print_summary(results_df: pd.DataFrame) -> None:
    """Print experiment summary to console."""
    print(f"\n{'='*60}")
    print("📊 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    final_results = results_df[results_df['iteration'] == results_df['iteration'].max()]
    
    for dataset in results_df['dataset'].unique():
        print(f"\n{dataset.upper()}:")
        dataset_final = final_results[final_results['dataset'] == dataset]
        
        for algo in dataset_final['algorithm'].unique():
            algo_final = dataset_final[dataset_final['algorithm'] == algo]
            mean_best = algo_final['best_value'].mean()
            std_best = algo_final['best_value'].std()
            mean_regret = algo_final['regret'].mean()
            print(f"  {algo:15s}: best = {mean_best:.4f} ± {std_best:.4f}, regret = {mean_regret:.4f}")

def print_dataset_info(dataset_name: str, A, X, y) -> None:
    """Print dataset information."""
    print(f"Dataset info:")
    print(f"  Nodes: {len(X):,}")
    print(f"  Edges: {A.nnz//2:,}")
    print(f"  Density: {A.nnz/(A.shape[0]*A.shape[1]):.6f}")
    print(f"  Target range: {y.min():.2f} to {y.max():.2f}")

def print_config(config: Dict[str, Any]) -> None:
    """Print experiment configuration."""
    print(f"📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
