import matplotlib.pyplot as plt

def plot_results(bo_results_df, config):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for algorithm in bo_results_df['algorithm'].unique():
        algo_data = bo_results_df[bo_results_df['algorithm'] == algorithm]
        mean_values = algo_data.groupby('iteration')['best_value'].mean()
        std_values = algo_data.groupby('iteration')['best_value'].std()
        
        ax1.plot(mean_values.index, mean_values.values, marker='o', label=f'{algorithm} (mean)', linewidth=2)
        ax1.fill_between(mean_values.index, mean_values.values - std_values.values, mean_values.values + std_values.values, alpha=0.2)
    
    ax1.axhline(y=bo_results_df['ground_truth_best'].iloc[0], color='red', linestyle='--', label='Ground Truth', linewidth=2)
    ax1.set_xlabel('BO Iteration')
    ax1.set_ylabel('Best Value Found')
    ax1.set_title(f'BO Convergence: Best Value (n={len(config.BO_SEEDS)} seeds)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for algorithm in bo_results_df['algorithm'].unique():
        algo_data = bo_results_df[bo_results_df['algorithm'] == algorithm]
        mean_regret = algo_data.groupby('iteration')['regret'].mean()
        std_regret = algo_data.groupby('iteration')['regret'].std()
        
        ax2.plot(mean_regret.index, mean_regret.values, marker='o', label=f'{algorithm} (mean)', linewidth=2)
        ax2.fill_between(mean_regret.index, mean_regret.values - std_regret.values, mean_regret.values + std_regret.values, alpha=0.2)
    
    ax2.set_xlabel('BO Iteration')
    ax2.set_ylabel('Regret')
    ax2.set_title(f'BO Convergence: Regret (n={len(config.BO_SEEDS)} seeds)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
