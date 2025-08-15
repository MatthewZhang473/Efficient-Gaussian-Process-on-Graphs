import torch
import gpytorch
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from .gp_models import SparseGraphGP

class Algorithm(ABC):
    def __init__(self, n_nodes, device):
        self.n_nodes, self.device = n_nodes, device
    
    @abstractmethod
    def select_next_points(self, X_observed, Y_observed):
        pass
    
    @abstractmethod
    def update(self, X_observed, Y_observed):
        pass

class RandomSearch(Algorithm):
    def select_next_points(self, X_observed, Y_observed, batch_size=1):
        return np.random.choice(self.n_nodes, size=batch_size, replace=False)
    def update(self, X_observed, Y_observed):
        pass

class SparseGRF(Algorithm):
    def __init__(self, n_nodes, device, step_matrices, max_walk_length, learning_rate, train_epochs, retrain_interval):
        super().__init__(n_nodes, device)
        self.step_matrices = step_matrices
        self.max_walk_length = max_walk_length
        self.learning_rate = learning_rate
        self.train_epochs = train_epochs
        self.retrain_interval = retrain_interval
        self.cached_model = None
        self.cached_likelihood = None
        self.last_training_size = 0
    
    def reset_cache(self):
        self.cached_model = self.cached_likelihood = None
        self.last_training_size = 0
    
    def _should_retrain(self, current_size):
        return (self.cached_model is None or 
                self.retrain_interval == 0 or 
                (current_size - self.last_training_size) >= self.retrain_interval)
    
    def _train_model(self, X_observed, Y_observed):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        model = SparseGraphGP(X_observed, Y_observed, likelihood, self.step_matrices, self.max_walk_length).to(self.device)
        
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        for _ in range(self.train_epochs):
            optimizer.zero_grad()
            loss = -mll(model(X_observed), Y_observed)
            loss.backward()
            optimizer.step()
        
        self.cached_model, self.cached_likelihood = model, likelihood
        self.last_training_size = len(X_observed)
        return model, likelihood
    
    def select_next_points(self, X_observed, Y_observed, batch_size=1):
        current_size = len(X_observed)
        
        if self._should_retrain(current_size):
            model, likelihood = self._train_model(X_observed, Y_observed)
        else:
            model, likelihood = self.cached_model, self.cached_likelihood
        
        model.eval()
        likelihood.eval()
        
        X_all = torch.arange(self.n_nodes, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        selected_indices = []
        with torch.no_grad():
            thompson_samples = model.predict(X_all, n_samples=1)
            selected_indices = torch.topk(thompson_samples[0, :], batch_size).indices.tolist()
        
        return selected_indices

    def update(self, X_observed, Y_observed):
        self.cached_model.x_train = X_observed
        self.cached_model.y_train = Y_observed

class BayesianOptimizer:
    def __init__(self, algorithm, objective_values, initial_points=10, batch_size=1):
        self.algorithm = algorithm
        self.objective_values = objective_values
        self.n_nodes = len(objective_values)
        self.initial_points = initial_points
        self.batch_size = batch_size
        self.gt_best_value = float(objective_values[np.argmax(objective_values)])
    
    def run_optimization(self, n_iterations, seed=None, algorithm_name="BO"):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        results = []
        observed_indices = np.random.choice(self.n_nodes, self.initial_points, replace=False)
        
        X_observed = torch.tensor(observed_indices.reshape(-1, 1), dtype=torch.float32, device=self.algorithm.device)
        Y_observed = torch.tensor(self.objective_values[observed_indices].flatten(), dtype=torch.float32, device=self.algorithm.device)
        
        best_value = float(Y_observed.max())
        best_idx = observed_indices[torch.argmax(Y_observed).item()]
        
        with tqdm(range(n_iterations), desc=f"    {algorithm_name}", leave=False) as pbar:
            for iteration in pbar:
                next_indices = self.algorithm.select_next_points(X_observed, Y_observed, self.batch_size)
                
                batch_results = []
                for next_idx in next_indices:
                    next_value = float(self.objective_values[next_idx])
                    if next_value > best_value:
                        best_value, best_idx = next_value, next_idx
                    
                    batch_results.append({'point': next_idx, 'value': next_value})
                    observed_indices = np.append(observed_indices, next_idx)
                
                X_observed = torch.tensor(observed_indices.reshape(-1, 1), dtype=torch.float32, device=self.algorithm.device)
                Y_observed = torch.tensor(self.objective_values[observed_indices].flatten(), dtype=torch.float32, device=self.algorithm.device)
                self.algorithm.update(X_observed, Y_observed)

                for batch_idx, batch_result in enumerate(batch_results):
                    results.append({
                        'iteration': iteration + 1,
                        'batch_idx': batch_idx,
                        'next_point': batch_result['point'],
                        'next_value': batch_result['value'],
                        'best_value': best_value,
                        'best_point': best_idx,
                        'regret': self.gt_best_value - best_value,
                        'dataset_size': len(observed_indices)
                    })
                
                pbar.set_postfix({
                    'best': f'{best_value:.4f}',
                    'regret': f'{(self.gt_best_value - best_value):.4f}',
                    'data': len(observed_indices)
                })
        
        return results
