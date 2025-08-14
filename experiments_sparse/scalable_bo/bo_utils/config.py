import os
import gpytorch
from gpytorch import settings as gsettings
from linear_operator import settings

def setup_gpytorch_settings():
    """Configure GPyTorch and Linear Operator settings"""
    settings.verbose_linalg._default = False
    settings._fast_covar_root_decomposition._default = False
    gsettings.max_cholesky_size._global_value = 0
    gsettings.cg_tolerance._global_value = 1e-2
    gsettings.max_lanczos_quadrature_iterations._global_value = 1
    settings.fast_computations.log_prob._state = True
    gsettings.num_trace_samples._global_value = 64
    gsettings.min_preconditioning_size._global_value = 1e10

def create_directories(data_dir, step_matrices_dir, results_dir):
    """Create necessary directories"""
    for dir_path in [data_dir, step_matrices_dir, results_dir]:
        os.makedirs(dir_path, exist_ok=True)

class Config:
    def __init__(self):
        setup_gpytorch_settings()
        
        self.N_NODES = int(1e2)
        self.NOISE_STD = 0.1
        self.WALKS_PER_NODE = 10000
        self.P_HALT = 0.1
        self.MAX_WALK_LENGTH = 3
        self.LEARNING_RATE = 0.01
        self.TRAIN_EPOCHS = 50
        self.NUM_BO_ITERATIONS = 20
        self.INITIAL_POINTS = 10
        self.BATCH_SIZE = 2
        self.GP_RETRAIN_INTERVAL = 5
        
        self.DATA_SEED = 42
        self.NUM_BO_RUNS = 3
        self.BO_SEEDS = [100 + i * 10 for i in range(self.NUM_BO_RUNS)]
        
        self.DATA_PARAMS = {'beta_sample': 1.0, 'kernel_std': 1.0, 'noise_std': self.NOISE_STD}
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(base_dir, 'synthetic_data')
        self.STEP_MATRICES_DIR = os.path.join(base_dir, 'step_matrices')
        self.RESULTS_DIR = os.path.join(base_dir, 'results')
        
        create_directories(self.DATA_DIR, self.STEP_MATRICES_DIR, self.RESULTS_DIR)
