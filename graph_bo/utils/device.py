import torch
import numpy as np
from typing import List
import scipy.sparse as sp

def get_device() -> torch.device:
    """Get the best available device."""
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def cleanup_gpu_memory():
    """Clean up GPU memory if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
