# deep_optimal_stopping/utils.py
import numpy as np, torch, random

def set_global_seed(seed: int) -> None:
    """Seed NumPy, Python `random`, and Torch (CPU & CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   
