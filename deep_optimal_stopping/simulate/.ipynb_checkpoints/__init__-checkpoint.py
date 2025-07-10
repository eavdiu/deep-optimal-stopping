# deep_optimal_stopping/simulate/__init__.py
"""
Public API for the simulation sub-package.

Import from here to keep user code tidy, e.g.:

    from deep_optimal_stopping.simulate import simulate_paths_bs
    from deep_optimal_stopping.simulate import simulate_fbm
"""

# --- GBM simulators (Examples 1 & 2) ----------------------------------------
from .bs import (
    simulate_paths_bs,           # Example 1
    simulate_paths_bs_barrier,   # Example 2
)

# --- fBm simulators & helpers (Example 3) -----------------------------------
from .fbm import (
    simulate_fbm,
    embed_state_matrix,
    compute_w_tilde_fbm,
    compute_w_mix,
    compute_z_tilde,
)

__all__ = [
    # GBM
    "simulate_paths_bs",
    "simulate_paths_bs_barrier",
    # fBm
    "simulate_fbm",
    "embed_state_matrix",
    "compute_w_tilde_fbm",
    "compute_w_mix",
    "compute_z_tilde",
]
