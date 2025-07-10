# experiments/ex3/helpers.py
"""
Utility functions for Example 3.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# 1.  Payoff g                                                                #
# --------------------------------------------------------------------------- #
def payoff_g(paths: np.ndarray) -> torch.Tensor:
    """
    Example 3 payoff  

    Parameters
    ----------
    paths : ndarray, shape (num_paths, N+1, d)
        Lower-triangular state matrix from `embed_state_matrix`  

    Returns
    -------
    torch.FloatTensor  shape (num_paths, time_step)
        Vectorised pay-off for every time slice ≤ current horizon.
    """
    # X_n is stored in column 0 of each row (see embed_state_matrix spec)
    time_step = paths.shape[1]
    payoff_np = paths[:, :time_step, 0]          # (num_paths, time_step)
    return torch.as_tensor(payoff_np, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# 2.  Batch & state extractors                                                #
# --------------------------------------------------------------------------- #
def get_batch(
    paths: np.ndarray,
    epoch: int,
    n: int,
    *,
    device: torch.device,
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Return ``(batch_paths, current_state)`` for training step *n*.
    """
    batch_paths = paths[epoch]                                   # (batch, …)
    # Current state is the row n of the lower-triangular matrix
    state = torch.as_tensor(batch_paths[:, n, :],
                            dtype=torch.float32,
                            device=device)                       # (batch, d)
    return batch_paths, state


# simple state extractor used during evaluation
def get_state_fn(paths: np.ndarray, i: int) -> np.ndarray:
    """
    Slice helper: ``paths`` has shape (num_paths, N+1, d) — return X_i.
    """
    return paths[:, i, :]
