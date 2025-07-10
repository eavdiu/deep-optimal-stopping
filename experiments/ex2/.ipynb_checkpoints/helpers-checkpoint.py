#experiments/ex2/helpers.py
"""
Utility functions for Example 2.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# 1.  Payoff helpers                                                          #
# --------------------------------------------------------------------------- #
def _coupon_or_floor(
    paths: np.ndarray,
    K: float,
    F: float,
) -> np.ndarray:
    """
    For each path & time step return

        F      if min_{asset}(S) > K   (i.e. no breach),
        min S  otherwise.

    Returns
    -------
    np.ndarray, shape (num_paths, time_step)
    """
    asset_min = paths[:, :-1, :].min(axis=1)         
    return np.where(asset_min > K, F, asset_min)


def payoff_g(
    paths: np.ndarray,
    *,
    T: float,
    N: int,
    K: float,
    F: float,
    c: float,
    r: float,
) -> torch.Tensor:
    """
    Vectorised cash-flow for Example 2.

    Returns
    -------
    torch.FloatTensor  shape (num_paths, time_step)
    """
    num_paths, _, t_step = paths.shape
    dt        = T / N

    start = max(1, N + 1 - t_step)
    idx   = np.arange(start, N + 1)              # length = step - (start==0)
    disc  = np.exp(-r * idx * dt).astype(np.float32)

    payoff = np.zeros((num_paths, t_step), dtype=np.float32)

    discount_repeated = np.tile(disc, (num_paths, 1))
    disc[(N+1-t_step+1)*(N+1==t_step):] = np.cumsum(disc[(N+1-t_step+1)*(N+1==t_step):])*c
    discount_sum_repeated = np.tile(disc, (num_paths, 1))
    
    payoff[:,(N+2-t_step)*(N+1==t_step):-1] = discount_sum_repeated[:,:-1] + discount_repeated[:,:-1] * F 

    # final column: depends on KO flag
    h_val = _coupon_or_floor(paths, K, F)
    ko    = paths[:, -1, -1]   
    payoff[:,-1] = discount_sum_repeated[:,-1] + discount_repeated[:,-1] * F * (1 - ko) + discount_repeated[:,-1] * h_val[:,-1] *ko

    return torch.as_tensor(payoff, dtype=torch.float32)


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
    Return (batch_paths, current_state) for the given epoch/time-step.

    • paths         shape (num_epochs, batch, d+1, N+1)
    • current_state shape (batch, d)   — excludes KO slice
    """
    batch_paths = paths[epoch]
    state       = torch.as_tensor(
        batch_paths[:, :-1, n],          
        dtype=torch.float32,
        device=device,
    )
    return batch_paths, state


# simple state extractor used during evaluation
get_state_fn = lambda paths, i: paths[:, :-1, i]    
