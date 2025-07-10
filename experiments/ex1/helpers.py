#experiments/ex1/helpers.py
"""
Utility functions for Example 1.
"""
from __future__ import annotations
import numpy as np
import torch


# --------------------------------------------------------------------------- #
# 1.  Payoff g                                                                #
# --------------------------------------------------------------------------- #
def payoff_g(
    x: np.ndarray,
    T: float,
    N: int,
    K: float,
    r: float,
) -> torch.Tensor:
    """
    Discounted payoff of a Bermudan call on the **max** component.

    Returns
    -------
    torch.FloatTensor  (paths,)
    """

    n_steps  = x.shape[-1]                       
    t_idx    = np.arange(N + 1 - n_steps, N + 1)
    discount = np.exp(-r * t_idx * (T / N))
    payoff   = np.maximum(0.0, x.max(axis=1) - K) * discount
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
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Return (batch_paths, current_state) for the given epoch/time-step.
    """
    batch_paths = paths[epoch]                              
    state       = torch.as_tensor(batch_paths[..., n],
                                  dtype=torch.float32,
                                  device=device)
    return batch_paths, state


# simple state extractor used during evaluation
get_state_fn = lambda paths, i: paths[..., i]               
