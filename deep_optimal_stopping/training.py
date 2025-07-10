# deep_optimal_stopping/training.py
"""
Neural-network training & evaluation utilities for Deep-Optimal-Stopping.

* `train_stopping_rule` – fits one `StoppingRuleNet` at time-step *n*
* `evaluate_stopping_rule` – rolls the fitted nets forward to get optimal
  stopping indices and corresponding payoffs

"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .models import StoppingRuleNet

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_stopping_rule(
    *,
    r: float,
    N: int,
    T: float,
    d: int,
    n: int,
    l: np.ndarray,                        
    get_batch_fn: Callable,
    g_fn: Callable,
    models: nn.ModuleDict,
    num_epochs: int,
    paths: np.ndarray | None = None,
    batch_size: int | None = None,
    maximize: bool = True,
    lr: float = 1e-4,
) -> np.ndarray:
    """
    Train a `StoppingRuleNet` for one time step *n* and
    update the current best stopping indices `l`.
    """
    device = _device()
    net = StoppingRuleNet(d, d + 40, d + 40).to(device)
    opt = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        # fetch batch --------------------------------------------------------
        paths_epoch, state = get_batch_fn(paths, epoch, n)
        state = torch.as_tensor(state, dtype=torch.float32, device=device)

        # continuation vs. immediate reward ----------------------------------
        if n == N - 1:
            continuation = g_fn(paths_epoch)[..., n + 1].to(device)
        else:
            l_epoch_t = torch.as_tensor(l[epoch], device=device, dtype=torch.long)
            continuation = g_fn(paths_epoch).to(device)[range(batch_size), l_epoch_t]

        stop_prob = net(state).squeeze(-1)
        mask = (stop_prob > 0.5).cpu().numpy()
        l[epoch, mask] = n

        imm_reward = g_fn(paths_epoch)[..., n].to(device)
        exp_reward = stop_prob * imm_reward + (1.0 - stop_prob) * continuation

        # optimize -----------------------------------------------------------
        loss = -exp_reward.mean() if maximize else exp_reward.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    models[str(n)] = net
    return l


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate_stopping_rule(
    *,
    N: int,
    paths: np.ndarray,
    g_fn: Callable,
    get_state_fn: Callable,
    models: Dict[str, nn.Module],
    t: int = 0,
    maximize: bool = True,
) -> Tuple[Tensor, np.ndarray]:
    """
    Evaluate a collection of trained nets and return payoffs and optimal indices.
    """
    device = _device()
    num_paths, _, _ = paths.shape

    time_idx = np.arange(N, t - 1, -1)
    slice_idx = np.arange(len(time_idx))[::-1]
    optimal_stopping_time = np.full(num_paths, N, dtype=int)              

    payoffs_all = g_fn(paths).to(device)

    for n, i in zip(time_idx, slice_idx):
        state_np = get_state_fn(paths, i)

        if n == N:
            continue

        # example 2 ---------------------------------------------------------------------------------
        if not maximize and n == 0:
            continue

        if n == 0:
            # remark 6 ------------------------------------------------------------------------------
            stopping_prob = payoffs_all[:, n] >= payoffs_all[range(num_paths), optimal_stopping_time]
            optimal_stopping_time[stopping_prob > 0.5] = 0
        
        else:
            state = torch.as_tensor(state_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                mask = models[str(n)](state).squeeze(-1) > 0.5
            optimal_stopping_time[mask.cpu().numpy()] = i

    payoffs = payoffs_all[range(num_paths), optimal_stopping_time]
    return payoffs, optimal_stopping_time
