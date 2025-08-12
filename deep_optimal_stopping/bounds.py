# deep_optimal_stopping/bounds.py
"""
Dual-estimator utilities for all three examples.

* Example 1 & 2 – standard Brownian/GBM noise
* Example 3     – fractional Brownian motion

"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import torch
from scipy.stats import norm
import gc

from .training import evaluate_stopping_rule


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# upper bounds
# --------------------------------------------------------------------------- #
# example 1 & 2
def upper_bound_standard(
    *,
    z: np.ndarray,
    K_U: int,
    L: float,
    N: int,
    models: Dict[str, torch.nn.Module],
    g_fn: Callable,
    get_state_fn: Callable,
    simulate_fn: Callable,
    seed: int | None = None,
    maximize: bool = True,
) -> Tuple[float, float]:
    """
    Dual upper bound for Brownian / GBM examples (standard noise).

    Returns
    -------
    float
        Upper-bound point estimate.
    float
        Sample standard deviation of the dual estimator.
    """
    device = _device()
    num_paths = z.shape[0]

    payoff = g_fn(z).to(device)
    cont = torch.zeros(num_paths, N + 1, dtype=torch.float32, device=device)
    f_theta = torch.zeros_like(cont)
    f_theta[:, -1] = 1.0


    for n in range(N - 1, -1, -1):
        current_path = get_state_fn(z, n)
        d = current_path.shape[1]
        
        if n == 0:       
            # remark 6 -------------------------------------------------------------------------------
            cont[:, 0] = L
            f_theta[:, 0] = (payoff[0, 0] >= L).float()
        else:
            # nested simulation for continuation values -----------------------------------------------
            for i in range(K_U):
                seed_i = None if seed is None else seed + n * K_U + i
                barrier = int(z[i, -1, n]) if z.shape[1] > d else 0

                z_tilde = simulate_fn(current_path[i], n,
                                      seed=seed_i, barrier_breached=barrier)
                payoffs, _ = evaluate_stopping_rule(
                    N=N - n,
                    paths=z_tilde,
                    g_fn=g_fn,
                    get_state_fn=get_state_fn,
                    models=models,
                    t=n + 1,
                )
                cont[i, n] = payoffs.mean()

            with torch.no_grad():
                current_path_tensor = torch.as_tensor(current_path, dtype=torch.float32, device=device)
                stop_prob = models[str(n)](current_path_tensor).squeeze(-1)
                f_theta[:, n] = (stop_prob >= 0.5).float()

            z_tilde = None
            gc.collect()

    # build martingale M ------------------------------------------------------------------------------
    dM = payoff[:, 1:] * f_theta[:, 1:] + (1.0 - f_theta[:, 1:]) * cont[:, 1:] - cont[:, :-1]
    M = torch.zeros_like(payoff)
    M[:, 1:] = np.cumsum(dM, axis=1)
    M = torch.as_tensor(M, dtype=torch.float32, device=device)

    if maximize:                                              # example 1 
        dual = (payoff - M).max(dim=1).values
    else:                                                     # example 2 (minimization)
        dual = -(-payoff[:, 1:] + M[:, 1:]).max(dim=1).values

    return dual.mean().item(), dual.std(unbiased=True).item()

# example 3
def upper_bound_fbm(
    *,
    z: np.ndarray,
    v: np.ndarray,
    K_U: int,
    L: float,
    N: int,
    B: float,
    H: float,
    T: float,
    J: int,
    models: Dict[str, torch.nn.Module],
    g_fn: Callable,
    get_state_fn: Callable,
    simulate_v: Callable,
    compute_w_tilde: Callable,
    compute_z_tilde: Callable,
    seed: int | None = None,
) -> Tuple[float, float]:
    """
    Dual upper bound for the fractional-Brownian-motion example (paper §4.3).
    """
    device = _device()
    num_paths = z.shape[0]

    payoff = g_fn(z).to(device)
    cont = torch.zeros(num_paths, N + 1, dtype=torch.float32, device=device)
    f_theta = torch.zeros_like(cont)
    f_theta[:, -1] = 1.0


    for n in range(N - 1, -1, -1):
        if n == 0:
            # remark 6 -------------------------------------------------------------------------------
            cont[:, 0] = L
            f_theta[:, 0] = (payoff[0, 0] >= L).float()
            continue

        # nested simulation for continuation values -----------------------------------------------
        for i in range(K_U):
            seed_i = None if seed is None else seed + n * K_U + i
            *_dummy, v_tilde = simulate_v(H, N, T, J, return_internals = True, seed=seed_i)
            w_tilde = compute_w_tilde(v, v_tilde, B, n, i)
            z_tilde_i = compute_z_tilde(z, w_tilde, n, i)

            payoffs, _ = evaluate_stopping_rule(
                N=N - n,
                paths=z_tilde_i,
                g_fn=g_fn,
                get_state_fn=get_state_fn,
                models=models,
                t=n + 1,
            )
            cont[i, n] = payoffs.mean()
            del z_tilde_i, w_tilde, v_tilde

        current_path = z[:, n, :]
        current_path_tensor = torch.as_tensor(current_path, dtype=torch.float32, device=device)
        with torch.no_grad():
            stop_prob = models[str(n)](current_path_tensor).squeeze(-1)
            f_theta[:, n] = (stop_prob >= 0.5).float()
            
        del current_path_tensor, stop_prob
        torch.cuda.empty_cache()
        gc.collect()
        
    # build martingale M ------------------------------------------------------------------------------
    dM = payoff[:, 1:] * f_theta[:, 1:] + (1.0 - f_theta[:, 1:]) * cont[:, 1:] - cont[:, :-1]
    M = torch.zeros_like(payoff)
    M[:, 1:] = np.cumsum(dM, axis=1)
    M = torch.as_tensor(M, dtype=torch.float32, device=device)

    dual = (payoff - M).max(dim=1).values
    return dual.mean().item(), dual.std(unbiased=True).item()
    
    
    

# --------------------------------------------------------------------------- #
# confidence interval & point estimate
# --------------------------------------------------------------------------- #
def confidence_interval_summary(
    L: float,
    U: float,
    sigma_L: float,
    sigma_U: float,
    K_L: int,
    K_U: int,
    alpha: float = 0.05,
) -> dict:
    """
    Two-sided (1-alpha) confidence interval for the option price.
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    ci_lower = L - z_alpha * sigma_L / np.sqrt(K_L)
    ci_upper = U + z_alpha * sigma_U / np.sqrt(K_U)
    return {
        "L": L,
        "U": U,
        "point_estimate": (L + U) / 2,
        "confidence_interval": [ci_lower, ci_upper],
    }



