# deep_optimal_stopping/simulate/bs.py
"""
Simulation helpers for Example 1 & 2.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Example 1: standard multi-asset GBM                                         #
# --------------------------------------------------------------------------- #
def simulate_paths_bs(
    *,
    S0: np.ndarray,
    r: float,
    delta: float,
    sigma: np.ndarray,
    corr: np.ndarray,
    T: float,
    N: int,
    num_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate correlated geometric Brownian-motion (GBM) paths.

    Parameters
    ----------
    S0 : (d,) array
        Initial spot prices.
    r, delta : float
        Risk-free rate and dividend yield.
    sigma : (d,) array
        Volatility for each asset.
    corr : (d, d) array
        Correlation matrix between Brownian drivers.
    T : float
        Maturity horizon.
    N : int
        Number of time steps (grid is ``0 … N``).
    num_paths : int
        Monte-Carlo sample size.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    paths : ndarray, shape (num_paths, d, N+1)
    """
    rng = np.random.default_rng(seed)
    d = len(S0)

    # Brownian increments -------------------------------------------------------
    Z = rng.standard_normal(size=(num_paths, d, N))      # (paths, d, N)
    Z = np.insert(Z, 0, 0.0, axis=2)                     

    L = np.linalg.cholesky(corr)                         # correlate across assets
    W = np.cumsum(L @ Z, axis=2) * np.sqrt(T / N)        # Brownian paths

    paths = np.empty((num_paths, d, N + 1), dtype="float64")
    paths[..., 0] = S0
    t_grid = np.linspace(0, T, N + 1)

    drift = (r - delta - 0.5 * sigma**2)
    for i in range(1, N + 1):
        exp_term = drift * t_grid[i] + (W[..., i] * sigma)
        paths[..., i] = paths[..., 0] * np.exp(exp_term)

    return paths

# -----------------------------------------------------------------------------
# Example 2: GBM with discrete dividend + knock-out barrier
# -----------------------------------------------------------------------------
def simulate_paths_bs_barrier(
    *,
    S0: np.ndarray,                 # (d,) initial prices
    r: float,
    delta: float,
    sigma: np.ndarray,              # (d,)
    corr: np.ndarray,               # (d, d)
    barrier_level: float,           # B
    T: float,                       # maturity of full contract
    T_i: float,                     # dividend time (absolute)
    N: int,                         # total grid points 0…N
    num_steps: int | None = None,   # steps to simulate (default=N)
    num_paths: int,
    seed: int | None = None,
    barrier_breached: int = 0,
) -> np.ndarray:
    """
    Correlated GBM with dividend & KO flag.
    If `num_steps < N`, we simulate only the *future* segment t_n … T.

    Returns
    -------
    paths : (num_paths, d+1, num_steps+1)
            last slice is KO indicator.
    """
    if num_steps is None:
        num_steps = N                       # full path
    assert 1 <= num_steps <= N, "`num_steps` must lie in [1, N]"

    rng = np.random.default_rng(seed)
    d   = len(S0)
    dt  = T / N                             

    start_idx = N - num_steps               # e.g. n = current time index
    t_grid    = np.linspace(start_idx * dt, T, num_steps + 1)

    # Brownian increments just for the segment ------------------------------
    Z = rng.standard_normal(size=(num_paths, d, num_steps))
    Z = np.insert(Z, 0, 0.0, axis=2)        

    L = np.linalg.cholesky(corr)
    W = np.cumsum(L @ Z, axis=2) * np.sqrt(dt)   # (paths, d, num_steps+1)

    paths = np.empty((num_paths, d + 1, num_steps + 1), dtype="float32")
    paths[..., :-1, 0] = S0                    # past S_n fed in by caller

    # KO flag initial state ------------------------------------------------
    initial_breach = (S0 < barrier_level).any().astype("float32")
    paths[:, -1, 0] = initial_breach
    
    drift = r - 0.5 * sigma**2

    for k in range(1, num_steps + 1):
        adjust = 1.0 - delta if t_grid[k] >= T_i else 1.0

        exp_term = drift * (t_grid[k]) + W[..., k] * sigma
        paths[..., :-1, k] = S0 * adjust * np.exp(exp_term)

        below = (paths[..., :-1, k] < barrier_level).any(axis=1)
        paths[:, -1, k] = np.maximum(paths[:, -1, k - 1], below.astype("float32"))

    # optional override for continuation paths -----------------------------
    if barrier_breached == 1:
        paths[:, -1, :] = 1.0

    return paths

