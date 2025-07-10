# deep_optimal_stopping/simulate/fbm.py
"""
Simulation helpers for Example 3 (fractional-Brownian-motion).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# fBm simulator                                                               #
# --------------------------------------------------------------------------- #
def simulate_fbm(
    H: float,
    N: int,
    T: float,
    num_paths: int,
    *,
    return_internals: bool = False,
    seed: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate fractional Brownian motion (fBm) paths using Cholesky decomposition.

    Parameters
    ----------
    H : float
        Hurst parameter (0 < H <= 1).
    N : int
        Number of time steps.
    T : float
        Time horizon.
    num_paths : int
        Number of sample paths.
    return_internals : bool, optional
        Whether to return (W, L, Z) instead of just W. Default is False.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    W : np.ndarray, shape (num_paths, N+1)
        Simulated fBm paths.

    If return_internals is True, also returns:
    L : np.ndarray, shape (N+1, N+1)
        Cholesky factor of the covariance matrix.
    Z : np.ndarray, shape (num_paths, N+1)
        Underlying standard normal samples.
    """
    rng = np.random.default_rng(seed)

    # Time grid normalized by T
    t_grid = np.linspace(0.0, T, N + 1) / T  # Rescaled for covariance definition

    # Construct the covariance matrix based on page 18 of the paper
    delta = np.abs(t_grid[:, None] - t_grid[None, :])
    cov_matrix = 0.5 * (
        t_grid[:, None] ** (2 * H) +
        t_grid[None, :] ** (2 * H) -
        delta ** (2 * H)
    )
    cov_matrix += 1e-10 * np.eye(N + 1)  # Add small diagonal jitter for numerical stability

    # Cholesky decomposition
    L = np.linalg.cholesky(cov_matrix)

    Z = rng.standard_normal(size=(num_paths, N))     # no W_0 yet
    Z = np.insert(Z, 0, 0.0, axis=1)                 # prepend W_0 = 0
    W = (L @ Z.T).T                                  # shape (paths, N+1)

    # normalise so Var(W_T) ≈ 1
    W *= 1.0 / np.sqrt(np.var(W[:, -1]))

    if return_internals:
        return W, L, Z
    return W


# --------------------------------------------------------------------------- #
# State embedding X_n from fBm path                                           #
# --------------------------------------------------------------------------- #
def embed_state_matrix(
    W: np.ndarray,
    drop_w0: bool = True,
) -> np.ndarray:
    """
    Build lower-triangular *state vectors* X_n = (W_n, W_{n-1}, …).

    Parameters
    ----------
    W : (paths, N+1) fBm (or any 1-d path)
    drop_w0 : remove the leading W_0 column (default True)

    Returns
    -------
    X : (paths, N+1, N) state matrix; row n contains reversed prefix of length n.
    """
    if drop_w0:
        W = W[:, 1:]                                 # remove W_0
    P, N = W.shape

    X = np.zeros((P, N + 1, N), dtype=W.dtype)
    for n in range(1, N + 1):                        # start at 1 so X_0 = 0
        X[:, n, :n] = W[:, :n][:, ::-1]              # reversed prefix

    return X


# --------------------------------------------------------------------------- #
# Dual-estimator helpers (page 20)                                            #
# --------------------------------------------------------------------------- #
def compute_w_tilde_fbm(
    v: np.ndarray,
    v_tilde: np.ndarray,
    B: np.ndarray,
    n: int,
    i: int,
) -> np.ndarray:
    """
    Compute w_tilde according to page 20 (first- & second-sum terms).

    Parameters
    ----------
    v : (J, N+1) original v-process
    v_tilde : (J, N+1) conditional v-process for path i
    B : (N, N) pre-computed B-matrix (paper page 20)
    n : current time index
    i : path index

    Returns
    -------
    np.ndarray
        w_tilde  shape (J, N+1)
    """
    # first sum  
    B1 = B[1:, 1 : n + 1]                   
    v1 = v[:, 1 : n + 1]
    first = np.dot(v1[i], B1.T)
    first = np.tile(first, (v_tilde.shape[0], 1)) # repeat across J rows

    # second sum 
    v2 = v_tilde[:, n + 1 :]
    B2 = B[1:, n + 1 :]
    v2_reshaped = v2.reshape(-1, v2.shape[-1])  # Flatten the first two axes
    second = np.dot(v2_reshaped, B2.T)

    return first + second

# --------------------------------------------------------------------------- #
# splice past-and-future Brownian pieces                                      #
# --------------------------------------------------------------------------- #
def compute_w_mix(
    z: np.ndarray,
    w_tilde: np.ndarray,
    n: int,
    i: int,
) -> np.ndarray:
    """
    Compute w_mix by combining z and w_tilde.

    Returns
    -------
    np.ndarray
        w_mix of shape (J, N+1) 
    """
    J = w_tilde.shape[0]
    past = np.tile(z[i, n, :], (J, 1))          # broadcast row n of path i
    w_mix = np.zeros_like(w_tilde)
    w_mix[..., :n]  += past[..., :n]              # copy past
    w_mix[..., n:]  += w_tilde[..., n:]           # copy future
    return w_mix


# --------------------------------------------------------------------------- #
# build continuation path  z̃                                                 #
# --------------------------------------------------------------------------- #
def compute_z_tilde(
    z: np.ndarray,
    w_tilde: np.ndarray,
    n: int,
    i: int,
) -> np.ndarray:
    """
    Construct z_tilde according to page 20.
    """
    # merge raw Brownian pieces
    w_mix = compute_w_mix(z, w_tilde, n, i)

    # embed into lower-triangular state representation
    z_tilde = embed_state_matrix(w_mix, drop_w0=False)[:,n:,:]

    return z_tilde
