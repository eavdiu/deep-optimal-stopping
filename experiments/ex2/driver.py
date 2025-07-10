# deep_optimal_stopping/experiments/ex2/driver.py
"""
Reusable driver that performs:

    simulate → train → lower bound → upper bound → confidence interval
"""

from __future__ import annotations
import gc, time, numpy as np, torch, torch.nn as nn

from deep_optimal_stopping.simulate      import simulate_paths_bs_barrier
from deep_optimal_stopping.training      import train_stopping_rule, evaluate_stopping_rule
from deep_optimal_stopping.bounds        import upper_bound_standard, confidence_interval_summary


def run_simulation_ex2(
    d: int,
    rho: float,
    cfg: dict,
) -> list[float]:
    """Return [L, U, point, CI_low, CI_high, t_L, t_U]."""

    # ---------------------------------------------------------- 1. TRAIN ---
    t0 = time.time()
    training_paths = simulate_paths_bs_barrier(
        S0=cfg["S0"],
        r=cfg["r"],
        delta=cfg["delta"],
        sigma=cfg["sigma"],
        corr=cfg["corr"],
        barrier_level = cfg["B"],
        T=cfg["T"],
        T_i = cfg["T_i"],
        N=cfg["N"],
        num_paths=cfg["K_L"],
        seed=cfg["seed_train"],
    ).reshape(cfg["num_epochs"], cfg["batch_size"], d + 1, cfg["N"] + 1)

    models = nn.ModuleDict()
    l = (cfg["N"] * np.ones(cfg["K_L"], dtype="int32")
                 .reshape(cfg["num_epochs"], cfg["batch_size"]))

    for n in range(cfg["N"] - 1, 0, -1):
        l = train_stopping_rule(
            r=cfg["r"], N=cfg["N"], T=cfg["T"], d=d, n=n, l=l,
            get_batch_fn=cfg["get_batch_fn"],
            g_fn=cfg["g_fn"],
            models=models,
            paths=training_paths,
            batch_size=cfg["batch_size"],
            num_epochs=cfg["num_epochs"],
            maximize=cfg["maximize"],
        )

    del training_paths; torch.cuda.empty_cache(); gc.collect()

    # ---------------------------------------------------- 2. LOWER BOUND ---
    test_paths = simulate_paths_bs_barrier(
        S0=cfg["S0"],
        r=cfg["r"],
        delta=cfg["delta"],
        sigma=cfg["sigma"],
        corr=cfg["corr"],
        barrier_level = cfg["B"],
        T=cfg["T"],
        T_i = cfg["T_i"],
        N=cfg["N"],
        num_paths=cfg["K_L_test"],
        seed=cfg["seed_L"],
    )
    payL, _ = evaluate_stopping_rule(
        N=cfg["N"], paths=test_paths,
        g_fn=cfg["g_fn"],
        get_state_fn=cfg["get_state_fn"],
        models=models, maximize=cfg["maximize"],
    )
    L, sigma_L = payL.mean().item(), payL.std(unbiased=True).item()
    t_L = time.time() - t0
    del test_paths, payL; torch.cuda.empty_cache(); gc.collect()

    # ---------------------------------------------------- 3. UPPER BOUND ---
    t1 = time.time()
    z_paths = simulate_paths_bs_barrier(
        S0=cfg["S0"],
        r=cfg["r"],
        delta=cfg["delta"],
        sigma=cfg["sigma"],
        corr=cfg["corr"],
        barrier_level = cfg["B"],
        T=cfg["T"],
        T_i = cfg["T_i"],
        N=cfg["N"],
        num_paths=cfg["K_U"],
        seed=cfg["seed_U"],
    )
    U, sigma_U = upper_bound_standard(
        z=z_paths, K_U=cfg["K_U"], L=L, N=cfg["N"],
        models=models,
        g_fn=cfg["g_fn"],
        get_state_fn=cfg["get_state_fn"],
        simulate_fn=cfg["simulate_fn"],
        seed=cfg["seed_U"], maximize=cfg["maximize"],
    )
    t_U = time.time() - t1
    del z_paths; torch.cuda.empty_cache(); gc.collect()

    # ---------------------------------------------- 4. CONFIDENCE INTERVAL --
    ci = confidence_interval_summary(
        L=U, U=L, sigma_L=sigma_U, sigma_U=sigma_L,
        K_L=cfg["K_U"], K_U=cfg["K_L_test"], alpha=cfg["alpha"],
    )
    return [L, U, ci["point_estimate"], *ci["confidence_interval"], t_L, t_U]
