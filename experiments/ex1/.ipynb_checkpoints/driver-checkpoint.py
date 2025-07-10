# deep_optimal_stopping/experiments/ex1/driver.py
"""
Reusable driver that performs:

    simulate → train → lower bound → upper bound → confidence interval
"""

from __future__ import annotations
import gc, time, numpy as np, torch, torch.nn as nn

from deep_optimal_stopping.simulate      import simulate_paths_bs
from deep_optimal_stopping.training      import train_stopping_rule, evaluate_stopping_rule
from deep_optimal_stopping.bounds        import upper_bound_standard, confidence_interval_summary


def run_simulation_ex1(
    d: int,
    S0_scalar: float,
    cfg: dict,
) -> list[float]:
    
    """Return [L, U, point, CI_low, CI_high, t_L, t_U]."""
    S0     = np.full(d, S0_scalar)

    # ---------------------------------------------------------- 1. TRAIN ---
    t0 = time.time()
    training_paths = simulate_paths_bs(
        S0=S0, r=cfg["r"], delta=cfg["delta"], sigma=cfg["sigma"],
        corr=cfg["corr"], T=cfg["T"], N=cfg["N"],
        num_paths=cfg["K_L"], seed=cfg["seed_train"],
    ).reshape(cfg["num_epochs"], cfg["batch_size"], d, cfg["N"] + 1)

    models = nn.ModuleDict()
    l = (cfg["N"] * np.ones(cfg["K_L"], dtype="int32")
                 .reshape(cfg["num_epochs"], cfg["batch_size"]))

    for n in range(cfg["N"] - 1, -1, -1):
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
    test_paths = simulate_paths_bs(
        S0=S0, r=cfg["r"], delta=cfg["delta"], sigma=cfg["sigma"],
        corr=cfg["corr"], T=cfg["T"], N=cfg["N"],
        num_paths=cfg["K_L_test"], seed=cfg["seed_L"],
    )
    payL, _ = evaluate_stopping_rule(
        N=cfg["N"], paths=test_paths,
        g_fn=cfg["g_fn"],
        get_state_fn=cfg["get_state_fn"],
        models=models,
    )
    L, sigma_L = payL.mean().item(), payL.std(unbiased=True).item()
    t_L = time.time() - t0
    del test_paths, payL; torch.cuda.empty_cache(); gc.collect()

    # ---------------------------------------------------- 3. UPPER BOUND ---
    t1 = time.time()
    z_paths = simulate_paths_bs(
        S0=S0, r=cfg["r"], delta=cfg["delta"], sigma=cfg["sigma"],
        corr=cfg["corr"], T=cfg["T"], N=cfg["N"],
        num_paths=cfg["K_U"], seed=cfg["seed_U"],
    )
    U, sigma_U = upper_bound_standard(
        z=z_paths, K_U=cfg["K_U"], L=L, N=cfg["N"],
        models=models,
        g_fn=cfg["g_fn"],
        get_state_fn=cfg["get_state_fn"],
        simulate_fn=cfg["simulate_fn"],
        seed=cfg["seed_U"],
    )
    t_U = time.time() - t1
    del z_paths; torch.cuda.empty_cache(); gc.collect()

    # ---------------------------------------------- 4. CONFIDENCE INTERVAL --
    ci = confidence_interval_summary(
        L=L, U=U, sigma_L=sigma_L, sigma_U=sigma_U,
        K_L=cfg["K_L_test"], K_U=cfg["K_U"], alpha=cfg["alpha"],
    )
    return [L, U, ci["point_estimate"], *ci["confidence_interval"], t_L, t_U]
