# deep_optimal_stopping/experiments/ex3/driver.py
"""
Reusable driver that performs:

    simulate → train → lower bound → upper bound → confidence interval
"""

from __future__ import annotations
import gc, time, numpy as np, torch, torch.nn as nn

from deep_optimal_stopping.simulate.fbm      import simulate_fbm, embed_state_matrix, compute_w_tilde_fbm, compute_w_mix, compute_z_tilde
from deep_optimal_stopping.training      import train_stopping_rule, evaluate_stopping_rule
from deep_optimal_stopping.bounds        import upper_bound_fbm, confidence_interval_summary

def run_simulation_ex3(
    H: float,
    cfg: dict,
) -> list[float]:
    """Return [L, U, point, CI_low, CI_high, t_L, t_U]."""

    # ---------------------------------------------------------- 1. TRAIN ---
    t0 = time.time()
    training_paths = np.zeros((cfg["num_epochs"] * cfg["batch_size"], cfg["N"] + 1, cfg["N"]))
    training_paths = training_paths.reshape(cfg["num_epochs"], cfg["batch_size"], cfg["N"] + 1, cfg["N"])

    # --- Simulate training paths
    for epoch in range(cfg["num_epochs"]):
        # Generate a new batch of paths with fractional Brownian motion for training
        seed =  cfg["seed_train"] + epoch
        batch_W = simulate_fbm(H=H, N=cfg["N"], T=cfg["T"], num_paths=cfg["batch_size"], seed=seed)

        # Construct the X matrix for each path in the batch
        batch_X = embed_state_matrix(batch_W)
        training_paths[epoch] = batch_X

        del batch_X, batch_W; torch.cuda.empty_cache(); gc.collect()
        
    models = nn.ModuleDict()
    l = (cfg["N"] * np.ones(cfg["K_L"], dtype="int32")
                 .reshape(cfg["num_epochs"], cfg["batch_size"]))

    for n in range(cfg["N"] - 1, -1, -1):
        l = train_stopping_rule(
            r=cfg["r"], N=cfg["N"], T=cfg["T"], d=cfg["d"], n=n, l=l,
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
    test_W = simulate_fbm(H=H, N=cfg["N"], T=cfg["T"], num_paths=cfg["K_L_test"],seed=cfg["seed_L"])
    test_paths = embed_state_matrix(test_W)
    
    payL, _ = evaluate_stopping_rule(
        N=cfg["N"], paths=test_paths,
        g_fn=cfg["g_fn"],
        get_state_fn=cfg["get_state_fn"],
        models=models, 
    )
    L, sigma_L = payL.mean().item(), payL.std(unbiased=True).item()
    t_L = time.time() - t0
    del test_W, test_paths, payL; torch.cuda.empty_cache(); gc.collect()

    # ---------------------------------------------------- 3. UPPER BOUND ---
    t1 = time.time()
    w,B,v = simulate_fbm(H=H, 
                         N=cfg["N"], 
                         T=cfg["T"], 
                         num_paths=cfg["K_U"], 
                         return_internals = True,
                         seed=cfg["seed_U"])
    z_paths = embed_state_matrix(w)

    U, sigma_U = upper_bound_fbm(
        z=z_paths, v=v, K_U=cfg["K_U"], L=L, N=cfg["N"], B=B, H=H,
        T=cfg["T"], J=cfg["J"], models=models,
        g_fn=cfg["g_fn"],
        get_state_fn=cfg["get_state_fn"],
        simulate_v=simulate_fbm,
        compute_w_tilde=compute_w_tilde_fbm,
        compute_z_tilde=compute_z_tilde,
        seed=cfg["seed_U"]
    )
    t_U = time.time() - t1
    del z_paths; torch.cuda.empty_cache(); gc.collect()

    # ---------------------------------------------- 4. CONFIDENCE INTERVAL --
    ci = confidence_interval_summary(
        L=L, U=U, sigma_L=sigma_L, sigma_U=sigma_U,
        K_L=cfg["K_L_test"], K_U=cfg["K_U"], alpha=cfg["alpha"],
    )
    return [L, U, ci["point_estimate"], *ci["confidence_interval"], t_L, t_U]
