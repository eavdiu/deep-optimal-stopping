#deep_optimal_stopping/experiments/ex2/sweep.py
"""
Grid sweep over (d, rho) pairs for Example 2.
Writes results to results/ex2_results.csv and prints a nice table.
"""

from __future__ import annotations
import csv, os, gc, torch, numpy as np
from pathlib import Path
import argparse

from deep_optimal_stopping.utils import set_global_seed
set_global_seed(2025) 

from helpers import (
    payoff_g, get_batch, get_state_fn,
)
from driver  import run_simulation_ex2
from deep_optimal_stopping.simulate import simulate_paths_bs_barrier


# ------------------------------------------------------------------ #
# sweep parameters                                                   #
# ------------------------------------------------------------------ #
Ds         = [2, 3, 5, 10, 15, 30]
rho_values  = [0.6, 0.1]
base_seed  = 2025

# output file
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
csv_path = results_dir / "ex2_results.csv"


# ------------------------------------------------------------------ #
# helper to build a config dict                                      #
# ------------------------------------------------------------------ #
def build_config(d: int, seed_offset: int) -> dict:
    """Return a fresh config dict for dimension d."""
    # ------------------------------- common params --------------------------
    cfg = dict(
        r      = 0.0,
        T      = 1.0,
        T_i      = 1/2,
        K      = 100.0,
        F      = 100.0,
        B      = 70.0,
        N      = 12,
        c      = 7/12,
        maximize = False,
        alpha    = 0.05,
        sigma    = np.full(d, 0.2),
        delta    = np.full(d, 0.05),
        S0    = np.full(d, 100),

        # placeholder for functions
        get_batch_fn = None,
        get_state_fn = get_state_fn,
        g_fn         = None,
        simulate_fn  = None,
    )

    # -------------------------- size / budget by dimension ------------------
    if d <= 5:
        cfg.update(
            num_epochs = 3000 + d,
            batch_size = 4096,
            J          = 1024,
            K_L_test   = 2**22,
        )
    elif d <= 20:
        cfg.update(
            num_epochs = 1500 + d,
            batch_size = 4096,
            J          = 1024,
            K_L_test   = 2**20,
        )
    elif d <= 50:                               
        cfg.update(
            num_epochs = 1500 + d,                
            batch_size = 2048,
            J          = 1024,
            K_L_test   = 2**20,
        )
    else:                                        # fallback for even larger d
        cfg.update(
            num_epochs = 250 + d,
            batch_size = 1024,
            J          = 1024,
            K_L_test   = 2**19,
        )

    # derived Monte-Carlo sizes
    cfg["K_L"] = cfg["batch_size"] * cfg["num_epochs"]
    cfg["K_U"] = 1024

    # ------------------------------- seeds ----------------------------------
    cfg["seed_train"] = base_seed + seed_offset
    cfg["seed_L"]     = base_seed + seed_offset + 1
    cfg["seed_U"]     = base_seed + seed_offset + 2
            
    return cfg


def _parse_args() -> tuple[list[int], list[int]]:
    p = argparse.ArgumentParser("Example-2 sweep")
    p.add_argument("--d",  type=int, nargs="+", help="dimensions to run")
    p.add_argument("--rho", type=int, nargs="+", help="initial spots to run")
    a = p.parse_args()
    dims = a.d  if a.d  else Ds         # Ds = default list from the file
    rhos  = a.rho if a.rho else rho_values  # rho_values = default list
    return dims, rhos

# ------------------------------------------------------------------- #
# main sweep loop                                                     #
# ------------------------------------------------------------------- #
def main() -> None:
    header = ["d","rho","L","t_L","U","t_U","Point","CI_Low","CI_High"]
    print(" | ".join(header))
    print("-"*85)

    mode = "a" if csv_path.exists() else "w"
    with csv_path.open(mode, newline="") as f_csv:
        writer = csv.writer(f_csv)
        if mode == "w":        
            writer.writerow(header)

        dims, spots = _parse_args()
        for d in dims:
            for j, rho in enumerate(spots):
                cfg = build_config(d, seed_offset=10*d+int(1000*rho))

                # bind functions that depend on cfg -------------------
                corr = (1 - rho) * np.eye(d) + rho * np.ones((d, d))
                cfg["corr"] = corr
                cfg["g_fn"] = lambda x, c=cfg: payoff_g(
                    x,
                    T=c["T"],
                    N=c["N"],
                    K=c["K"],
                    F=c["F"],
                    c=c["c"],
                    r=c["r"],
                )
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                cfg["get_batch_fn"] = lambda p,e,n, dev=dev: get_batch(p, e, n, device=dev)
                cfg["simulate_fn"] = lambda x_i, n, seed, barrier_breached=0: simulate_paths_bs_barrier(
                    S0=x_i,
                    r=cfg["r"],
                    delta=cfg["delta"],
                    sigma=cfg["sigma"],
                    corr=cfg["corr"],
                    barrier_level=cfg["B"],
                    T=cfg["T"],
                    T_i=cfg["T_i"],
                    N=cfg["N"],
                    num_steps=cfg["N"] - n,
                    num_paths=cfg["J"],
                    seed=seed,
                    barrier_breached=barrier_breached
                )
                # run --------------------------------------------------
                U,L,pt,lo,hi,tU,tL = run_simulation_ex2(d, rho, cfg)
                print(f"{d:2} | {rho:3} | {L:7.3f} | {tL:5.1f}s | "
                      f"{U:7.3f} | {tU:5.1f}s | {pt:7.3f} | {lo:7.3f} | {hi:7.3f}")

                writer.writerow([d, rho, L, tL, U, tU, pt, lo, hi])
                torch.cuda.empty_cache(); gc.collect()

    print(f"\nSaved CSV → {csv_path}")



if __name__ == "__main__":
    main()
