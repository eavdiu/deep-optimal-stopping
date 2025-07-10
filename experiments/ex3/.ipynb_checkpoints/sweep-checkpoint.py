#deep_optimal_stopping/experiments/ex3/sweep.py
"""
Grid sweep over H for Example 3.
Writes results to results/ex3_results.csv and prints a nice table.
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
from driver  import run_simulation_ex3


# ------------------------------------------------------------------ #
# sweep parameters                                                   #
# ------------------------------------------------------------------ #
H         = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
base_seed  = 2025
d = 30

# output file
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
csv_path = results_dir / "ex3_results.csv"


# ------------------------------------------------------------------ #
# helper to build a config dict                                      #
# ------------------------------------------------------------------ #
def build_config(H: float, seed_offset: int) -> dict:
    """Return a fresh config dict for dimension d."""
    # ------------------------------- common params --------------------------
    cfg = dict(
        r      = 0.0,
        T      = 1.0,
        N      = d,
        d      = d,
        maximize = True,
        alpha    = 0.05,

        # placeholder for functions
        get_batch_fn = None,
        get_state_fn = get_state_fn,
        g_fn         = None,
        simulate_fn  = None,

        num_epochs = 1500 + d,
        batch_size = 512, 
        J = 8192, 
        K_L_test = 2**18
    )

    # derived Monte-Carlo sizes
    cfg["K_L"] = cfg["batch_size"] * cfg["num_epochs"]
    cfg["K_U"] = 1024

    # ------------------------------- seeds ----------------------------------
    cfg["seed_train"] = base_seed + seed_offset
    cfg["seed_L"]     = base_seed + seed_offset + 1
    cfg["seed_U"]     = base_seed + seed_offset + 2
            
    return cfg


def _parse_args() -> list[float]:
    p = argparse.ArgumentParser("Example-3 sweep")
    p.add_argument("--H",  type=float, nargs="+", help="H to run")
    a = p.parse_args()
    H = a.H  if a.H  else H         # H = default list from the file
    return H

# ------------------------------------------------------------------- #
# main sweep loop                                                     #
# ------------------------------------------------------------------- #
def main() -> None:
    header = ["H","L","t_L","U","t_U","Point","CI_Low","CI_High"]
    print(" | ".join(header))
    print("-"*85)

    mode = "a" if csv_path.exists() else "w"
    with csv_path.open(mode, newline="") as f_csv:
        writer = csv.writer(f_csv)
        if mode == "w":        
            writer.writerow(header)

        H_values = _parse_args()
        for j, H in enumerate(H_values):
            cfg = build_config(H, seed_offset=10*j+int(1000*H))

            # bind functions that depend on cfg -------------------
            cfg["g_fn"] = lambda x, c=cfg: payoff_g(x)
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cfg["get_batch_fn"] = lambda p,e,n, dev=dev: get_batch(p, e, n, device=dev)

            # run --------------------------------------------------
            L,U,pt,lo,hi,tL,tU = run_simulation_ex3(H, cfg)
            print(f"{H:3} | {L:7.3f} | {tL:5.1f}s | "f"{U:7.3f} | {tU:5.1f}s | {pt:7.3f} | {lo:7.3f} | {hi:7.3f}")

            writer.writerow([H, L, tL, U, tU, pt, lo, hi])
            torch.cuda.empty_cache(); gc.collect()

    print(f"\nSaved CSV → {csv_path}")



if __name__ == "__main__":
    main()
