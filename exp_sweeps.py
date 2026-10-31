"""Robustness and sensitivity experiments for the final CertiFlow policy (CertiFlow-DR-FT, 3 seeds x 20 rings).

1. Human-uncertainty sweep: human acceleration noise sigma = 0.3*k and parameter spread = 0.1*k,
   k in {0, 0.5, 1 (nominal), 1.5, 2, 2.5, 3}.  Also unshielded PPO for the collision panel.
2. Shield-parameter sensitivity under distribution shift (S2): gamma, alpha, d0, tau varied one at a time
   around the nominal values (0.3, 0.01, 2 m, 0.3 s) with the trained policy fixed.

Usage:  python exp_sweeps.py [uncertainty] [sensitivity]
Writes results/sweep_uncertainty.csv and results/sweep_sensitivity.csv (one row per episode).
"""
from __future__ import annotations

import sys
import pandas as pd

from evaluate import run, load_agent
from certiflow.env import EnvConfig
from certiflow.shield import ConformalCBFShield, calibrate_scores
from certiflow.methods import make_cfg

SEEDS = [0, 1, 2]
POLICY = "CertiFlow-DR-FT"


def shield_for(cfg, scores, **kw):
    lo, up = scores
    return ConformalCBFShield(cfg, scores_low=lo, scores_up=up, **kw)


def uncertainty_sweep(scores):
    rows = []
    for k in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for si, seed in enumerate(SEEDS):
            base = EnvConfig(n_cav=2, accel_noise=0.3 * k, idm_spread=0.1 * k)
            cfg = make_cfg("CertiFlow", base)
            df, _ = run(cfg, agent=load_agent(POLICY, seed, required=True), shield=shield_for(cfg, scores),
                        n_envs=20, seed=7000 + 100 * si)
            df["method"], df["k"], df["seed"] = "CertiFlow", k, seed; rows.append(df)
            cfgp = make_cfg("PPO", base)
            df, _ = run(cfgp, agent=load_agent("PPO", seed, required=True), shield=None, n_envs=20, seed=7000 + 100 * si)
            df["method"], df["k"], df["seed"] = "PPO", k, seed; rows.append(df)
        print("uncertainty k", k, flush=True)
    out = pd.concat(rows, ignore_index=True); out.to_csv("results/sweep_uncertainty.csv", index=False)


def sensitivity_sweep(scores):
    grid = {"gamma": [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
            "alpha": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
            "d0": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
            "tau": [0.0, 0.15, 0.3, 0.5, 0.8, 1.0]}
    rows = []
    for name, values in grid.items():
        for val in values:
            for si, seed in enumerate(SEEDS):
                base = EnvConfig(n_cav=2, idm_spread=0.30, accel_noise=0.6)      # S2: distribution shift
                cfg = make_cfg("CertiFlow", base)
                sh = shield_for(cfg, scores, **{name: val})
                df, _ = run(cfg, agent=load_agent(POLICY, seed, required=True), shield=sh, n_envs=20,
                            seed=8000 + 100 * si)
                df["param"], df["value"], df["seed"] = name, val, seed; rows.append(df)
            print("sensitivity", name, val, flush=True)
    out = pd.concat(rows, ignore_index=True); out.to_csv("results/sweep_sensitivity.csv", index=False)


if __name__ == "__main__":
    scores = calibrate_scores(EnvConfig())
    which = sys.argv[1:] or ["uncertainty", "sensitivity"]
    if "uncertainty" in which:
        uncertainty_sweep(scores)
    if "sensitivity" in which:
        sensitivity_sweep(scores)
