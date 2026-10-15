"""Quick check of a trained CertiFlow policy (about 30 s on one CPU core).

Runs the final policy on 20 rings in each test scenario, with and without the shield, and prints
collisions, wave amplitude, mean speed and shield intervention rate.

Usage:  python demo.py [--seed 0]
"""
from __future__ import annotations

import argparse

from evaluate import run, load_agent, SCENARIOS
from certiflow.env import EnvConfig
from certiflow.shield import calibrate_scores
from certiflow.methods import make_cfg, make_shield
from certiflow.certify import spectral_certificate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, choices=[0, 1, 2])
    args = ap.parse_args()
    agent = load_agent("CertiFlow-DR-FT", args.seed, required=True)
    scores = calibrate_scores(EnvConfig())

    print(f"{'scenario':14s} {'shield':7s} {'collisions':>10s} {'wave (m/s)':>11s} {'speed (m/s)':>12s} {'interv.':>8s}")
    for scen, over in SCENARIOS.items():
        cfg = make_cfg("CertiFlow", EnvConfig(n_cav=2, **over))
        for shielded in (True, False):
            sh = make_shield("CertiFlow", cfg, scores) if shielded else None
            df, _ = run(cfg, agent=agent, shield=sh, n_envs=20, seed=1000 + 100 * args.seed)
            ok = df[~df.collision]
            wave = f"{ok.std_v_ss.mean():.3f}" if len(ok) else "--"
            speed = f"{ok.mean_v.mean():.2f}" if len(ok) else "--"
            print(f"{scen:14s} {'on' if shielded else 'off':7s} {int(df.collision.sum()):>7d}/20 {wave:>11s} "
                  f"{speed:>12s} {df.interv_rate.mean():>8.3f}")

    cfg = make_cfg("CertiFlow", EnvConfig(n_cav=2))
    hu = spectral_certificate(EnvConfig(n_cav=0))
    cf = spectral_certificate(cfg, agent, make_shield("CertiFlow", cfg, scores))
    print(f"\nspectral radius, human-only ring : {hu['rho']:.4f}  (unstable)")
    print(f"spectral radius, CertiFlow       : {cf['rho']:.4f}  "
          f"({'certified stable' if cf['rho'] < 1 - 1e-3 and cf['shield_margin'] > 0 else 'not certified'}, "
          f"barrier margin at equilibrium {cf['shield_margin']:.2f} m)")


if __name__ == "__main__":
    main()
