"""Evaluate the domain-randomised (DR) and SUMO fine-tuned (DR-FT) policies in the NumPy ring.

Usage:  python evaluate_dr.py
Expects runs/{PPO-DR,CertiFlow-DR,CertiFlow-DR-FT}_s{0,1,2}/policy.pkl
Outputs results/episodes_dr_numpy.csv   scenarios S1-S3
        results/episodes_s4_ft.csv      penetration sweep (1-4 CAVs) of the final policy
        results/certificates_dr.json    spectral certificates (2 CAVs; 1, 3, 4 CAVs for the final policy)
        results/traces_S3.npz           adds the S3 trajectory of the final policy (run evaluate.py first)
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import pandas as pd

from evaluate import run, load_agent, SCENARIOS
from certiflow.env import EnvConfig
from certiflow.shield import calibrate_scores
from certiflow.methods import make_cfg, make_shield
from certiflow.certify import spectral_certificate

# policy weights -> method whose configuration and shield are used at test time
DR_METHODS = {"PPO-DR": "PPO", "CertiFlow-DR": "CertiFlow", "CertiFlow-DR-FT": "CertiFlow"}
FINAL = "CertiFlow-DR-FT"


def to_json(d):
    return {k: (np.abs(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_envs", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    scores = calibrate_scores(EnvConfig())

    rows = []
    for scen, over in SCENARIOS.items():
        for si, seed in enumerate(args.seeds):
            base = EnvConfig(n_cav=2, **over)
            for w, m in DR_METHODS.items():
                cfg = make_cfg(m, base)
                df, _ = run(cfg, agent=load_agent(w, seed, args.runs, required=True),
                            shield=make_shield(m, cfg, scores), n_envs=args.n_envs, seed=1000 + 100 * si)
                df["method"], df["scenario"], df["seed"] = w, scen, seed
                rows.append(df)
        print("done", scen, flush=True)
    pd.concat(rows, ignore_index=True).to_csv(os.path.join(args.out, "episodes_dr_numpy.csv"), index=False)

    rows = []
    for n_cav in [1, 2, 3, 4]:
        for si, seed in enumerate(args.seeds):
            cfg = make_cfg("CertiFlow", EnvConfig(n_cav=n_cav))
            df, _ = run(cfg, agent=load_agent(FINAL, seed, args.runs, required=True),
                        shield=make_shield("CertiFlow", cfg, scores), n_envs=args.n_envs, seed=5000 + 100 * si)
            df["method"], df["scenario"], df["seed"], df["n_cav"] = FINAL, "S4-penetration", seed, n_cav
            rows.append(df)
        print("done penetration", n_cav, flush=True)
    pd.concat(rows, ignore_index=True).to_csv(os.path.join(args.out, "episodes_s4_ft.csv"), index=False)

    certs = {}
    for w, m in DR_METHODS.items():
        for seed in args.seeds:
            cfg = make_cfg(m, EnvConfig(n_cav=2))
            r = spectral_certificate(cfg, load_agent(w, seed, args.runs, required=True), make_shield(m, cfg, scores))
            certs[f"{w} s{seed}"] = to_json(r)
            print("certificate", w, seed, f"rho={r['rho']:.4f}", flush=True)
    for n_cav in [1, 3, 4]:
        for seed in args.seeds:
            cfg = make_cfg("CertiFlow", EnvConfig(n_cav=n_cav))
            r = spectral_certificate(cfg, load_agent(FINAL, seed, args.runs, required=True),
                                     make_shield("CertiFlow", cfg, scores))
            certs[f"{FINAL} s{seed} K={n_cav}"] = to_json(r)
            print("certificate", FINAL, seed, f"K={n_cav}", f"rho={r['rho']:.4f}", flush=True)
    with open(os.path.join(args.out, "certificates_dr.json"), "w") as fh:
        json.dump(certs, fh, indent=1)

    path = os.path.join(args.out, "traces_S3.npz")
    traces = dict(np.load(path)) if os.path.exists(path) else {}
    cfg = make_cfg("CertiFlow", EnvConfig(n_cav=2, **SCENARIOS["S3-emergency"]))
    _, tr = run(cfg, agent=load_agent(FINAL, args.seeds[0], args.runs, required=True),
                shield=make_shield("CertiFlow", cfg, scores), n_envs=args.n_envs, seed=1000, keep_trace=True)
    traces.update({f"{FINAL}|{part}": arr for part, arr in zip("xva", tr)})
    np.savez_compressed(path, **traces)


if __name__ == "__main__":
    main()
