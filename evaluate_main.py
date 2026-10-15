"""Evaluate every method on S1-S3 with the full metric set (incl. minimum TTC and CAV jerk), plus the
test-time ablations of the final policy (CertiFlow-DR-FT).  Evaluation seeds are those of evaluate.py.

Usage:  python evaluate_main.py
Output: results/episodes_main.csv  (used for Tables 6, 7, 10 and Fig. 8)
"""
from __future__ import annotations

import pandas as pd
from evaluate import run, load_agent, SCENARIOS
from certiflow.env import EnvConfig
from certiflow.shield import ConformalCBFShield, calibrate_scores
from certiflow.methods import make_cfg, make_shield
from certiflow.baselines import HumanIDM, PISaturation

SEEDS = [0, 1, 2]
# label -> (policy weights, config/shield method)
LEARNED = {
    "PPO": ("PPO", "PPO"), "PPO-DR": ("PPO-DR", "PPO"), "PPO+CBF": ("PPO+CBF", "PPO+CBF"),
    "PPO+CF-shield (post-hoc)": ("PPO", "CertiFlow"),
    "CertiFlow": ("CertiFlow", "CertiFlow"), "CertiFlow-noSTL": ("CertiFlow-noSTL", "CertiFlow-noSTL"),
    "CertiFlow-noCoop": ("CertiFlow-noCoop", "CertiFlow-noCoop"), "CertiFlow-noACI": ("CertiFlow-noACI", "CertiFlow-noACI"),
    "CertiFlow-DR": ("CertiFlow-DR", "CertiFlow"), "CertiFlow-DR-FT": ("CertiFlow-DR-FT", "CertiFlow"),
}
# test-time ablations of the final policy: shield kwargs (None = no shield)
TEST_ABL = {
    "Ours w/o shield": None,
    "Ours w/o conformal (nominal CBF)": dict(conformal=False, cooperative=False, adaptive=False),
    "Ours w/o cooperative term": dict(cooperative=False),
    "Ours w/o ACI": dict(adaptive=False),
}


def main():
    scores = calibrate_scores(EnvConfig())
    lo, up = scores
    rows = []
    for scen, over in SCENARIOS.items():
        for si, seed in enumerate(SEEDS):
            evs = 1000 + 100 * si
            base = EnvConfig(n_cav=2, **over)

            def add(df, m):
                df["method"], df["scenario"], df["seed"] = m, scen, seed; rows.append(df)
            add(run(make_cfg("none", base), controller=HumanIDM(), n_envs=20, seed=evs)[0], "Human (IDM)")
            add(run(make_cfg("none", base), controller=PISaturation(), n_envs=20, seed=evs)[0], "PI-Saturation")
            for lab, (w, m) in LEARNED.items():
                cfg = make_cfg(m, base)
                add(run(cfg, agent=load_agent(w, seed, required=True), shield=make_shield(m, cfg, scores),
                        n_envs=20, seed=evs)[0], lab)
            for lab, kw in TEST_ABL.items():
                cfg = make_cfg("CertiFlow", base)
                sh = None if kw is None else ConformalCBFShield(cfg, scores_low=lo, scores_up=up, **kw)
                add(run(cfg, agent=load_agent("CertiFlow-DR-FT", seed, required=True), shield=sh, n_envs=20,
                        seed=evs)[0], lab)
        print("done", scen, flush=True)
    pd.concat(rows, ignore_index=True).to_csv("results/episodes_main.csv", index=False)


if __name__ == "__main__":
    main()
