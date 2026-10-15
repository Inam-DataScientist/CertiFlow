"""Evaluate all controllers on the test scenarios, verify STL specs, compute certificates.

Usage:  python evaluate.py            (expects runs/<method>_s<seed>/policy.pkl)
Outputs results/episodes.csv, results/summary.csv, results/certificates.json,
        results/traces_*.npz (for figures)
"""
from __future__ import annotations

import argparse, copy, json, os, pickle
import numpy as np
import pandas as pd

from certiflow.env import RingEnv, EnvConfig
from certiflow.ppo import PPOAgent
from certiflow.shield import calibrate_scores
from certiflow.methods import make_cfg, make_shield
from certiflow.baselines import HumanIDM, PISaturation
from certiflow.stl import traffic_specs, signals_from_recording
from certiflow.certify import spectral_certificate, clopper_pearson_upper

RL_METHODS = ["PPO", "PPO+CBF", "CertiFlow", "CertiFlow-noSTL", "CertiFlow-noCoop", "CertiFlow-noACI"]

SCENARIOS = {
    "S1-nominal": {},
    "S2-shift": {"idm_spread": 0.30, "accel_noise": 0.6},
    "S3-emergency": {"perturbations": [(20.0, 1, 6.0, 3.0), (55.0, 3, 6.0, 3.0)]},
}


def load_agent(method, seed, runs="runs", required=False):
    p = os.path.join(runs, f"{method}_s{seed}", "policy.pkl")
    if not os.path.exists(p):
        if required:
            raise FileNotFoundError(p)
        return None
    ag = PPOAgent(RingEnv.OBS_DIM, 3, seed=seed)
    with open(p, "rb") as fh:
        ag.load_state_dict(pickle.load(fh))
    return ag


def energy_per_km(v, a, dt):
    """Positive tractive energy [MJ/km] (m=1500 kg, Cr=0.01, rho*CdA=0.84)."""
    P = 1500.0 * v * (a + 9.81 * 0.01) + 0.5 * 1.2 * 0.7 * v ** 3
    E = np.maximum(P, 0).sum(axis=(0, 2)) * dt                       # J per env
    D = v.sum(axis=(0, 2)) * dt                                      # m per env
    return E / np.maximum(D, 1.0) * 1e-3                             # MJ/km


def run(cfg, agent=None, controller=None, shield=None, n_envs=20, seed=1000, keep_trace=False):
    cfg = copy.deepcopy(cfg); cfg.seed = seed
    env = RingEnv(cfg, n_envs=n_envs)
    env.shield = shield
    if shield is not None:
        shield.alpha_t = None; shield.stats = {"miscov": 0, "n": 0}
    env.reset()
    env.start_recording()
    interv, infeas = np.zeros(n_envs), np.zeros(n_envs)
    coll_t = np.full(n_envs, np.inf)
    for k in range(cfg.horizon):
        if agent is not None:
            a, _ = agent.act(env.obs(), deterministic=True)
            _, _, _, info = env.step(a[..., 0] * agent.act_scale)
        elif controller is not None:
            _, _, _, info = env.step(None, controller=controller)
        else:
            _, _, _, info = env.step(np.zeros((n_envs, env.K)))
        if env.K:
            interv += info["interv"].mean(1); infeas += info["infeasible"].sum(1)
        newc = info["collision"] & ~np.isfinite(coll_t)
        coll_t[newc] = k * cfg.substeps * cfg.dt
    x, v, a = env.recording()
    sig = signals_from_recording(x, v, a, env, ttc_min=1.5)
    specs = traffic_specs(cfg.dt)
    rob = {name: f.robustness(sig) for name, f in specs.items()}
    T = v.shape[0]
    half = T // 2
    L = env.L[None, :, None]
    gap = np.mod(x[:, :, env.lead] - x, L) - env.cfg.veh_length
    clo = v - v[:, :, env.lead]
    ttc = np.where(clo > 0.05, np.maximum(gap, 0.0) / np.maximum(clo, 0.05), np.inf)
    min_ttc = np.minimum(ttc.min(axis=(0, 2)), 60.0)                  # capped at 60 s for averaging
    jerk = (np.abs(np.diff(a[:, :, env.cav_idx], axis=0)).mean(axis=(0, 2)) / cfg.dt if env.K
            else np.zeros(n_envs))
    out = pd.DataFrame({
        "min_ttc": min_ttc,
        "cav_mean_abs_jerk": jerk,
        "collision": np.isfinite(coll_t),
        "min_gap": sig["gap_min"].min(1),
        "ttc_viol_frac": (sig["ttc_marg"] < 0).mean(1),
        "std_v_ss": v[half:].std(axis=2).mean(0),                    # steady-state wave amplitude
        "mean_v": v.mean(axis=(0, 2)),
        "energy_MJkm": energy_per_km(v, a, cfg.dt),
        "cav_harsh_frac": ((a[:, :, env.cav_idx] < -4.0).mean(axis=(0, 2)) if env.K else np.zeros(n_envs)),
        "interv_rate": interv / cfg.horizon,
        "infeasible_steps": infeas,
        **{f"rho_{k}": r for k, r in rob.items()},
    })
    if shield is not None and shield.stats["n"] > 0:
        out["miscoverage"] = shield.stats["miscov"] / shield.stats["n"]
    trace = (x[:, 0], v[:, 0], a[:, 0]) if keep_trace else None
    return out, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_envs", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    scores = calibrate_scores(EnvConfig())
    rows, certs, traces = [], {}, {}

    def add(df, method, scen, seed, n_cav):
        df = df.copy(); df["method"], df["scenario"], df["seed"], df["n_cav"] = method, scen, seed, n_cav
        rows.append(df)

    # ---------------- main comparison (2 CAVs out of 22 = 9 %) ----------------
    for scen, over in SCENARIOS.items():
        for si, seed in enumerate(args.seeds):
            evs = 1000 + 100 * si
            base = EnvConfig(n_cav=2, **over)
            keep = (scen == "S3-emergency" and si == 0)
            df, tr = run(make_cfg("none", base), controller=HumanIDM(), n_envs=args.n_envs, seed=evs, keep_trace=keep)
            add(df, "Human (IDM)", scen, seed, 2)
            if tr is not None: traces[("Human (IDM)", scen)] = tr
            df, tr = run(make_cfg("none", base), controller=PISaturation(), n_envs=args.n_envs, seed=evs, keep_trace=keep)
            add(df, "PI-Saturation", scen, seed, 2)
            if tr is not None: traces[("PI-Saturation", scen)] = tr
            for m in RL_METHODS:
                ag = load_agent(m, seed, args.runs)
                if ag is None:
                    continue
                cfg = make_cfg(m, base)
                df, tr = run(cfg, agent=ag, shield=make_shield(m, cfg, scores), n_envs=args.n_envs, seed=evs,
                             keep_trace=keep)
                add(df, m, scen, seed, 2)
                if tr is not None: traces[(m, scen)] = tr
            # post-hoc shielding: PPO policy + CertiFlow shield only at deployment
            ag = load_agent("PPO", seed, args.runs)
            if ag is not None:
                cfg = make_cfg("CertiFlow", base)
                df, _ = run(cfg, agent=ag, shield=make_shield("CertiFlow", cfg, scores), n_envs=args.n_envs, seed=evs)
                add(df, "PPO+CF-shield (post-hoc)", scen, seed, 2)
            print("done", scen, seed, flush=True)

    # ---------------- penetration-rate sweep (nominal) ----------------
    for n_cav in [1, 2, 3, 4]:
        for si, seed in enumerate(args.seeds):
            evs = 5000 + 100 * si
            base = EnvConfig(n_cav=n_cav)
            df, _ = run(base, controller=HumanIDM(), n_envs=args.n_envs, seed=evs)
            add(df, "Human (IDM)", "S4-penetration", seed, n_cav)
            df, _ = run(base, controller=PISaturation(), n_envs=args.n_envs, seed=evs)
            add(df, "PI-Saturation", "S4-penetration", seed, n_cav)
            for m in ["PPO", "CertiFlow"]:
                ag = load_agent(m, seed, args.runs)
                if ag is None:
                    continue
                cfg = make_cfg(m, base)
                df, _ = run(cfg, agent=ag, shield=make_shield(m, cfg, scores), n_envs=args.n_envs, seed=evs)
                add(df, m, "S4-penetration", seed, n_cav)
        print("done penetration", n_cav, flush=True)

    ep = pd.concat(rows, ignore_index=True)
    ep.to_csv(os.path.join(args.out, "episodes.csv"), index=False)

    # ---------------- certificates ----------------
    certs["Human (IDM), 0 CAV"] = spectral_certificate(EnvConfig(n_cav=0))
    for m in RL_METHODS:
        for seed in args.seeds:
            ag = load_agent(m, seed, args.runs)
            if ag is None:
                continue
            cfg = make_cfg(m, EnvConfig(n_cav=2))
            certs[f"{m} s{seed}"] = spectral_certificate(cfg, ag, make_shield(m, cfg, scores))
    for n_cav in [1, 3, 4]:
        for seed in args.seeds:
            ag = load_agent("CertiFlow", seed, args.runs)
            if ag is None:
                continue
            cfg = make_cfg("CertiFlow", EnvConfig(n_cav=n_cav))
            certs[f"CertiFlow s{seed} K={n_cav}"] = spectral_certificate(cfg, ag, make_shield("CertiFlow", cfg, scores))
    with open(os.path.join(args.out, "certificates.json"), "w") as fh:
        json.dump({k: {kk: (vv if not isinstance(vv, np.ndarray) else np.abs(vv).tolist())
                       for kk, vv in d.items()} for k, d in certs.items()}, fh, indent=1)

    # ---------------- traces for figures ----------------
    np.savez_compressed(os.path.join(args.out, "traces_S3.npz"),
                        **{f"{m}|{part}": arr for (m, s), tr in traces.items() if tr is not None
                           for part, arr in zip("xva", tr)})

    # ---------------- summary with statistical verification ----------------
    def summ(g):
        n = len(g)
        k_coll = int(g["collision"].sum())
        k_safe = int((g["rho_phi_safe"] <= 0).sum())
        k_all = int((g["rho_phi_all"] <= 0).sum())
        ok = g[~g["collision"]]
        return pd.Series({
            "episodes": n,
            "collision_rate": k_coll / n,
            "collision_CP95_upper": clopper_pearson_upper(k_coll, n),
            "phi_safe_viol_rate": k_safe / n,
            "phi_safe_CP95_upper": clopper_pearson_upper(k_safe, n),
            "phi_ttc_sat": (g["rho_phi_ttc"] > 0).mean(),
            "phi_damp_sat": (g["rho_phi_damp"] > 0).mean(),
            "phi_comfort_sat": (g["rho_phi_comfort"] > 0).mean(),
            "phi_all_sat": 1 - k_all / n,
            "min_gap": g["min_gap"].min(),
            "ttc_viol_frac": g["ttc_viol_frac"].mean(),
            "std_v_ss": ok["std_v_ss"].mean(), "std_v_ss_sd": ok["std_v_ss"].std(),
            "mean_v": ok["mean_v"].mean(),
            "energy_MJkm": ok["energy_MJkm"].mean(), "energy_sd": ok["energy_MJkm"].std(),
            "interv_rate": g["interv_rate"].mean(),
            "infeasible_steps": g["infeasible_steps"].mean(),
            "miscoverage": g["miscoverage"].mean() if "miscoverage" in g else np.nan,
        })
    summary = ep.groupby(["scenario", "n_cav", "method"], sort=False).apply(summ, include_groups=False)
    summary.to_csv(os.path.join(args.out, "summary.csv"))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(summary.round(3))


if __name__ == "__main__":
    main()
