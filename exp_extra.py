"""Supplementary experiments.

  sumo_traces  : SUMO ring trajectories (Human EIDM, PPO, CertiFlow) -> results/sumo_traces.npz
  residuals    : predictor-selection residuals (IDM vs persistence) in SUMO ring/highway -> results/residuals.npz
  runtime      : wall-clock cost of the shield and of the policy -> results/runtime.json
  phase        : CAV states (gap, closing speed, speed, barrier) with/without shield, S2+S3 -> results/phase.npz

Usage: python exp_extra.py [runtime] [phase] [residuals] [sumo_traces]
(runtime, residuals and sumo_traces import sumo_eval.py and therefore need SUMO)
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np


def sumo_traces():
    import sumo_eval as S
    work = os.path.abspath("sumo_extra"); os.makedirs(work, exist_ok=True)
    pred, lo, up, _ = S.select_and_calibrate(S.calibrate_sumo_ring, work)
    out = {}
    for em in (False, True):
        tag = "em" if em else "nom"
        for m in ("Human", "PPO", "CertiFlow-DR-FT"):
            tr = {}
            if m == "Human":
                S.run_ring_episode(work, "Human", seed=2003, emergency=em, trace=tr)
            else:
                ag = S.load_agent(m, 0)
                sh = None if S.SHIELD_CFG.get(m) is None else S.Shield(lo, up, predictor=pred, **S.SHIELD_CFG[m])
                S.run_ring_episode(work, m, agent=ag, shield=sh, seed=2003, emergency=em, trace=tr)
            out[f"{m}|{tag}|pos"] = tr["pos"]; out[f"{m}|{tag}|v"] = tr["v"]; out["L"] = tr["L"]
            print("trace", m, tag, flush=True)
    np.savez_compressed("results/sumo_traces.npz", **out)


def residuals():
    import sumo_eval as S
    work = os.path.abspath("sumo_extra"); os.makedirs(work, exist_ok=True)
    ring = S.calibrate_sumo_ring(work, seed=902)
    hw = S.calibrate_sumo_hw(work, seed=902)
    from certiflow.env import EnvConfig
    from certiflow.shield import calibrate_scores
    lo, _ = calibrate_scores(EnvConfig(), seconds=100)
    rng = np.random.default_rng(0)
    sub = lambda a: rng.choice(a, size=min(len(a), 40000), replace=False)
    np.savez_compressed("results/residuals.npz", ring_idm=sub(ring["idm"]), ring_persist=sub(ring["persist"]),
                        hw_idm=sub(hw["idm"]), hw_persist=sub(hw["persist"]), numpy_idm=sub(lo))
    print("residuals saved", flush=True)


def runtime():
    from certiflow.env import RingEnv, EnvConfig
    from certiflow.shield import ConformalCBFShield, calibrate_scores
    from evaluate import load_agent
    import sumo_eval as S
    lo, up = calibrate_scores(EnvConfig(), seconds=60)
    res = {}
    ag = load_agent("CertiFlow-DR-FT", 0, required=True)
    for K in (1, 2, 4, 8):
        cfg = EnvConfig(n_cav=K, n_vehicles=max(22, 4 * K))
        env = RingEnv(cfg, n_envs=1); sh = ConformalCBFShield(cfg, scores_low=lo, scores_up=up); env.shield = sh; env.reset()
        a = np.zeros((1, K))
        sh.filter(env, a)
        n = 2000; t0 = time.perf_counter()
        for _ in range(n):
            sh.filter(env, a)
        res[f"vectorised_shield_K{K}_us_per_call"] = 1e6 * (time.perf_counter() - t0) / n
    # per-CAV scalar shield (as deployed in SUMO, one call per CAV)
    shs = S.Shield(lo, up)
    n = 5000; t0 = time.perf_counter()
    for i in range(n):
        shs.filter(0, 1.0, 6.0, 6.0, 4.5, -0.5, False, 5.0, 6.4, 0.2)
    res["scalar_shield_us_per_cav_step"] = 1e6 * (time.perf_counter() - t0) / n
    o = np.zeros((1, 1, 10)); ag.act(o, deterministic=True)
    t0 = time.perf_counter()
    for _ in range(n):
        ag.act(o, deterministic=True)
    res["policy_us_per_cav_decision"] = 1e6 * (time.perf_counter() - t0) / n
    json.dump(res, open("results/runtime.json", "w"), indent=1); print(res, flush=True)


def phase():
    """Record every CAV's (gap, closing speed, speed, barrier value) with and without the shield."""
    from certiflow.env import RingEnv, EnvConfig
    from certiflow.shield import ConformalCBFShield, calibrate_scores
    from certiflow.methods import make_cfg
    from evaluate import load_agent, SCENARIOS
    lo, up = calibrate_scores(EnvConfig())
    out = {}
    for label, shielded in (("shield", True), ("noshield", False)):
        S_, DV, V, H, C = [], [], [], [], []
        for scen in ("S2-shift", "S3-emergency"):
            for si, seed in enumerate([0, 1, 2]):
                cfg = make_cfg("CertiFlow", EnvConfig(n_cav=2, **SCENARIOS[scen])); cfg.seed = 1000 + 100 * si
                env = RingEnv(cfg, n_envs=20)
                sh = ConformalCBFShield(cfg, scores_low=lo, scores_up=up) if shielded else None
                env.shield = sh; env.reset(); env.start_recording()
                ag = load_agent("CertiFlow-DR-FT", seed, required=True)
                for _ in range(cfg.horizon):
                    a, _ = ag.act(env.obs(), deterministic=True)
                    env.step(a[..., 0] * ag.act_scale)
                x, v, _ = env.recording()
                i = env.cav_idx; Lead = env.lead[i]
                gap = np.mod(x[:, :, Lead] - x[:, :, i], env.L[None, :, None]) - 5.0      # (T,E,K)
                dv = v[:, :, i] - v[:, :, Lead]
                hb = gap - 2.0 - 0.3 * v[:, :, i] - (v[:, :, i] ** 2 - v[:, :, Lead] ** 2) / 14.0
                S_.append(gap[::5].ravel()); DV.append(dv[::5].ravel()); V.append(v[:, :, i][::5].ravel())
                H.append(hb[::5].ravel()); C.append(np.repeat((gap.min(axis=(0, 2)) < 0)[None, :, None], gap[::5].shape[0], 0)
                                                       .repeat(gap.shape[2], 2).ravel())
        for k, arr in zip(("gap", "dv", "v", "h", "crash_ep"), (S_, DV, V, H, C)):
            out[f"{label}|{k}"] = np.concatenate(arr)
        print("phase", label, flush=True)
    np.savez_compressed("results/phase.npz", **out)


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["runtime", "phase", "residuals", "sumo_traces"]):
        globals()[name]()
