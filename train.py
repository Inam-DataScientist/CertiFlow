"""Train a parameter-shared PPO policy for the CAVs.

Usage:  python train.py --method CertiFlow --seed 0 --iters 300 [--dr]
Writes  runs/<method>[-DR]_s<seed>/{policy.pkl, log.csv}
"""
from __future__ import annotations

import argparse, csv, os, pickle, time
import numpy as np

from certiflow.env import RingEnv, EnvConfig
from certiflow.ppo import PPOAgent
from certiflow.shield import calibrate_scores
from certiflow.methods import make_cfg, make_shield


def global_feats(env):
    g = env.gaps()
    f = np.stack([env.v.std(1) / 2.0, env.v.mean(1) / 10.0, g.min(1) / 10.0], -1)   # (E,3)
    return np.repeat(f[:, None, :], env.K, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="CertiFlow")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--n_envs", type=int, default=16)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--n_cav", type=int, default=2)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--dr", action="store_true", help="domain-randomised human model (reaction delay, perception noise)")
    args = ap.parse_args()

    cfg = make_cfg(args.method, EnvConfig(n_cav=args.n_cav, seed=args.seed, dr=args.dr))
    scores = calibrate_scores(EnvConfig(dr=args.dr))  # calibration data disjoint from training seeds
    env = RingEnv(cfg, n_envs=args.n_envs)
    env.shield = make_shield(args.method, cfg, scores)
    agent = PPOAgent(RingEnv.OBS_DIM, 3, seed=args.seed)
    run_dir = os.path.join(args.out, f"{args.method}{'-DR' if args.dr else ''}_s{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    obs = env.reset()
    ep_ret = np.zeros(env.E); ep_rets, ep_coll = [], []
    T, E, K = args.rollout, env.E, env.K
    t0 = time.time()
    with open(os.path.join(run_dir, "log.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iter", "steps", "ep_return", "collisions_per_ep", "interv_rate", "std_v", "kl",
                    "log_std", "time_s"])
        for it in range(args.iters):
            B_obs = np.zeros((T, E, K, RingEnv.OBS_DIM)); B_cobs = np.zeros((T, E, K, RingEnv.OBS_DIM + 3))
            B_act = np.zeros((T, E, K, 1)); B_lp = np.zeros((T, E, K)); B_rew = np.zeros((T, E, K))
            B_val = np.zeros((T, E, K)); B_done = np.zeros((T, E, K))
            interv, stdv = [], []
            for t in range(T):
                cobs = np.concatenate([obs, global_feats(env)], -1)
                a, lp = agent.act(obs)
                v = agent.value(cobs)
                obs2, r, done, info = env.step(a[..., 0] * agent.act_scale)
                B_obs[t], B_cobs[t], B_act[t], B_lp[t], B_rew[t], B_val[t] = obs, cobs, a, lp, r, v
                B_done[t] = done[:, None]
                interv.append(info["interv"].mean()); stdv.append(env.v.std(1).mean())
                ep_ret += r.mean(1)
                if done.any():
                    ids = np.where(done)[0]
                    ep_rets += list(ep_ret[ids]); ep_coll += list(info["collision"][ids])
                    ep_ret[ids] = 0
                    obs2 = env.reset(ids)
                obs = obs2
            agent.obs_norm.update(B_obs); agent.glob_norm.update(B_cobs)
            last_v = agent.value(np.concatenate([obs, global_feats(env)], -1))
            adv, ret = agent.gae(B_rew.reshape(T, -1) / 10.0, B_val.reshape(T, -1),
                                 B_done.reshape(T, -1), last_v.reshape(-1))
            st = agent.update(B_obs.reshape(-1, RingEnv.OBS_DIM), B_cobs.reshape(-1, RingEnv.OBS_DIM + 3),
                              B_act.reshape(-1, 1), B_lp.reshape(-1), adv.reshape(-1), ret.reshape(-1))
            # rolling stats over the last 32 finished episodes
            rr = np.mean(ep_rets[-32:]) if ep_rets else np.nan
            cc = np.sum(ep_coll[-32:]) / max(1, len(ep_coll[-32:])) if ep_coll else np.nan
            w.writerow([it, (it + 1) * T * E, rr, cc, np.mean(interv), np.mean(stdv), st["kl"],
                        float(agent.log_std[0]), time.time() - t0])
            fh.flush()
            if it % 10 == 0 or it == args.iters - 1:
                print(f"[{args.method}{'-DR' if args.dr else ''} s{args.seed}] it {it:4d} ret {rr:8.2f} coll/ep {cc:.3f} "
                      f"interv {np.mean(interv):.3f} std_v {np.mean(stdv):.3f} kl {st['kl']:.4f} "
                      f"t {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(run_dir, "policy.pkl"), "wb") as fh:
        pickle.dump(agent.state_dict(), fh)


if __name__ == "__main__":
    main()
