"""Safe online adaptation: fine-tune a CertiFlow policy INSIDE SUMO with the shield always on.

E parallel SUMO rings (EIDM humans) form a vectorised environment with the same
observation, action timing, reward and shield as the NumPy training environment.
Starting from a policy trained in NumPy, PPO continues training in SUMO; because every
action passes through the conformal CBF shield, exploration in the new domain is safe.

Usage:  python sumo_train.py --init CertiFlow-DR --seed 0 --iters 80
Writes  runs/<init>-FT_s<seed>/{policy.pkl, log.csv}
"""
from __future__ import annotations

import argparse, csv, os, pickle, time
import numpy as np

import sumo_eval as S

DT, SUB, HOR = 0.1, 5, 200            # 0.1 s sim step, 2 Hz control, 100 s episodes


class SumoRingVec:
    def __init__(self, work, n_envs, shield_fn, seed=0, k=2, w_stl=1.0, w_interv=0.2, v_ref=None):
        self.work, self.E, self.K, self.n = work, n_envs, k, 22
        self.shield_fn, self.base_seed = shield_fn, seed * 1000 + 50000
        self.w_stl, self.w_interv = w_stl, w_interv
        self.v_ref = v_ref          # phi_flow: G(mean speed >= 0.95 v_ref); robustness penalised when < 0
        self.cav_idx = [int(i) for i in (np.arange(k) * self.n // k)]
        self.envs = [None] * n_envs
        self.ep = 0

    # --------------------------------------------------------------- #
    def _reset_one(self, e):
        if self.envs[e] is not None:
            self.envs[e].close()
        self.ep += 1
        env = S.RingSUMO(self.work, k=self.K, seed=self.base_seed + self.ep, label=f"tr{self.base_seed}_{e}")
        env.start()
        for _ in range(int(60 / DT)):
            env.c.simulationStep()
        for cid in env.cav_ids:
            env.c.vehicle.setType(cid, "cav"); env.c.vehicle.setSpeedMode(cid, 0)
        env.t = 0
        env.d, env.v = env.state()
        env.a_prev = np.zeros(self.n)
        env.shield = self.shield_fn()
        self.envs[e] = env

    def reset(self):
        for e in range(self.E):
            self._reset_one(e)
        return self.obs()

    def _obs_one(self, env):
        g, _ = env.gaps(env.d); v = env.v; n = self.n
        o = []
        for i in self.cav_idx:
            L1, F1 = (i + 1) % n, (i - 1) % n; L2 = (L1 + 1) % n
            o.append([v[i] / 20, v[L1] / 20, (v[L1] - v[i]) / 5, g[i] / 30, v[F1] / 20, g[F1] / 30,
                      env.a_prev[i] / 3, v[L2] / 20, g[L1] / 30, v.mean() / 20])
        return o

    def obs(self):
        return np.array([self._obs_one(env) for env in self.envs])

    def global_feats(self):
        f = []
        for env in self.envs:
            g, _ = env.gaps(env.d)
            f.append([env.v.std() / 2.0, env.v.mean() / 10.0, g.min() / 10.0])
        return np.repeat(np.array(f)[:, None, :], self.K, 1)

    # --------------------------------------------------------------- #
    def step(self, act):
        rew = np.zeros((self.E, self.K)); done = np.zeros(self.E, bool)
        coll = np.zeros(self.E, bool); interv = np.zeros(self.E)
        n = self.n
        for e, env in enumerate(self.envs):
            c, sh = env.c, env.shield
            rho_min = np.full(self.K, np.inf)
            for _ in range(SUB):
                g, _ = env.gaps(env.d); v = env.v
                bounds = {}
                for j, i in enumerate(self.cav_idx):
                    L1, F1 = (i + 1) % n, (i - 1) % n
                    a_des = float(np.clip(act[e, j], -3, 2))
                    if sh.predictor == "persist":
                        al_hat, af_hat = float(env.a_prev[L1]), float(env.a_prev[F1])
                    else:
                        al_hat = S.idm_hat(v[L1], v[(L1 + 1) % n], g[L1]); af_hat = S.idm_hat(v[F1], v[i], g[F1])
                    a_ex, _, bounds[i] = sh.filter(i, a_des, g[i], v[i], v[L1], al_hat, False, g[F1], v[F1], af_hat)
                    interv[e] += abs(a_ex - a_des) > 1e-6
                    rew[e, j] -= self.w_interv * abs(a_ex - a_des) / SUB
                    rew[e, j] -= 0.05 * a_ex ** 2 / SUB
                    c.vehicle.setSpeed(env.ids[i], max(0.0, v[i] + a_ex * DT))
                c.simulationStep()
                d2, v2 = env.state()
                acc = (v2 - v) / DT
                for i, bnd in bounds.items():
                    sh.update(i, bnd, acc[(i + 1) % n] if v2[(i + 1) % n] > 0 else None, acc[(i - 1) % n])
                env.a_prev, env.d, env.v = acc, d2, v2
                g2, _ = env.gaps(d2)
                rew[e] += (-v2.std() + 0.1 * v2.mean()) / SUB
                for j, i in enumerate(self.cav_idx):
                    L1, F1 = (i + 1) % n, (i - 1) % n
                    ci = max(0.0, v2[i] - v2[L1]); cf = max(0.0, v2[F1] - v2[i])
                    rho_min[j] = min(rho_min[j], g2[i] - 1, g2[i] - 1.5 * ci, g2[F1] - 1, g2[F1] - 1.5 * cf)
                if g2.min() < 0:
                    coll[e] = True
            rew[e] += self.w_stl * np.clip(rho_min / 5.0, -1.0, 0.2)
            if self.v_ref is not None:
                rew[e] += self.w_stl * np.clip((env.v.mean() - 0.95 * self.v_ref) / 1.0, -1.0, 0.0)
            env.t += 1
            if coll[e]:
                rew[e] -= 50.0
            done[e] = coll[e] or env.t >= HOR
        info = {"collision": coll, "interv": interv / (SUB * self.K)}
        return self.obs(), rew, done, info

    def close(self):
        for env in self.envs:
            if env is not None:
                env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="CertiFlow-DR")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--n_envs", type=int, default=4)
    ap.add_argument("--rollout", type=int, default=100)
    ap.add_argument("--work", default="sumo_work")
    args = ap.parse_args()
    work = os.path.abspath(args.work); os.makedirs(work, exist_ok=True)

    pred, lo, up, width = S.select_and_calibrate(S.calibrate_sumo_ring, work)
    print("predictor", pred, {k: round(float(v), 3) for k, v in width.items()}, flush=True)
    shield_fn = lambda: S.Shield(lo, up, predictor=pred)
    v_ref = float(np.mean([S.run_ring_episode(work, "Human", seed=700 + i)["mean_v"] for i in range(3)]))
    print(f"human-only reference speed v_ref = {v_ref:.3f} m/s", flush=True)

    agent = S.load_agent(args.init, args.seed)
    agent.opt_a.lr = agent.opt_c.lr = 1e-4                # gentler fine-tuning
    agent.obs_norm.n = agent.glob_norm.n = 1e5            # keep the NumPy normalisation, adapt slowly
    agent.rng = np.random.default_rng(100 + args.seed)
    env = SumoRingVec(work, args.n_envs, shield_fn, seed=args.seed, v_ref=v_ref)
    run_dir = os.path.join("runs", f"{args.init}-FT_s{args.seed}"); os.makedirs(run_dir, exist_ok=True)
    obs = env.reset()
    T, E, K = args.rollout, env.E, env.K
    ep_ret = np.zeros(E); rets, colls = [], []
    t0 = time.time()
    with open(os.path.join(run_dir, "log.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["iter", "steps", "ep_return", "collisions", "interv_rate", "std_v", "mean_v", "time_s"])
        total_coll = 0
        for it in range(args.iters):
            B = {k: [] for k in ("o", "c", "a", "lp", "r", "v", "d")}
            iv, sv, mv = [], [], []
            for t in range(T):
                cobs = np.concatenate([obs, env.global_feats()], -1)
                a, lp = agent.act(obs); val = agent.value(cobs)
                obs2, r, done, info = env.step(a[..., 0] * agent.act_scale)
                for k, x in zip(("o", "c", "a", "lp", "r", "v"), (obs, cobs, a, lp, r, val)):
                    B[k].append(x)
                B["d"].append(np.repeat(done[:, None], K, 1).astype(float))
                iv.append(info["interv"].mean()); sv.append(np.mean([e.v.std() for e in env.envs]))
                mv.append(np.mean([e.v.mean() for e in env.envs]))
                ep_ret += r.mean(1); total_coll += int(info["collision"].sum())
                for e in np.where(done)[0]:
                    rets.append(ep_ret[e]); colls.append(bool(info["collision"][e])); ep_ret[e] = 0
                    env._reset_one(e)
                obs2 = env.obs() if done.any() else obs2
                obs = obs2
            Bo, Bc, Ba = np.array(B["o"]), np.array(B["c"]), np.array(B["a"])
            Blp, Br, Bv, Bd = np.array(B["lp"]), np.array(B["r"]), np.array(B["v"]), np.array(B["d"])
            last_v = agent.value(np.concatenate([obs, env.global_feats()], -1))
            adv, ret = agent.gae(Br.reshape(T, -1) / 10.0, Bv.reshape(T, -1), Bd.reshape(T, -1), last_v.reshape(-1))
            agent.obs_norm.update(Bo); agent.glob_norm.update(Bc)
            agent.update(Bo.reshape(-1, 10), Bc.reshape(-1, 13), Ba.reshape(-1, 1), Blp.reshape(-1),
                         adv.reshape(-1), ret.reshape(-1))
            rr = np.mean(rets[-8:]) if rets else np.nan
            w.writerow([it, (it + 1) * T * E, rr, total_coll, np.mean(iv), np.mean(sv), np.mean(mv), time.time() - t0])
            fh.flush()
            if it % 5 == 0 or it == args.iters - 1:
                print(f"[FT {args.init} s{args.seed}] it {it:3d} ret {rr:7.2f} collisions so far {total_coll} "
                      f"interv {np.mean(iv):.3f} std_v {np.mean(sv):.3f} mean_v {np.mean(mv):.2f} t {time.time()-t0:.0f}s",
                      flush=True)
    env.close()
    with open(os.path.join(run_dir, "policy.pkl"), "wb") as fh:
        pickle.dump(agent.state_dict(), fh)


if __name__ == "__main__":
    main()
