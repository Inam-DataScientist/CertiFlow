"""Dependency-free PPO (Schulman et al., 2017) in NumPy, for parameter-shared MARL.

* Actor  : MLP(obs) -> Gaussian mean; state-independent log-std.  Shared by all CAVs
           (decentralised execution: each CAV only uses its own observation).
* Critic : MLP(obs ++ global features) -> V.  Global features are only used during
           training (centralised training, MAPPO-style, Yu et al., 2022).
* GAE(lambda) advantages, clipped surrogate, entropy bonus, Adam, grad-norm clipping,
  running observation normalisation.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
class MLP:
    def __init__(self, sizes, rng, out_scale=1.0):
        self.W, self.b = [], []
        for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
            gain = np.sqrt(2.0) if i < len(sizes) - 2 else out_scale
            q, _ = np.linalg.qr(rng.standard_normal((max(m, n), min(m, n))))   # orthogonal init
            w = q if m >= n else q.T
            self.W.append(gain * w[:m, :n]); self.b.append(np.zeros(n))

    def params(self):
        return self.W + self.b

    def forward(self, x):
        self.cache = [x]
        h = x
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = h @ w + b
            h = np.tanh(z) if i < len(self.W) - 1 else z
            self.cache.append(h)
        return h

    def backward(self, dout):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for i in reversed(range(len(self.W))):
            h_in = self.cache[i]
            gW[i] = h_in.T @ d
            gb[i] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * (1.0 - self.cache[i] ** 2)
        return gW + gb


class Adam:
    def __init__(self, params, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t); vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class RunningNorm:
    def __init__(self, dim):
        self.mean, self.var, self.n = np.zeros(dim), np.ones(dim), 1e-4

    def update(self, x):
        x = x.reshape(-1, x.shape[-1])
        bm, bv, bn = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean; tot = self.n + bn
        self.mean = self.mean + d * bn / tot
        self.var = (self.var * self.n + bv * bn + d ** 2 * self.n * bn / tot) / tot
        self.n = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)


# --------------------------------------------------------------------------- #
class PPOAgent:
    def __init__(self, obs_dim, glob_dim, act_dim=1, hidden=64, seed=0, lr=3e-4,
                 clip=0.2, ent_coef=1e-3, vf_coef=0.5, epochs=8, minibatch=1024, max_grad=0.5,
                 gamma=0.99, lam=0.95, act_scale=2.0):
        self.rng = np.random.default_rng(seed)
        self.actor = MLP([obs_dim, hidden, hidden, act_dim], self.rng, out_scale=0.01)
        self.critic = MLP([obs_dim + glob_dim, hidden, hidden, 1], self.rng, out_scale=1.0)
        self.log_std = np.full(act_dim, -0.5)
        self.opt_a = Adam(self.actor.params() + [self.log_std], lr)
        self.opt_c = Adam(self.critic.params(), lr)
        self.obs_norm = RunningNorm(obs_dim)
        self.glob_norm = RunningNorm(obs_dim + glob_dim)
        self.clip, self.ent, self.vf, self.epochs, self.mb = clip, ent_coef, vf_coef, epochs, minibatch
        self.max_grad, self.gamma, self.lam = max_grad, gamma, lam
        self.act_scale = act_scale     # network output * act_scale = acceleration [m/s^2]

    # ---------------------------------------------------------------- #
    def act(self, obs, deterministic=False):
        """obs (..., obs_dim) -> action (..., 1), logp (...)."""
        shp = obs.shape[:-1]
        o = self.obs_norm(obs.reshape(-1, obs.shape[-1]))
        mu = self.actor.forward(o)
        std = np.exp(self.log_std)
        a = mu if deterministic else mu + std * self.rng.standard_normal(mu.shape)
        logp = (-0.5 * ((a - mu) / std) ** 2 - self.log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        return a.reshape(*shp, -1), logp.reshape(shp)

    def value(self, cobs):
        c = self.glob_norm(cobs.reshape(-1, cobs.shape[-1]))
        return self.critic.forward(c)[:, 0].reshape(cobs.shape[:-1])

    # ---------------------------------------------------------------- #
    def gae(self, rew, val, done, last_val):
        """rew, val, done: (T, B); returns advantages and returns (T, B)."""
        T = rew.shape[0]
        adv = np.zeros_like(rew); last = 0.0
        for t in reversed(range(T)):
            nv = last_val if t == T - 1 else val[t + 1]
            nonterm = 1.0 - done[t]
            delta = rew[t] + self.gamma * nv * nonterm - val[t]
            last = delta + self.gamma * self.lam * nonterm * last
            adv[t] = last
        return adv, adv + val

    def update(self, obs, cobs, act, logp_old, adv, ret):
        n = obs.shape[0]
        o_all = self.obs_norm(obs)
        c_all = self.glob_norm(cobs)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        stats = {"kl": 0.0, "clipfrac": 0.0, "vloss": 0.0}
        for ep in range(self.epochs):
            idx = self.rng.permutation(n)
            for s in range(0, n, self.mb):
                j = idx[s:s + self.mb]; m = len(j)
                o, c, a, lp0, A, R = o_all[j], c_all[j], act[j], logp_old[j], adv[j], ret[j]
                # ---- actor ----
                mu = self.actor.forward(o)
                std = np.exp(self.log_std)
                z = (a - mu) / std
                lp = (-0.5 * z ** 2 - self.log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
                ratio = np.exp(lp - lp0)
                unclipped = ((A >= 0) & (ratio <= 1 + self.clip)) | ((A < 0) & (ratio >= 1 - self.clip))
                dlp = -(A * ratio * unclipped) / m                  # d loss / d logp
                dmu = (dlp[:, None] * z / std)                       # d logp/d mu = z/std
                dlogstd = (dlp[:, None] * (z ** 2 - 1)).sum(0) - self.ent  # entropy bonus
                g_a = self.actor.backward(dmu) + [dlogstd]
                # ---- critic ----
                v = self.critic.forward(c)[:, 0]
                dv = self.vf * (v - R)[:, None] / m
                g_c = self.critic.backward(dv)
                for grads, opt in ((g_a, self.opt_a), (g_c, self.opt_c)):
                    gn = np.sqrt(sum((g ** 2).sum() for g in grads))
                    if gn > self.max_grad:
                        grads = [g * self.max_grad / gn for g in grads]
                    opt.step(grads)
                self.log_std[:] = np.clip(self.log_std, -3.0, 0.5)
                stats["kl"] += float(((ratio - 1) - np.log(ratio + 1e-12)).mean())
                stats["clipfrac"] += float((~unclipped).mean())
                stats["vloss"] += float(((v - R) ** 2).mean())
        nb = self.epochs * int(np.ceil(n / self.mb))
        return {k: v / nb for k, v in stats.items()}

    # ---------------------------------------------------------------- #
    def state_dict(self):
        return {"actor_W": self.actor.W, "actor_b": self.actor.b, "log_std": self.log_std,
                "critic_W": self.critic.W, "critic_b": self.critic.b,
                "obs_mean": self.obs_norm.mean, "obs_var": self.obs_norm.var,
                "glob_mean": self.glob_norm.mean, "glob_var": self.glob_norm.var,
                "act_scale": self.act_scale}

    def load_state_dict(self, d):
        for i in range(len(self.actor.W)):
            self.actor.W[i][...] = d["actor_W"][i]; self.actor.b[i][...] = d["actor_b"][i]
        for i in range(len(self.critic.W)):
            self.critic.W[i][...] = d["critic_W"][i]; self.critic.b[i][...] = d["critic_b"][i]
        self.log_std[...] = d["log_std"]
        self.obs_norm.mean, self.obs_norm.var = d["obs_mean"], d["obs_var"]
        self.glob_norm.mean, self.glob_norm.var = d["glob_mean"], d["glob_var"]
        self.act_scale = float(d["act_scale"])
