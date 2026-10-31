"""Conformal, cooperative discrete-time control-barrier-function (CBF) shield.

Barrier (RSS-type stopping-distance certificate, Vahidi/Ames/Shalev-Shwartz):

    h(s, v, v_l) = s - d0 - tau * v - (v^2 - v_l^2) / (2 b)

h >= 0 means that, even if the leader brakes at the physical limit b from now on,
the ego can stop behind it with margin d0 after a reaction allowance tau * v.

Discrete-time CBF condition (Agrawal & Sreenath, 2017):

    h(x_{k+1}) >= (1 - gamma) * h(x_k),      0 < gamma <= 1

Only one quantity in x_{k+1} is not under the CAV's control: the leader's
acceleration over the next step.  For a human leader we predict it with a nominal
IDM and use a *conformal* lower bound  a_l >= a_hat - q_{1-alpha}  that is valid
with probability >= 1 - alpha without any distributional assumption (split
conformal prediction, Vovk et al.; Angelopoulos & Bates).  Under distribution
shift the level alpha_t is adapted online (adaptive conformal inference,
Gibbs & Candes, 2021), which keeps long-run miscoverage at alpha.

Because h(x_{k+1}) is monotonically non-increasing in the ego acceleration a,
the set of safe actions is an interval (-inf, a_up]; we find a_up exactly by
vectorised bisection -- no QP solver is needed, the filter is O(E*K) per step.

Cooperative term: the shield additionally protects the human FOLLOWER of every
CAV, whose barrier h_f is non-decreasing in a, giving a lower bound a >= a_lo.
If a_lo > a_up the ego's own safety takes priority and the step is counted as
"infeasible" (reported in the experiments).
"""
from __future__ import annotations

import numpy as np
from .env import idm_accel


class ConformalCBFShield:
    def __init__(self, cfg, d0=2.0, tau=0.3, tau_f=0.3, gamma=0.3, alpha=0.01,
                 conformal=True, cooperative=True, adaptive=True, aci_lr=0.005,
                 scores_low=None, scores_up=None, n_bisect=25):
        self.cfg = cfg
        self.d0, self.tau, self.tau_f, self.gamma = d0, tau, tau_f, gamma
        self.alpha = alpha
        self.conformal, self.cooperative, self.adaptive = conformal, cooperative, adaptive
        self.lr = aci_lr
        self.b = cfg.b_max_phys
        self.n_bisect = n_bisect
        self.scores_low = np.sort(scores_low) if scores_low is not None else np.zeros(1)
        self.scores_up = np.sort(scores_up) if scores_up is not None else np.zeros(1)
        self.alpha_t = None
        self.stats = {"miscov": 0, "n": 0}

    # ------------------------------------------------------------------ #
    def reset(self, ids):
        pass

    def _ensure(self, env):
        if self.alpha_t is None or self.alpha_t.shape != (env.E, env.K):
            self.alpha_t = np.full((env.E, env.K), self.alpha)
            self.alpha_f = np.full((env.E, env.K), self.alpha)
            self.pred_l = np.zeros((env.E, env.K))
            self.pred_f = np.zeros((env.E, env.K))

    def _quantile(self, scores, alpha_t):
        """Conformal quantile with finite-sample correction; alpha<=0 -> worst case."""
        n = len(scores)                                   # scores are pre-sorted
        k = np.ceil((n + 1) * (1.0 - np.clip(alpha_t, 1e-9, 1.0))).astype(int) - 1
        q = scores[np.clip(k, 0, n - 1)]
        # level beyond the data (k >= n) or alpha_t <= 0  ->  worst case (no information)
        return np.where((alpha_t <= 0) | (k >= n), np.inf, q)

    def barrier(self, s, v, vl, tau):
        return s - self.d0 - tau * v - (v ** 2 - vl ** 2) / (2.0 * self.b)

    def _next_h(self, a, s, v, vl, al, tau):
        dt = self.cfg.dt
        vl2 = np.maximum(0.0, vl + al * dt)
        v2 = np.maximum(0.0, v + a * dt)
        s2 = s + 0.5 * (vl + vl2) * dt - 0.5 * (v + v2) * dt
        return self.barrier(s2, v2, vl2, tau)

    # ------------------------------------------------------------------ #
    def _nominal(self, env, ego, lead):
        p = env.cfg.idm
        g = env.gaps()
        return idm_accel(env.v[:, ego], env.v[:, lead], g[:, ego], p.v0, p.T, p.a, p.b, p.s0, p.delta)

    def filter(self, env, a_des):
        self._ensure(env)
        i = env.cav_idx
        L, F = env.lead[i], env.foll[i]
        g = env.gaps()
        v = env.v
        s, ve, vl = g[:, i], v[:, i], v[:, L]

        # ---- leader acceleration lower bound (conformal) ----
        a_hat_l = self._nominal(env, L, env.lead[L])
        self.pred_l = a_hat_l
        if self.conformal:
            q_l = self._quantile(self.scores_low, self.alpha_t if self.adaptive else
                                 np.full_like(self.alpha_t, self.alpha))
        else:
            q_l = np.zeros_like(a_hat_l)
        al_lb = np.maximum(a_hat_l - q_l, -self.b)
        lead_is_cav = env.is_cav[L]
        al_lb = np.where(lead_is_cav[None, :], -self.b, al_lb)
        al_lb = np.clip(al_lb, -self.b, env.cfg.a_max_phys)

        h0 = self.barrier(s, ve, vl, self.tau)
        rhs = (1.0 - self.gamma) * h0
        amax = env.cfg.a_max_phys

        def h_next(a):
            return self._next_h(a, s, ve, vl, al_lb, self.tau)

        # a_up = largest a in [-b, amax] with h' >= rhs   (h' non-increasing in a)
        lo = np.full_like(a_des, -self.b)
        hi = np.full_like(a_des, amax)
        top_ok = h_next(hi) >= rhs
        for _ in range(self.n_bisect):
            mid = 0.5 * (lo + hi)
            ok = h_next(mid) >= rhs
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        a_up = np.where(top_ok, amax, lo)        # if even -b fails, lo stays at -b (max braking)
        infeasible = np.zeros_like(a_des)
        a_lo = np.full_like(a_des, -self.b)

        # ---- cooperative constraint for the human follower ----
        if self.cooperative:
            sf, vf = g[:, F], v[:, F]
            a_hat_f = self._nominal(env, F, i)
            self.pred_f = a_hat_f
            if self.conformal:
                q_f = self._quantile(self.scores_up, self.alpha_f if self.adaptive else
                                     np.full_like(self.alpha_f, self.alpha))
            else:
                q_f = np.zeros_like(a_hat_f)
            af_ub = np.clip(a_hat_f + q_f, -self.b, amax)
            hf0 = self.barrier(sf, vf, ve, self.tau_f)
            rhs_f = (1.0 - self.gamma) * hf0

            def hf_next(a):
                # the follower is the "ego"; the CAV is its leader, accelerating at a
                return self._next_h(af_ub, sf, vf, ve, a, self.tau_f)
            # a_lo = smallest a in [-b, amax] with hf' >= rhs_f   (hf' non-decreasing in a)
            lo2 = np.full_like(a_des, -self.b)
            hi2 = np.full_like(a_des, amax)
            bot_ok = hf_next(lo2) >= rhs_f
            for _ in range(self.n_bisect):
                mid = 0.5 * (lo2 + hi2)
                ok = hf_next(mid) >= rhs_f
                hi2 = np.where(ok, mid, hi2)
                lo2 = np.where(ok, lo2, mid)
            a_lo = np.where(bot_ok, -self.b, hi2)
            infeasible = (a_lo > a_up).astype(float)
            a_lo = np.minimum(a_lo, a_up)          # ego safety has priority
        a_out = np.clip(a_des, a_lo, a_up)
        return a_out, infeasible

    # ------------------------------------------------------------------ #
    def update(self, env, v_before):
        """Adaptive conformal update after observing the realised accelerations."""
        if not (self.conformal and self.adaptive):
            return
        i = env.cav_idx
        L, F = env.lead[i], env.foll[i]
        dt = env.cfg.dt
        a_l = (env.v[:, L] - v_before[:, L]) / dt
        a_f = (env.v[:, F] - v_before[:, F]) / dt
        q_l = self._quantile(self.scores_low, self.alpha_t)
        q_f = self._quantile(self.scores_up, self.alpha_f)
        # ignore the v=0 saturation, where the realised accel is clipped upwards (safe side)
        err_l = (a_l < self.pred_l - q_l - 1e-9) & (env.v[:, L] > 0)
        err_f = (a_f > self.pred_f + q_f + 1e-9)
        self.stats["miscov"] += int(err_l.sum()); self.stats["n"] += err_l.size
        self.alpha_t = self.alpha_t + self.lr * (self.alpha - err_l)
        self.alpha_f = self.alpha_f + self.lr * (self.alpha - err_f)
        self.alpha_t = np.clip(self.alpha_t, -0.05, self.alpha)
        self.alpha_f = np.clip(self.alpha_f, -0.05, self.alpha)


def calibrate_scores(cfg, n_envs=16, seconds=200.0, seed=12345):
    """Collect nonconformity scores of the nominal-IDM predictor on human-only data.

    score_low = a_hat - a_true  (large when the human brakes harder than predicted)
    score_up  = a_true - a_hat  (large when the human accelerates more than predicted)
    The calibration traffic uses a different seed (exchangeable but disjoint data).
    """
    from .env import RingEnv
    import copy
    c = copy.deepcopy(cfg); c.n_cav = 0; c.seed = seed; c.perturbations = []
    env = RingEnv(c, n_envs=n_envs)
    env.reset()
    p = c.idm
    lo, up = [], []
    for _ in range(int(seconds / c.dt)):
        g = env.gaps()
        a_hat = idm_accel(env.v, env.v[:, env.lead], g, p.v0, p.T, p.a, p.b, p.s0, p.delta)
        v0 = env.v.copy()
        a = env._hdv_accel()
        env._integrate(a, np.arange(env.E))
        a_true = (env.v - v0) / c.dt
        mask = env.v > 0
        lo.append((a_hat - a_true)[mask]); up.append((a_true - a_hat)[mask])
    return np.concatenate(lo), np.concatenate(up)
