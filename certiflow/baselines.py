"""Non-learning reference controllers for the CAVs."""
from __future__ import annotations

import numpy as np


class HumanIDM:
    """CAVs drive like the surrounding humans (noisy IDM) -> 0 % effective penetration."""
    name = "Human (IDM)"

    def __call__(self, env):
        return env._hdv_accel()[:, env.cav_idx]


class PISaturation:
    """PI controller with saturation (Stern et al., 2018, Transp. Res. C 89:205-221).

    The CAV tracks the running average of its own past speed (a proxy for the
    equilibrium speed), blended with the leader speed when the gap is small.
    """
    name = "PI-Saturation"

    def __init__(self, window_s=38.0, g_l=7.0, g_u=30.0, v_catch=1.0, gamma=2.0, a_lim=(-3.0, 2.0)):
        self.window_s, self.g_l, self.g_u, self.v_catch, self.gamma = window_s, g_l, g_u, v_catch, gamma
        self.a_lim = a_lim
        self.hist = None

    def __call__(self, env):
        i, L = env.cav_idx, env.lead[env.cav_idx]
        v, vl = env.v[:, i], env.v[:, L]
        gap = env.gaps()[:, i]
        n = int(self.window_s / env.cfg.dt)
        if self.hist is None or self.hist.shape[1:] != v.shape:
            self.hist = np.repeat(v[None], 1, 0)
            self.v_cmd = v.copy()
        self.hist = np.concatenate([self.hist, v[None]], 0)[-n:]
        U = self.hist.mean(0)
        v_target = U + self.v_catch * np.clip((gap - self.g_l) / (self.g_u - self.g_l), 0, 1)
        dx_s = np.maximum(2.0 * (vl - v), 4.0)
        alpha = np.clip((gap - dx_s) / self.gamma, 0, 1)
        beta = 1.0 - 0.5 * alpha
        self.v_cmd = beta * (alpha * v_target + (1 - alpha) * vl) + (1 - beta) * self.v_cmd
        return np.clip((self.v_cmd - v) / env.cfg.dt, *self.a_lim)
