"""Design-time certificates for the closed loop.

1. Local asymptotic stability of the *collective* traffic state
   ------------------------------------------------------------------
   The noise-free closed-loop ring (humans = nominal IDM, CAVs = deterministic learned
   policy + shield) is a discrete-time map z_{k+1} = Phi(z_k) at the control period.
   We (i) compute its uniform-flow equilibrium z*, (ii) linearise Phi at z* by central
   finite differences, and (iii) compute the spectral radius of the Jacobian after
   removing the trivial eigenvalue 1 that corresponds to rigid translation along the
   ring.  rho(J) < 1  certifies that small perturbations -- i.e. stop-and-go waves --
   decay exponentially (Lyapunov's indirect method), and its value gives the decay rate.
   For a human-only ring rho(J) > 1 reproduces the well-known string instability.
   We also verify that the shield is *inactive* in a neighbourhood of z* (barrier
   margin h(z*) > 0), so the certificate applies to the shielded system.

2. Statistical verification of STL specifications
   ------------------------------------------------------------------
   Exact Clopper-Pearson upper confidence bound on the probability that an episode
   violates a specification, from n i.i.d. evaluation episodes with k violations.
"""
from __future__ import annotations

import copy
import numpy as np
from scipy.stats import beta

from .env import RingEnv


def clopper_pearson_upper(k: int, n: int, conf: float = 0.95) -> float:
    if k >= n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


class _DetSystem:
    """Deterministic, single-environment copy of the closed loop."""

    def __init__(self, cfg, agent=None, shield=None):
        c = copy.deepcopy(cfg)
        c.accel_noise = 0.0; c.idm_spread = 0.0; c.perturbations = []
        self.env = RingEnv(c, n_envs=1)
        self.env.reset()
        for k in ("v0", "T", "a", "b", "s0"):
            self.env.params[k][:] = getattr(c.idm, k)
        self.agent, self.shield = agent, shield
        if shield is not None:
            shield.adaptive_backup = shield.adaptive
            shield.adaptive = False          # freeze conformal level for the linearisation
            shield.alpha_t = None
        self.env.shield = shield
        self.N, self.K = self.env.N, self.env.K

    def set_state(self, z):
        N, K = self.N, self.K
        self.env.x[0] = z[:N]; self.env.v[0] = z[N:2 * N]
        self.env.a_prev[0] = 0.0
        if K:
            self.env.a_prev[0, self.env.cav_idx] = z[2 * N:2 * N + K]
        self.env.t[:] = 0

    def get_state(self):
        e = self.env
        return np.concatenate([e.x[0], e.v[0], e.a_prev[0, e.cav_idx]])

    def step(self, z):
        self.set_state(z)
        e = self.env
        if self.K and self.agent is not None:
            a, _ = self.agent.act(e.obs(), deterministic=True)
            act = a[..., 0] * self.agent.act_scale
            e.step(act)
        elif self.K:
            # CAVs drive as noise-free IDM (human-only reference)
            from .env import idm_accel
            p = e.cfg.idm
            ctrl = lambda en: idm_accel(en.v, en.v[:, en.lead], en.gaps(), p.v0, p.T, p.a, p.b, p.s0,
                                        p.delta)[:, en.cav_idx]
            e.step(np.zeros((1, self.K)), controller=ctrl)
        else:
            e.step(np.zeros((1, 0)))
        return self.get_state()


def spectral_certificate(cfg, agent=None, shield=None, settle_steps=2000, eps=1e-4):
    """Return dict(rho, eigs, eq_speed, eq_gaps, shield_margin, converged)."""
    sysm = _DetSystem(cfg, agent, shield)
    N = sysm.N
    # start from (almost) uniform flow at the IDM equilibrium speed for this density
    L = cfg.ring_length; s_eq = L / N - cfg.veh_length
    p = cfg.idm
    vs = np.linspace(0.01, p.v0 - 0.01, 4000)
    s_star = (p.s0 + vs * p.T) / np.sqrt(np.maximum(1 - (vs / p.v0) ** p.delta, 1e-9))
    v_eq = float(vs[np.argmin(np.abs(s_star - s_eq))])
    z = np.concatenate([np.arange(N) * (L / N), np.full(N, v_eq), np.zeros(sysm.K)])
    # settle the noise-free closed loop (the equilibrium of the CAV policy may differ from IDM's).
    # With IDM-only vehicles the uniform flow is the exact equilibrium, so no settling is needed
    # (and settling an unstable ring would only produce waves).
    if agent is not None and sysm.K:
        for _ in range(settle_steps):
            z = sysm.step(z)
    z_prev = z.copy(); z = sysm.step(z)
    drift = z.copy(); drift[:N] -= z_prev[:N]                     # positions advance by v*dT
    dT = cfg.dt * cfg.substeps
    converged = bool(np.max(np.abs(z[N:2 * N] - z_prev[N:2 * N])) < 1e-3)
    # linearise Phi at z (positions taken relative to the moving frame)
    n = len(z)
    J = np.zeros((n, n))
    for j in range(n):
        dz = np.zeros(n); dz[j] = eps
        fp = sysm.step(z + dz); fm = sysm.step(z - dz)
        J[:, j] = (fp - fm) / (2 * eps)
    eigs = np.linalg.eigvals(J)
    # remove the translation mode (eigenvalue 1 with eigenvector = ones on positions)
    trans = np.zeros(n); trans[:N] = 1.0
    jt = J @ trans
    # should be ~0; large values mean Phi is not smooth at z* (e.g. shield active)
    trans_err = float(np.max(np.abs(jt - trans)))
    k_tr = np.argmin(np.abs(eigs - 1.0))
    eigs_red = np.delete(eigs, k_tr)
    rho = float(np.max(np.abs(eigs_red)))
    # shield margin at the equilibrium
    e = sysm.env
    sysm.set_state(z)
    g = e.gaps()[0]
    margin = None
    if shield is not None and sysm.K:
        i = e.cav_idx
        margin = float(np.min(shield.barrier(g[i], e.v[0, i], e.v[0, e.lead[i]], shield.tau)))
        shield.adaptive = shield.adaptive_backup
    return {"rho": rho, "eigs": eigs_red, "eq_speed": float(e.v[0].mean()),
            "eq_gap_cav": (g[e.cav_idx].tolist() if sysm.K else []),
            "eq_gap_hdv": float(np.delete(g, e.cav_idx).mean()) if sysm.K else float(g.mean()),
            "shield_margin": margin, "converged": converged, "dT": dT, "translation_err": trans_err}
