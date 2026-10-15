"""Vectorised ring-road mixed-autonomy traffic simulator.

N vehicles drive on a single-lane ring of length L (Sugiyama et al., 2008 set-up).
Human-driven vehicles (HDVs) follow the Intelligent Driver Model (IDM) with
per-vehicle parameters and additive Gaussian acceleration noise, which makes the
uniform flow unstable and produces stop-and-go waves.  K of the N vehicles are
connected autonomous vehicles (CAVs) whose longitudinal acceleration is chosen by
a controller (an RL policy, a baseline, or IDM for the human-only reference).

Everything is vectorised over E independent environments so that PPO can collect
experience quickly in pure NumPy.  Vehicle i follows vehicle (i+1) mod N.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class IDMParams:
    v0: float = 30.0      # desired speed [m/s]
    T: float = 1.0        # desired time headway [s]
    a: float = 1.0        # max acceleration [m/s^2]
    b: float = 1.5        # comfortable deceleration [m/s^2]
    s0: float = 2.0       # jam distance [m]
    delta: float = 4.0    # acceleration exponent


@dataclass
class EnvConfig:
    n_vehicles: int = 22
    ring_length: float = 260.0
    veh_length: float = 5.0
    n_cav: int = 2
    dt: float = 0.1                 # simulation step [s]
    substeps: int = 5               # sim steps per control step (control at 2 Hz)
    horizon: int = 200              # control steps per episode (100 s)
    warmup: float = 60.0            # human-only warm-up [s] so waves exist at t=0
    idm: IDMParams = field(default_factory=IDMParams)
    idm_spread: float = 0.10        # relative heterogeneity of HDV parameters
    accel_noise: float = 0.3        # std of HDV acceleration noise [m/s^2]
    a_max_phys: float = 3.0         # physical accel limit (all vehicles)
    b_max_phys: float = 7.0         # physical braking limit (all vehicles)
    a_rl_min: float = -3.0          # comfort range of the RL action
    a_rl_max: float = 2.0
    # perturbation: list of (time_s, vehicle_index_offset_from_cav0, decel, duration_s)
    perturbations: list = field(default_factory=list)
    # reward weights
    w_wave: float = 1.0
    w_speed: float = 0.1
    w_accel: float = 0.05
    w_headway: float = 0.0          # heuristic headway penalty (PPO baseline)
    w_stl: float = 0.0              # weight of past-time STL robustness reward (CertiFlow)
    w_interv: float = 0.0           # penalty on |a_rl - a_shield| (shield-aware learning)
    stl_d_min: float = 1.0
    stl_ttc: float = 1.5
    collision_penalty: float = 50.0
    seed: int = 0
    # domain randomisation of the HUMAN model (used for the *-DR policies)
    dr: bool = False
    dr_delay_max: float = 0.8       # human reaction delay ~ U[0, max] s, per ring
    dr_gap_noise: float = 0.10      # relative perception error on gap (AR(1), per driver)
    dr_speed_noise: float = 0.2     # perception error on leader speed [m/s]
    dr_spread: float = 0.30         # IDM parameter heterogeneity
    dr_accel_noise: tuple = (0.2, 0.6)
    dr_ring_range: tuple = (260.0, 800.0)   # density randomisation: equilibrium speeds ~4-22 m/s


def idm_accel(v, v_lead, gap, p_v0, p_T, p_a, p_b, p_s0, delta=4.0):
    """Vectorised IDM acceleration (Treiber et al., 2000)."""
    dv = v - v_lead
    s_star = p_s0 + np.maximum(0.0, v * p_T + v * dv / (2.0 * np.sqrt(p_a * p_b)))
    gap_safe = np.maximum(gap, 0.1)
    return p_a * (1.0 - (v / p_v0) ** delta - (s_star / gap_safe) ** 2)


class RingEnv:
    """E parallel ring roads.  Arrays have shape (E, N)."""

    OBS_DIM = 10

    def __init__(self, cfg: EnvConfig, n_envs: int = 8):
        self.cfg = cfg
        self.E = n_envs
        self.N = cfg.n_vehicles
        self.K = cfg.n_cav
        self.rng = np.random.default_rng(cfg.seed)
        # CAVs spread evenly around the ring
        self.cav_idx = (np.arange(self.K) * self.N // max(self.K, 1)).astype(int) if self.K > 0 \
            else np.zeros(0, dtype=int)
        self.is_cav = np.zeros(self.N, dtype=bool)
        self.is_cav[self.cav_idx] = True
        self.lead = (np.arange(self.N) + 1) % self.N
        self.foll = (np.arange(self.N) - 1) % self.N
        self.shield = None              # optional safety shield object
        self.record = False
        self._alloc()

    # ------------------------------------------------------------------ #
    def _alloc(self):
        E, N = self.E, self.N
        self.x = np.zeros((E, N))
        self.v = np.zeros((E, N))
        self.a_prev = np.zeros((E, N))
        self.t = np.zeros(E, dtype=int)
        self.params = {}
        self.crashed = np.zeros(E, dtype=bool)
        self.H = int(round(self.cfg.dr_delay_max / self.cfg.dt)) + 1
        self.hist_x = np.zeros((self.H, E, N)); self.hist_v = np.zeros((self.H, E, N))
        self.ptr = np.zeros(E, dtype=int); self.delay = np.zeros(E, dtype=int)
        self.e_gap = np.zeros((E, N)); self.e_v = np.zeros((E, N))
        self.noise_env = np.full(E, self.cfg.accel_noise)
        self.L = np.full(E, float(self.cfg.ring_length))

    def _sample_params(self, idx):
        c, n = self.cfg, len(idx)
        sp = c.dr_spread if c.dr else c.idm_spread
        if c.dr:
            self.delay[idx] = self.rng.integers(0, self.H, size=n)
            self.noise_env[idx] = self.rng.uniform(*c.dr_accel_noise, size=n)
            self.e_gap[idx] = 0.0; self.e_v[idx] = 0.0
        def draw(base):
            return base * (1.0 + sp * self.rng.uniform(-1, 1, size=(n, self.N)))
        for k, base in (("v0", c.idm.v0), ("T", c.idm.T), ("a", c.idm.a),
                        ("b", c.idm.b), ("s0", c.idm.s0)):
            if k not in self.params:
                self.params[k] = np.zeros((self.E, self.N))
            self.params[k][idx] = draw(base)

    def gaps(self, x=None):
        x = self.x if x is None else x
        return np.mod(x[:, self.lead] - x, self.L[:, None]) - self.cfg.veh_length

    # ------------------------------------------------------------------ #
    def reset(self, env_ids=None):
        c = self.cfg
        ids = np.arange(self.E) if env_ids is None else np.asarray(env_ids)
        if len(ids) == 0:
            return self.obs()
        if c.dr:
            self.L[ids] = self.rng.uniform(*c.dr_ring_range, size=len(ids))
        spacing = self.L[ids, None] / self.N
        self.x[ids] = np.arange(self.N)[None] * spacing + self.rng.normal(0, 0.5, (len(ids), self.N))
        self.v[ids] = 0.0
        self.a_prev[ids] = 0.0
        self.t[ids] = 0
        self.crashed[ids] = False
        self._sample_params(ids)
        for h in range(self.H):
            self.hist_x[h, ids] = self.x[ids]; self.hist_v[h, ids] = self.v[ids]
        # human-only warm-up (CAVs also drive as IDM) so that waves are present
        n_warm = int(c.warmup / c.dt)
        for _ in range(n_warm):
            a = self._hdv_accel()[ids]
            self._integrate(a, ids)
        if self.shield is not None:
            self.shield.reset(ids)
        return self.obs()

    # ------------------------------------------------------------------ #
    def _hdv_accel(self, noise=True):
        p = self.params
        c = self.cfg
        if c.dr:
            # delayed, noisy perception of the leader (EIDM-like reaction time + estimation error)
            e = np.arange(self.E)
            k = (self.ptr - self.delay) % self.H
            xd, vd = self.hist_x[k, e], self.hist_v[k, e]
            g = self.gaps(xd)
            rho = 0.95
            self.e_gap = rho * self.e_gap + np.sqrt(1 - rho ** 2) * c.dr_gap_noise * self.rng.standard_normal(g.shape)
            self.e_v = rho * self.e_v + np.sqrt(1 - rho ** 2) * c.dr_speed_noise * self.rng.standard_normal(g.shape)
            g = g * (1.0 + self.e_gap)
            vl = vd[:, self.lead] + self.e_v
            a = idm_accel(self.v, vl, g, p["v0"], p["T"], p["a"], p["b"], p["s0"], c.idm.delta)
            sig = self.noise_env[:, None]
        else:
            g = self.gaps()
            a = idm_accel(self.v, self.v[:, self.lead], g, p["v0"], p["T"], p["a"], p["b"], p["s0"],
                          c.idm.delta)
            sig = c.accel_noise
        if noise and c.accel_noise > 0:
            a = a + sig * self.rng.standard_normal(a.shape)
        # a human never ignores an imminent collision: true-gap emergency braking
        if c.dr:
            gt = self.gaps(); vt = self.v[:, self.lead]
            need = (self.v ** 2 - vt ** 2) / (2 * np.maximum(gt - 1.0, 0.05))
            a = np.where((need > 1.5) & (self.v > vt), np.minimum(a, -np.minimum(1.2 * need, c.b_max_phys)), a)
        return np.clip(a, -self.cfg.b_max_phys, self.cfg.a_max_phys)

    def _integrate(self, a, ids):
        dt = self.cfg.dt
        v_old = self.v[ids]
        v_new = np.maximum(0.0, v_old + a * dt)
        # exact displacement when the vehicle stops inside the step
        self.x[ids] = self.x[ids] + 0.5 * (v_old + v_new) * dt
        self.a_prev[ids] = (v_new - v_old) / dt
        self.v[ids] = v_new
        self.ptr[ids] = (self.ptr[ids] + 1) % self.H
        self.hist_x[self.ptr[ids], ids] = self.x[ids]
        self.hist_v[self.ptr[ids], ids] = self.v[ids]

    # ------------------------------------------------------------------ #
    def obs(self):
        """Local + V2V observation for each CAV, shape (E, K, OBS_DIM)."""
        g = self.gaps()
        i = self.cav_idx
        L1, F1 = self.lead[i], self.foll[i]
        L2 = self.lead[L1]
        v = self.v
        o = np.stack([
            v[:, i] / 20.0,
            v[:, L1] / 20.0,
            (v[:, L1] - v[:, i]) / 5.0,
            g[:, i] / 30.0,
            v[:, F1] / 20.0,
            g[:, F1] / 30.0,
            self.a_prev[:, i] / 3.0,
            v[:, L2] / 20.0,
            g[:, L1] / 30.0,
            v.mean(axis=1, keepdims=True).repeat(len(i), 1) / 20.0,   # broadcast via V2I
        ], axis=-1)
        return o.astype(np.float64)

    # ------------------------------------------------------------------ #
    def _perturbation_accel(self, a):
        c = self.cfg
        if not c.perturbations:
            return a
        time_s = self.t * c.dt   # per env
        for (t0, off, dec, dur) in c.perturbations:
            j = (self.cav_idx[0] + off) % self.N if self.K > 0 else off % self.N
            active = (time_s >= t0) & (time_s < t0 + dur)
            a[active, j] = -abs(dec)
        return a

    def step(self, act_rl, controller=None):
        """Advance one control step.

        act_rl     : (E, K) desired CAV accelerations from the policy (ignored if controller given)
        controller : optional callable(env)->(E,K) used at every sub-step (baselines)
        Returns obs, reward (E,K), done (E,), info dict.
        """
        c = self.cfg
        E, K = self.E, self.K
        rew = np.zeros(E)
        rew_loc = np.zeros((E, K))
        rho_min = np.full((E, K), np.inf)    # past-time robustness of G_[t-dT, t](mu_local)
        info = {"interv": np.zeros((E, K)), "a_rl": np.zeros((E, K)), "a_exec": np.zeros((E, K)),
                "collision": np.zeros(E, dtype=bool), "infeasible": np.zeros((E, K))}
        all_ids = np.arange(E)
        for _ in range(c.substeps):
            a = self._hdv_accel()
            if K > 0:
                if controller is not None:
                    a_cav = controller(self)
                else:
                    a_cav = np.clip(act_rl, c.a_rl_min, c.a_rl_max)
                a_des = a_cav.copy()
                if self.shield is not None:
                    a_cav, inf = self.shield.filter(self, a_cav)
                    info["infeasible"] += inf
                info["interv"] += (np.abs(a_cav - a_des) > 1e-6)
                info["a_rl"] += a_des / c.substeps
                info["a_exec"] += a_cav / c.substeps
                a[:, self.cav_idx] = np.clip(a_cav, -c.b_max_phys, c.a_max_phys)
            a = self._perturbation_accel(a)
            v_before = self.v.copy()
            self._integrate(a, all_ids)
            if self.shield is not None:
                self.shield.update(self, v_before)
            g = self.gaps()
            coll = (g < 0.0).any(axis=1)
            info["collision"] |= coll
            if self.record:
                self._rec_x.append(self.x.copy()); self._rec_v.append(self.v.copy())
                self._rec_a.append(a.copy())
            # reward (shared, averaged over sub-steps)
            std_v = self.v.std(axis=1)
            rew += (-c.w_wave * std_v + c.w_speed * self.v.mean(axis=1)) / c.substeps
            if K > 0:
                rew += (-c.w_accel * (a[:, self.cav_idx] ** 2).mean(axis=1)) / c.substeps
                if c.w_headway > 0:
                    gi = g[:, self.cav_idx]; vi = self.v[:, self.cav_idx]
                    viol = np.maximum(0.0, (2.0 + 0.8 * vi) - gi) / 5.0
                    rew_loc -= c.w_headway * viol / c.substeps
                if c.w_stl > 0:
                    # local predicates for the CAV and its human follower
                    i, F = self.cav_idx, self.foll[self.cav_idx]
                    vv = self.v
                    clo_i = np.maximum(0.0, vv[:, i] - vv[:, self.lead[i]])
                    clo_f = np.maximum(0.0, vv[:, F] - vv[:, i])
                    rho = np.minimum.reduce([g[:, i] - c.stl_d_min, g[:, i] - c.stl_ttc * clo_i,
                                             g[:, F] - c.stl_d_min, g[:, F] - c.stl_ttc * clo_f])
                    rho_min = np.minimum(rho_min, rho)
                if c.w_interv > 0 and self.shield is not None:
                    rew_loc -= c.w_interv * np.abs(a_cav - a_des) / c.substeps
            self.t += 1
        if c.w_stl > 0 and K > 0:
            rew_loc += c.w_stl * np.clip(rho_min / 5.0, -1.0, 0.2)
        rew = rew - c.collision_penalty * info["collision"]
        self.crashed |= info["collision"]
        done = info["collision"] | (self.t >= c.horizon * c.substeps)
        info["interv"] /= c.substeps
        return self.obs(), rew[:, None] + rew_loc, done, info

    # ------------------------------------------------------------------ #
    def start_recording(self):
        self.record = True
        self._rec_x, self._rec_v, self._rec_a = [], [], []

    def recording(self):
        return np.array(self._rec_x), np.array(self._rec_v), np.array(self._rec_a)
