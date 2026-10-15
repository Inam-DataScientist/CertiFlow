"""Zero-shot sim-to-sim transfer of CertiFlow to SUMO (Eclipse SUMO + TraCI).

Policies trained in the NumPy ring are deployed in SUMO, where human drivers follow SUMO's
Extended IDM (EIDM) with 0.8 s reaction time and perception errors instead of noisy IDM -- a
deliberate model mismatch.  Before deployment the shield selects its human-acceleration
predictor (nominal IDM or persistence) on one human-only calibration run and computes the
conformal quantile on a second, disjoint run (see select_and_calibrate).

Scenarios
  ring      : Sugiyama ring, 22 vehicles, 260 m (nominal and emergency braking)
  highway   : 2-lane 3 km freeway with an on-ramp merge at 1.2 km, open boundary,
              humans change lanes (LC2013), CAVs keep their lane

Usage:  python sumo_eval.py --scenario ring --episodes 10
        python sumo_eval.py --scenario highway --episodes 3 --methods "..."
SUMO is located through $SUMO_HOME or the eclipse-sumo Python package.
"""
from __future__ import annotations

import argparse, os, pickle, subprocess, sys
import numpy as np
import pandas as pd


def _sumo_home():
    if os.environ.get("SUMO_HOME"):
        return os.environ["SUMO_HOME"]
    try:
        import sumo
        return sumo.SUMO_HOME
    except ImportError:
        sys.exit("SUMO not found: set SUMO_HOME or install it with `pip install eclipse-sumo traci`")


SUMO_HOME = _sumo_home()
sys.path.insert(0, os.path.join(SUMO_HOME, "tools"))
import traci  # noqa: E402

from certiflow.env import EnvConfig, idm_accel  # noqa: E402
from certiflow.ppo import PPOAgent  # noqa: E402
from certiflow.stl import traffic_specs  # noqa: E402

SUMO_BIN = os.path.join(SUMO_HOME, "bin", "sumo")
NETCONVERT = os.path.join(SUMO_HOME, "bin", "netconvert")
DT = 0.1
B = 7.0          # physical braking limit used by the barrier [m/s^2]
A_MAX = 3.0
IDM = EnvConfig().idm


# --------------------------------------------------------------------------- #
# Network / route generation
# --------------------------------------------------------------------------- #
VTYPES = """
  <vType id="human" vClass="passenger" length="5" minGap="2" accel="1.0" decel="1.5" emergencyDecel="7"
         tau="1.0" maxSpeed="30" carFollowModel="EIDM" sigmaerror="0.15" sigmagap="0.15" sigmaleader="0.15"
         treaction="0.8" speedDev="0.1" laneChangeModel="LC2013"/>
  <vType id="cav" vClass="passenger" length="5" minGap="2" accel="1.0" decel="1.5" emergencyDecel="7"
         sigma="0" tau="1.0" maxSpeed="30" color="0,0.4,1"/>
"""


def build_ring(d, L=260.0):
    R = L / (2 * np.pi)
    th1 = np.linspace(0, np.pi, 40)[1:-1]; th2 = np.linspace(np.pi, 2 * np.pi, 40)[1:-1]
    sh = lambda th: " ".join(f"{R*np.cos(t):.3f},{R*np.sin(t):.3f}" for t in th)
    open(f"{d}/ring.nod.xml", "w").write(
        f'<nodes><node id="A" x="{R:.3f}" y="0" type="priority"/><node id="B" x="{-R:.3f}" y="0" type="priority"/></nodes>')
    open(f"{d}/ring.edg.xml", "w").write(
        f'<edges><edge id="top" from="A" to="B" numLanes="1" speed="30" shape="{R:.3f},0 {sh(th1)} {-R:.3f},0"/>'
        f'<edge id="bot" from="B" to="A" numLanes="1" speed="30" shape="{-R:.3f},0 {sh(th2)} {R:.3f},0"/></edges>')
    subprocess.run([NETCONVERT, "-n", f"{d}/ring.nod.xml", "-e", f"{d}/ring.edg.xml", "-o", f"{d}/ring.net.xml",
                    "--no-internal-links", "true", "--no-turnarounds", "true", "--junctions.corner-detail", "0",
                    "--geometry.min-radius.fix", "false"], check=True, capture_output=True)
    return f"{d}/ring.net.xml"


def build_highway(d):
    open(f"{d}/hw.nod.xml", "w").write(
        '<nodes><node id="n0" x="0" y="0"/><node id="n1" x="1200" y="0"/><node id="n2" x="1400" y="0"/>'
        '<node id="n3" x="3000" y="0"/><node id="r0" x="800" y="-150"/></nodes>')
    open(f"{d}/hw.edg.xml", "w").write(
        '<edges><edge id="up" from="n0" to="n1" numLanes="2" speed="30"/>'
        '<edge id="merge" from="n1" to="n2" numLanes="3" speed="30"/>'
        '<edge id="down" from="n2" to="n3" numLanes="2" speed="30"/>'
        '<edge id="ramp" from="r0" to="n1" numLanes="1" speed="20"/></edges>')
    open(f"{d}/hw.con.xml", "w").write(
        '<connections><connection from="up" to="merge" fromLane="0" toLane="1"/>'
        '<connection from="up" to="merge" fromLane="1" toLane="2"/>'
        '<connection from="ramp" to="merge" fromLane="0" toLane="0"/>'
        '<connection from="merge" to="down" fromLane="1" toLane="0"/>'
        '<connection from="merge" to="down" fromLane="2" toLane="1"/></connections>')
    subprocess.run([NETCONVERT, "-n", f"{d}/hw.nod.xml", "-e", f"{d}/hw.edg.xml", "-x", f"{d}/hw.con.xml",
                    "-o", f"{d}/hw.net.xml"], check=True, capture_output=True)
    return f"{d}/hw.net.xml"


# --------------------------------------------------------------------------- #
# Shield (same barrier, conformal bounds and bisection as certiflow/shield.py)
# --------------------------------------------------------------------------- #
class Shield:
    def __init__(self, scores_low, scores_up, d0=2.0, tau=0.3, tau_f=0.3, gamma=0.3, alpha=0.01,
                 conformal=True, cooperative=True, adaptive=True, lr=0.005, predictor="idm"):
        self.predictor = predictor
        self.sl, self.su = np.sort(scores_low), np.sort(scores_up)
        self.d0, self.tau, self.tau_f, self.gamma, self.alpha = d0, tau, tau_f, gamma, alpha
        self.conformal, self.cooperative, self.adaptive, self.lr = conformal, cooperative, adaptive, lr
        self.alpha_l, self.alpha_f = {}, {}
        self.miscov = 0; self.n = 0

    def q(self, scores, a):
        n = len(scores); k = int(np.ceil((n + 1) * (1 - max(a, 1e-9)))) - 1
        return np.inf if (a <= 0 or k >= n) else scores[max(k, 0)]

    def h(self, s, v, vl, tau):
        return s - self.d0 - tau * v - (v ** 2 - vl ** 2) / (2 * B)

    def h_next(self, a, s, v, vl, al, tau):
        vl2 = max(0.0, vl + al * DT); v2 = np.maximum(0.0, v + a * DT)
        s2 = s + 0.5 * (vl + vl2) * DT - 0.5 * (v + v2) * DT
        return self.h(s2, v2, vl2, tau)

    def _upper(self, s, v, vl, al_lb):
        rhs = (1 - self.gamma) * self.h(s, v, vl, self.tau)
        if self.h_next(A_MAX, s, v, vl, al_lb, self.tau) >= rhs:
            return A_MAX
        lo, hi = -B, A_MAX
        for _ in range(25):
            mid = 0.5 * (lo + hi)
            if self.h_next(mid, s, v, vl, al_lb, self.tau) >= rhs: lo = mid
            else: hi = mid
        return lo

    def filter(self, cid, a_des, s, v, vl, al_hat, lead_is_cav, sf=None, vf=None, af_hat=None, extra=()):
        """Returns (a_exec, infeasible, predicted bounds for the ACI update).

        extra: further leader candidates (s, vl, al_hat) -- vehicles cutting into the lane.
        They are bounded worst-case (al = -B): their future in our lane has no calibration data."""
        al = self.alpha_l.setdefault(cid, self.alpha); af = self.alpha_f.setdefault(cid, self.alpha)
        if s is None:                                    # free road: no leader in range
            al_lb = None; a_up = A_MAX
        else:
            q_l = self.q(self.sl, al if self.adaptive else self.alpha) if self.conformal else 0.0
            al_lb = -B if (lead_is_cav or not np.isfinite(q_l)) else max(al_hat - q_l, -B)
            a_up = self._upper(s, v, vl, al_lb)
        for (s2, vl2, _) in extra:
            a_up = min(a_up, self._upper(s2, v, vl2, -B))
        a_lo = -B; infeasible = False; af_ub = None
        if self.cooperative and sf is not None:
            q_f = self.q(self.su, af if self.adaptive else self.alpha) if self.conformal else 0.0
            af_ub = A_MAX if not np.isfinite(q_f) else float(np.clip(af_hat + q_f, -B, A_MAX))
            rhs_f = (1 - self.gamma) * self.h(sf, vf, v, self.tau_f)
            f = lambda a: self.h_next(af_ub, sf, vf, v, a, self.tau_f)
            if f(-B) < rhs_f:
                lo, hi = -B, A_MAX
                for _ in range(25):
                    mid = 0.5 * (lo + hi)
                    if f(mid) >= rhs_f: hi = mid
                    else: lo = mid
                a_lo = hi
            if a_lo > a_up:
                infeasible = True; a_lo = a_up
        return float(np.clip(a_des, a_lo, a_up)), infeasible, (al_lb, af_ub)

    def update(self, cid, bounds, al_true, af_true):
        if not (self.conformal and self.adaptive):
            return
        al_lb, af_ub = bounds
        if al_lb is not None and al_true is not None:
            err = float(al_true < al_lb - 1e-9); self.miscov += err; self.n += 1
            self.alpha_l[cid] = float(np.clip(self.alpha_l[cid] + self.lr * (self.alpha - err), -0.05, self.alpha))
        if af_ub is not None and af_true is not None:
            err = float(af_true > af_ub + 1e-9)
            self.alpha_f[cid] = float(np.clip(self.alpha_f[cid] + self.lr * (self.alpha - err), -0.05, self.alpha))


def idm_hat(v, vl, s):
    return float(idm_accel(np.array(v), np.array(vl), np.array(s), IDM.v0, IDM.T, IDM.a, IDM.b, IDM.s0, IDM.delta))


class PISat:
    """Functional PI-with-saturation (Stern et al., 2018), one state per CAV."""
    def __init__(self):
        self.hist, self.vcmd = {}, {}

    def __call__(self, cid, v, vl, gap):
        h = self.hist.setdefault(cid, []); h.append(v); del h[:-380]
        vc = self.vcmd.setdefault(cid, v)
        U = float(np.mean(h)); vt = U + float(np.clip((gap - 7) / 23, 0, 1))
        alpha = float(np.clip((gap - max(2 * (vl - v), 4)) / 2, 0, 1)); beta = 1 - 0.5 * alpha
        vc = beta * (alpha * vt + (1 - alpha) * vl) + (1 - beta) * vc; self.vcmd[cid] = vc
        return float(np.clip((vc - v) / DT, -3, 2))


# --------------------------------------------------------------------------- #
# Ring scenario
# --------------------------------------------------------------------------- #
class RingSUMO:
    def __init__(self, workdir, n=22, k=2, seed=0, port=None, label="r"):
        self.n, self.k, self.seed, self.d, self.label = n, k, seed, workdir, label
        self.net = os.path.join(workdir, "ring.net.xml")
        if not os.path.exists(self.net):
            build_ring(workdir)
        self.cav_ids = [f"v{i}" for i in (np.arange(k) * n // k)] if k else []

    def start(self):
        routes = os.path.join(self.d, f"ring_{self.label}.rou.xml")
        loop = " ".join(["top bot"] * 60)
        spacing = 260.0 / self.n
        veh = []
        for i in range(self.n):
            pos = i * spacing
            edge = "top" if pos < 130 else "bot"
            p = pos if pos < 130 else pos - 130
            t = "cav" if f"v{i}" in self.cav_ids else "human"
            # CAVs drive as humans during warm-up: they get the human type and are switched later
            veh.append(f'<vehicle id="v{i}" type="human" depart="0" departPos="{p:.2f}" departSpeed="0">'
                       f'<route edges="{loop if edge=="top" else "bot " + loop}"/></vehicle>')
        open(routes, "w").write(f"<routes>{VTYPES}{''.join(veh)}</routes>")
        traci.start([SUMO_BIN, "-n", self.net, "-r", routes, "--step-length", str(DT), "--seed", str(self.seed),
                     "--no-step-log", "true", "--no-warnings", "true", "--collision.action", "warn",
                     "--collision.check-junctions", "true", "--time-to-teleport", "-1",
                     ], label=self.label)
        self.c = traci.getConnection(self.label)
        self.c.simulationStep()
        self.ids = [f"v{i}" for i in range(self.n)]
        self.Lring = self.c.lane.getLength("top_0") + self.c.lane.getLength("bot_0")

    def state(self):
        c = self.c
        d = np.array([c.vehicle.getDistance(v) for v in self.ids])
        v = np.array([c.vehicle.getSpeed(i) for i in self.ids])
        return d, v

    def gaps(self, d):
        # initial offsets: vehicle i departs at i*spacing; odometer starts at 0
        pos = d + np.arange(self.n) * (260.0 / self.n)
        lead = np.r_[1:self.n, 0]
        return np.mod(pos[lead] - pos, self.Lring) - 5.0, pos

    def close(self):
        self.c.close()


def calibrate_sumo_ring(workdir, seconds=300, seed=999):
    env = RingSUMO(workdir, k=0, seed=seed, label="cal"); env.start()
    for _ in range(int(60 / DT)):
        env.c.simulationStep()
    d, v = env.state()
    res = {"idm": [], "persist": []}
    a_last = np.zeros(env.n)
    for _ in range(int(seconds / DT)):
        g, _ = env.gaps(d)
        vl = v[np.r_[1:env.n, 0]]
        ahat = idm_accel(v, vl, g, IDM.v0, IDM.T, IDM.a, IDM.b, IDM.s0, IDM.delta)
        env.c.simulationStep()
        d2, v2 = env.state()
        a = (v2 - v) / DT
        m = v2 > 0
        res["idm"].append((ahat - a)[m]); res["persist"].append((a_last - a)[m])
        a_last = a
        d, v = d2, v2
    env.close()
    return {k: np.concatenate(v) for k, v in res.items()}


def select_and_calibrate(cal_fn, work, alpha=0.01):
    """Pick the predictor with the tighter two-sided 1-alpha quantile on calibration run A,
    then calibrate its scores on a DISJOINT run B (keeps split-conformal validity)."""
    A = cal_fn(work, seed=901); Bc = cal_fn(work, seed=902)
    width = {k: np.quantile(r, 1 - alpha) + np.quantile(-r, 1 - alpha) for k, r in A.items()}
    best = min(width, key=width.get)
    r = Bc[best]
    return best, r, -r, width


def run_ring_episode(workdir, method, agent=None, shield=None, seed=0, emergency=False, label="r", trace=None):
    env = RingSUMO(workdir, k=2, seed=seed, label=label); env.start(); c = env.c
    for _ in range(int(60 / DT)):                     # human-only warm-up (waves form)
        c.simulationStep()
    cav_idx = [int(x[1:]) for x in env.cav_ids]
    controlled = method != "Human"
    if controlled:
        for cid in env.cav_ids:
            c.vehicle.setType(cid, "cav"); c.vehicle.setSpeedMode(cid, 0)
    lead_of = lambda i: (i + 1) % env.n
    foll_of = lambda i: (i - 1) % env.n
    pi = PISat()
    T = int(100 / DT)
    V, A, G, interv, infeas = [], [], [], 0, 0
    act = np.zeros(len(cav_idx)); a_prev = np.zeros(env.n)
    d, v = env.state()
    em_id = f"v{lead_of(cav_idx[0])}"
    for t in range(T):
        g, pos = env.gaps(d)
        if emergency:
            tt = t * DT
            if 20 <= tt < 23 or 55 <= tt < 58:
                vid = em_id if tt < 40 else f"v{(cav_idx[0] + 3) % env.n}"
                c.vehicle.setSpeedMode(vid, 0); c.vehicle.setSpeed(vid, max(0.0, c.vehicle.getSpeed(vid) - 6.0 * DT))
            elif 23 <= tt < 23 + DT or 58 <= tt < 58 + DT:
                for vid in (em_id, f"v{(cav_idx[0] + 3) % env.n}"):
                    c.vehicle.setSpeedMode(vid, 31); c.vehicle.setSpeed(vid, -1)
        bounds = {}
        if controlled:
            if t % 5 == 0 and agent is not None:      # policy at 2 Hz, held for 0.5 s
                obs = []
                for i in cav_idx:
                    L1, F1 = lead_of(i), foll_of(i); L2 = lead_of(L1)
                    obs.append([v[i] / 20, v[L1] / 20, (v[L1] - v[i]) / 5, g[i] / 30, v[F1] / 20, g[F1] / 30,
                                a_prev[i] / 3, v[L2] / 20, g[L1] / 30, v.mean() / 20])
                a, _ = agent.act(np.array(obs)[None], deterministic=True)
                act = np.clip(a[0, :, 0] * agent.act_scale, -3, 2)
            for j, i in enumerate(cav_idx):
                L1, F1 = lead_of(i), foll_of(i)
                a_des = act[j] if agent is not None else pi(i, v[i], v[L1], g[i])
                a_ex = a_des
                if shield is not None:
                    if shield.predictor == "persist":
                        al_hat, af_hat = float(a_prev[L1]), float(a_prev[F1])
                    else:
                        al_hat = idm_hat(v[L1], v[lead_of(L1)], g[L1])
                        af_hat = idm_hat(v[F1], v[i], g[F1])
                    a_ex, inf, bounds[i] = shield.filter(i, a_des, g[i], v[i], v[L1], al_hat, False, g[F1], v[F1], af_hat)
                    interv += abs(a_ex - a_des) > 1e-6; infeas += inf
                c.vehicle.setSpeed(env.ids[i], max(0.0, v[i] + a_ex * DT))
        c.simulationStep()
        d2, v2 = env.state()
        acc = (v2 - v) / DT
        if shield is not None:
            for i, bnd in bounds.items():
                shield.update(i, bnd, acc[lead_of(i)] if v2[lead_of(i)] > 0 else None, acc[foll_of(i)])
        a_prev = acc
        d, v = d2, v2
        g2, _ = env.gaps(d)
        V.append(v.copy()); A.append(acc.copy()); G.append(g2.copy())
        if trace is not None:
            trace.setdefault("pos", []).append(env.gaps(d)[1].copy())
    ncoll = c.simulation.getCollidingVehiclesNumber()
    env.close()
    V, A, G = np.array(V), np.array(A), np.array(G)
    if trace is not None:
        trace["v"] = np.array(V); trace["pos"] = np.array(trace["pos"]); trace["L"] = env.Lring
        trace["cav_idx"] = cav_idx
    return episode_metrics(V, A, G, cav_idx, interv / (T * max(1, len(cav_idx))), infeas)


def episode_metrics(V, A, G, cav_idx, interv, infeas):
    lead = np.r_[1:V.shape[1], 0]
    closing = np.maximum(0, V - V[:, lead])
    sig = {"gap_min": G.min(1)[None], "ttc_marg": (G - 1.5 * closing).min(1)[None], "std_v": V.std(1)[None],
           "a_cav_min": (A[:, cav_idx].min(1) if len(cav_idx) else np.full(len(V), 9.0))[None]}
    rob = {k: float(f.robustness(sig)[0]) for k, f in traffic_specs(DT).items()}
    half = len(V) // 2
    P = 1500 * V * (A + 9.81 * 0.01) + 0.5 * 1.2 * 0.7 * V ** 3
    return {"collision": bool(G.min() < 0), "min_gap": float(G.min()), "std_v_ss": float(V[half:].std(1).mean()),
            "mean_v": float(V.mean()), "energy_MJkm": float(np.maximum(P, 0).sum() * DT / max(V.sum() * DT, 1) * 1e-3),
            "interv_rate": interv, "infeasible_steps": infeas, **{f"rho_{k}": r for k, r in rob.items()}}


# --------------------------------------------------------------------------- #
def load_agent(method, seed, runs="runs"):
    ag = PPOAgent(10, 3, seed=seed)
    with open(os.path.join(runs, f"{method}_s{seed}", "policy.pkl"), "rb") as fh:
        ag.load_state_dict(pickle.load(fh))
    return ag


SHIELD_CFG = {"PPO": None, "PPO+CBF": dict(conformal=False, cooperative=False, adaptive=False),
              "CertiFlow": dict(), "PPO+CF-shield (post-hoc)": dict(),
              "PPO-DR": None, "CertiFlow-DR": dict(), "CertiFlow-DR-FT": dict()}
WEIGHTS = {"PPO": "PPO", "PPO+CBF": "PPO+CBF", "CertiFlow": "CertiFlow", "PPO+CF-shield (post-hoc)": "PPO",
           "PPO-DR": "PPO-DR", "CertiFlow-DR": "CertiFlow-DR", "CertiFlow-DR-FT": "CertiFlow-DR-FT"}


# --------------------------------------------------------------------------- #
# Highway with on-ramp merge
# --------------------------------------------------------------------------- #
def hw_routes(d, seed, cav_frac, q_main=3400, q_ramp=700, horizon=900, label="h"):
    """Mainline demand q_main veh/h over 2 lanes (cav_frac of it CAVs), ramp demand q_ramp veh/h."""
    p = os.path.join(d, f"hw_{label}.rou.xml")
    qc = q_main * cav_frac; qh = q_main - qc
    flows = (f'<route id="main" edges="up merge down"/><route id="onr" edges="ramp merge down"/>'
             f'<flow id="hm" type="human" route="main" begin="0" end="{horizon}" vehsPerHour="{qh:.0f}" departLane="random" departSpeed="desired"/>'
             f'<flow id="hr" type="human" route="onr" begin="0" end="{horizon}" vehsPerHour="{q_ramp:.0f}" departLane="0" departSpeed="desired"/>')
    if qc > 0:
        flows += (f'<flow id="cm" type="cav" route="main" begin="0" end="{horizon}" vehsPerHour="{qc:.0f}" '
                  f'departLane="random" departSpeed="desired"/>')
    open(p, "w").write(f"<routes>{VTYPES}{flows}</routes>")
    return p


def _start_hw(work, routes, seed, label):
    net = os.path.join(work, "hw.net.xml")
    if not os.path.exists(net):
        build_highway(work)
    traci.start([SUMO_BIN, "-n", net, "-r", routes, "--step-length", str(DT), "--seed", str(seed),
                 "--no-step-log", "true", "--no-warnings", "true", "--collision.action", "remove",
                 "--collision.check-junctions", "true", "--time-to-teleport", "-1",
                 "--lanechange.duration", "2"], label=label)
    return traci.getConnection(label)


MINGAP = {"human": 2.0, "cav": 2.0}   # declared minGap (only shapes how OTHER drivers judge gaps)


def calibrate_sumo_hw(work, seconds=400, seed=998):
    """Residuals of the nominal IDM predictor for human car-following on the highway.

    The prediction is made BEFORE a step and compared with the acceleration realised
    DURING that step; pairs whose leader changed in the step (lane changes) are excluded,
    since the shield treats a new leader through the barrier state itself.
    """
    c = _start_hw(work, hw_routes(work, seed, 0.0, label=f"cal{seed}"), seed, f"hcal{seed}")
    res = {"idm": [], "persist": []}
    for t in range(int(seconds / DT)):
        pred = {}
        if t >= 1500 and t % 3 == 0:
            for vid in c.vehicle.getIDList()[::3]:
                ld = c.vehicle.getLeader(vid, 150.0)
                if ld and ld[0] != "":
                    v = c.vehicle.getSpeed(vid); vl = c.vehicle.getSpeed(ld[0])
                    pred[vid] = (idm_hat(v, vl, ld[1] + 2.0), c.vehicle.getAcceleration(vid), ld[0])
        c.simulationStep()
        if pred:
            alive = set(c.vehicle.getIDList())
            for vid, (a_idm, a_per, lid) in pred.items():
                if vid not in alive:
                    continue
                ld = c.vehicle.getLeader(vid, 150.0)
                if not ld or ld[0] != lid or c.vehicle.getSpeed(vid) <= 0:
                    continue
                a = c.vehicle.getAcceleration(vid)
                res["idm"].append(a_idm - a); res["persist"].append(a_per - a)
    c.close()
    return {k: np.array(v) for k, v in res.items()}   # score_low = r, score_up = -r


def run_hw_episode(work, method, agent=None, shield=None, seed=0, cav_frac=0.10, horizon=900, warm=300,
                   label="h"):
    controlled = method != "Human"
    routes = hw_routes(work, seed, cav_frac, horizon=horizon, label=label)
    if not controlled:   # CAV flow exists but its vehicles drive as humans
        txt = open(routes).read().replace('type="cav"', 'type="human"'); open(routes, "w").write(txt)
    c = _start_hw(work, routes, seed, label)
    pi = PISat()
    held, aprev, pending = {}, {}, {}
    n_coll, coll_ids = 0, set()
    coll_log = {}
    interv = n_ctrl = infeas = 0
    harsh_h = harsh_c = 0
    sp_up, sp_all, energy, dist = [], [], 0.0, 0.0
    ttc_viol = n_ttc = 0
    arrived = 0
    a_cav_min = 9.0
    for t in range(int(horizon / DT)):
        tt = t * DT
        ids = c.vehicle.getIDList()
        cavs = [v for v in ids if v.startswith("cm")] if controlled else []
        acts = {}
        for cid in cavs:
            v = c.vehicle.getSpeed(cid)
            if cid not in held:
                c.vehicle.setSpeedMode(cid, 0); c.vehicle.setLaneChangeMode(cid, 0)
                held[cid] = 0.0
            ld = c.vehicle.getLeader(cid, 200.0)
            if ld and ld[0] != "":
                lid, g = ld[0], ld[1] + MINGAP["cav"]
                vl = c.vehicle.getSpeed(lid)
                l2 = c.vehicle.getLeader(lid, 200.0)
                if l2 and l2[0] != "":
                    vl2 = c.vehicle.getSpeed(l2[0]); gl = l2[1] + MINGAP[c.vehicle.getTypeID(lid)]
                else:
                    vl2, gl = vl, 200.0
                lead_cav = c.vehicle.getTypeID(lid) == "cav"
            else:
                lid, g, vl, vl2, gl, lead_cav = None, 200.0, v, v, 200.0, False
            extra = []
            if shield is not None:
                for mode, sig in ((2, 2), (3, 1)):       # right-lane leaders signalling left; left-lane signalling right
                    for nid, nd in c.vehicle.getNeighbors(cid, mode):
                        if nd + MINGAP["cav"] < 60.0 and (c.vehicle.getSignals(nid) & sig):
                            extra.append((max(nd + MINGAP["cav"], 0.0), c.vehicle.getSpeed(nid), 0.0))
            fo = c.vehicle.getFollower(cid, 200.0)
            if fo and fo[0] != "":
                fid = fo[0]; vf = c.vehicle.getSpeed(fid); gf = fo[1] + MINGAP[c.vehicle.getTypeID(fid)]
            else:
                fid, vf, gf = None, v, 200.0
            vmean = c.edge.getLastStepMeanSpeed(c.vehicle.getRoadID(cid)) if c.vehicle.getRoadID(cid) else v
            if agent is not None:
                if t % 5 == 0:
                    o = np.array([[[v / 20, vl / 20, (vl - v) / 5, min(g, 60) / 30, vf / 20, min(gf, 60) / 30,
                                    aprev.get(cid, 0.0) / 3, vl2 / 20, min(gl, 60) / 30, vmean / 20]]])
                    a, _ = agent.act(o, deterministic=True)
                    held[cid] = float(np.clip(a[0, 0, 0] * agent.act_scale, -3, 2))
                a_des = held[cid]
            else:
                a_des = pi(cid, v, vl, g)
            a_ex = a_des
            if shield is not None:
                if shield.predictor == "persist":
                    al_hat = c.vehicle.getAcceleration(lid) if lid else 0.0
                    af_hat = c.vehicle.getAcceleration(fid) if fid else 0.0
                else:
                    al_hat = idm_hat(vl, vl2, gl) if lid else 0.0
                    af_hat = idm_hat(vf, v, gf) if fid else 0.0
                a_ex, inf, bnd = shield.filter(cid, a_des, g if lid else None, v, vl, al_hat, lead_cav,
                                               gf if fid else None, vf, af_hat, extra=extra)
                pending[cid] = (bnd, lid, fid)
                if tt >= warm:
                    interv += abs(a_ex - a_des) > 1e-6; infeas += inf; n_ctrl += 1
            acts[cid] = a_ex
            c.vehicle.setSpeed(cid, max(0.0, v + a_ex * DT))
        c.simulationStep()
        # post-step bookkeeping
        if tt >= warm:
            n_coll += c.simulation.getCollidingVehiclesNumber()
            for col in c.simulation.getCollisions():
                key = (col.collider, col.victim)
                if key not in coll_log:
                    coll_log[key] = (round(tt, 1), col.colliderType, col.victimType, col.type, col.lane)
            coll_ids.update(c.simulation.getCollidingVehiclesIDList())
            arrived += c.simulation.getArrivedNumber()
        alive = set(c.vehicle.getIDList())
        if shield is not None:
            for cid, (bnd, lid, fid) in pending.items():
                al = c.vehicle.getAcceleration(lid) if (lid and lid in alive) else None
                af = c.vehicle.getAcceleration(fid) if (fid and fid in alive) else None
                shield.update(cid, bnd, al, af)
            pending = {}
        for cid in cavs:
            if cid in alive:
                aprev[cid] = c.vehicle.getAcceleration(cid)
        if tt >= warm and t % 5 == 0:
            ids2 = c.vehicle.getIDList()
            if ids2:
                vs = np.array([c.vehicle.getSpeed(v) for v in ids2])
                acc = np.array([c.vehicle.getAcceleration(v) for v in ids2])
                is_c = np.array([v.startswith("cm") and controlled for v in ids2])
                harsh_h += int(((acc < -4.0) & ~is_c).sum()); harsh_c += int(((acc < -4.0) & is_c).sum())
                if is_c.any():
                    a_cav_min = min(a_cav_min, float(acc[is_c].min()))
                P = 1500 * vs * (acc + 9.81 * 0.01) + 0.5 * 1.2 * 0.7 * vs ** 3
                energy += np.maximum(P, 0).sum() * 0.5; dist += vs.sum() * 0.5
                sp_all.append(vs.mean())
                sp_up.append(c.edge.getLastStepMeanSpeed("up"))
                for vid in ids2[::3]:
                    ld = c.vehicle.getLeader(vid, 60.0)
                    if ld and ld[0] != "":
                        g = ld[1] + MINGAP[c.vehicle.getTypeID(vid)]
                        clo = max(0.0, c.vehicle.getSpeed(vid) - c.vehicle.getSpeed(ld[0]))
                        n_ttc += 1; ttc_viol += (g - 1.5 * clo) < 0
    c.close()
    T_eval = (horizon - warm) / 3600.0
    sp_up = np.array(sp_up)
    cav_at_fault = sum(1 for (a, b), v in coll_log.items() if v[1] == "cav")
    cav_victim = sum(1 for (a, b), v in coll_log.items() if v[2] == "cav" and v[1] != "cav")
    human_only = sum(1 for (a, b), v in coll_log.items() if v[1] == "human" and v[2] == "human")
    return {"collisions": n_coll, "colliding_vehicles": len(coll_ids), "collision": len(coll_ids) > 0,
            "coll_events": len(coll_log), "coll_cav_rear_ends": cav_at_fault, "coll_human_into_cav": cav_victim,
            "coll_human_human": human_only,
            "throughput_vph": arrived / T_eval, "mean_v": float(np.mean(sp_all)),
            "upstream_speed_std": float(sp_up.std()), "upstream_mean_speed": float(sp_up.mean()),
            "energy_MJkm": energy / max(dist, 1) * 1e-3,
            # vehicle-samples (taken every 0.5 s) with acceleration below -4 m/s^2, per evaluated hour
            "harsh_brake_human_per_h": harsh_h / T_eval, "harsh_brake_cav_per_h": harsh_c / T_eval, "ttc_viol_frac": ttc_viol / max(n_ttc, 1),
            "interv_rate": interv / max(n_ctrl, 1), "infeasible_steps": infeas, "a_cav_min": a_cav_min}


def main_highway(args):
    work = os.path.abspath(args.work); os.makedirs(work, exist_ok=True)
    pred, lo, up, width = select_and_calibrate(calibrate_sumo_hw, work)
    print(f"SUMO highway predictor widths {width} -> {pred}; n={len(lo)} q99_low={np.quantile(lo,.99):.3f} "
          f"q99_up={np.quantile(up,.99):.3f}", flush=True)
    rows = []
    methods = args.methods.split(",")
    for ep in range(args.ep_start, args.ep_start + args.episodes):
        seed = 3000 + ep
        for m in methods:
            if m == "Human (EIDM)":
                r = run_hw_episode(work, "Human", seed=seed, horizon=args.horizon, warm=args.warm)
            elif m == "PI-Saturation":
                r = run_hw_episode(work, "PI", seed=seed, horizon=args.horizon, warm=args.warm)
            else:
                ag = load_agent(WEIGHTS[m], ep % 3)
                sh = None if SHIELD_CFG[m] is None else Shield(lo, up, predictor=pred, **SHIELD_CFG[m])
                r = run_hw_episode(work, m, agent=ag, shield=sh, seed=seed, horizon=args.horizon, warm=args.warm)
                if sh is not None and sh.n:
                    r["miscoverage"] = sh.miscov / sh.n
            rows.append({"method": m, "episode": ep, "scenario": "SUMO-highway merge", **r})
            print(ep, m, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(args.out, f"sumo_highway_episodes{args.tag}.csv"), index=False)


RING_METHODS = ["PPO", "PPO+CBF", "PPO+CF-shield (post-hoc)", "CertiFlow", "PPO-DR", "CertiFlow-DR", "CertiFlow-DR-FT"]


def main_ring(args):
    work = os.path.abspath(args.work); os.makedirs(work, exist_ok=True)
    pred, lo, up, width = select_and_calibrate(calibrate_sumo_ring, work)
    print(f"SUMO ring predictor widths {width} -> {pred}; n={len(lo)} q99_low={np.quantile(lo,.99):.3f} "
          f"q99_up={np.quantile(up,.99):.3f}", flush=True)
    rows = []
    for scen, em in (("SUMO-ring nominal", False), ("SUMO-ring emergency", True)):
        for ep in range(args.episodes):
            sumo_seed = 2000 + ep
            r = run_ring_episode(work, "Human", seed=sumo_seed, emergency=em); rows.append({"method": "Human (EIDM)", **r})
            r = run_ring_episode(work, "PI", seed=sumo_seed, emergency=em); rows.append({"method": "PI-Saturation", **r})
            for m in RING_METHODS:
                pseed = ep % 3
                ag = load_agent(WEIGHTS[m], pseed)
                sh = None if SHIELD_CFG[m] is None else Shield(lo, up, predictor=pred, **SHIELD_CFG[m])
                r = run_ring_episode(work, m, agent=ag, shield=sh, seed=sumo_seed, emergency=em)
                if sh is not None and sh.n:
                    r["miscoverage"] = sh.miscov / sh.n
                rows.append({"method": m, **r})
            for rr in rows[-(2 + len(RING_METHODS)):]:
                rr["scenario"], rr["episode"] = scen, ep
            print(scen, ep, {r["method"]: (r["collision"], round(r["std_v_ss"], 3), round(r["mean_v"], 2))
                             for r in rows[-(2 + len(RING_METHODS)):]}, flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(args.out, "sumo_ring_episodes.csv"), index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="ring")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--work", default="sumo_work")
    ap.add_argument("--out", default="results")
    ap.add_argument("--ep_start", type=int, default=0)
    ap.add_argument("--horizon", type=float, default=600.0)
    ap.add_argument("--warm", type=float, default=200.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--methods", default="Human (EIDM),PI-Saturation,PPO,PPO+CF-shield (post-hoc),CertiFlow-DR,CertiFlow-DR-FT")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.scenario == "ring":
        main_ring(args)
    else:
        main_highway(args)
