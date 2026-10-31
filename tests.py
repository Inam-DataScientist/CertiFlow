"""Self-checks.  Usage: python tests.py  (each check prints PASS or FAIL)."""
import numpy as np

from certiflow.env import RingEnv, EnvConfig
from certiflow.shield import ConformalCBFShield, calibrate_scores
from certiflow.stl import _window
from certiflow.ppo import MLP
from certiflow.certify import spectral_certificate, clopper_pearson_upper


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    return ok


def test_backprop():
    rng = np.random.default_rng(1); net = MLP([5, 8, 8, 2], rng, out_scale=1.0)
    x = rng.normal(size=(7, 5)); w = rng.normal(size=(7, 2))
    f = lambda: (net.forward(x) * w).sum()
    f(); g = net.backward(w); err = 0
    for p, gp in zip(net.params(), g):
        for _ in range(5):
            idx = tuple(rng.integers(0, s) for s in p.shape); old = p[idx]
            p[idx] = old + 1e-6; fp = f(); p[idx] = old - 1e-6; fm = f(); p[idx] = old
            err = max(err, abs((fp - fm) / 2e-6 - gp[idx]))
    return check(f"MLP backprop matches finite differences (err={err:.1e})", err < 1e-6)


def test_stl_window():
    rng = np.random.default_rng(0); r = rng.normal(size=(3, 57)); ok = True
    for a, b in [(0, None), (2, 9), (5, 100)]:
        bb = 56 if b is None else min(b, 56)
        ref = np.array([[r[i, t + a:t + bb + 1].min() if t + a <= 56 else np.inf for t in range(57)]
                        for i in range(3)])
        ok &= np.allclose(_window(r, a, b, np.minimum, np.inf), ref)
    return check("STL sliding-window robustness equals brute force", ok)


def test_shield_blocks_reckless_policy():
    cfg = EnvConfig(n_cav=2)
    lo, up = calibrate_scores(cfg, seconds=60)
    res = {}
    for name, sh in [("none", None), ("shield", ConformalCBFShield(cfg, scores_low=lo, scores_up=up))]:
        env = RingEnv(cfg, n_envs=16); env.shield = sh; env.reset(); coll = 0
        for _ in range(200):
            _, _, d, info = env.step(np.full((16, 2), 2.0))      # always floor it
            coll += info["collision"].sum()
            if d.any():
                env.reset(np.where(d)[0])
        res[name] = coll
    return check(f"shield: reckless CAV collisions {res['none']} -> {res['shield']}",
                 res["none"] > 0 and res["shield"] == 0)


def test_braking_feasibility():
    """Lemma 2: maximum braking never decreases the barrier, for any leader acceleration a_l >= -b."""
    cfg = EnvConfig(); sh = ConformalCBFShield(cfg)
    s = np.linspace(-5.0, 60.0, 131)[:, None, None, None]
    v = np.linspace(0.0, 30.0, 121)[None, :, None, None]
    vl = np.linspace(0.0, 30.0, 121)[None, None, :, None]
    al = np.linspace(-sh.b, cfg.a_max_phys, 41)[None, None, None, :]
    d = sh._next_h(-sh.b, s, v, vl, al, sh.tau) - sh.barrier(s, v, vl, sh.tau)
    return check(f"Lemma 2: min h(x+) - h(x) under maximum braking = {d.min():.1e}", d.min() > -1e-9)


def test_human_ring_unstable():
    r = spectral_certificate(EnvConfig(n_cav=0))
    return check(f"human-only ring is linearly unstable (rho={r['rho']:.4f})", r["rho"] > 1.0)


def test_cp():
    return check("Clopper-Pearson 0/60 upper bound = 0.0487",
                 abs(clopper_pearson_upper(0, 60) - 0.0487) < 1e-3)


if __name__ == "__main__":
    results = [t() for t in (test_backprop, test_stl_window, test_braking_feasibility,
                             test_shield_blocks_reckless_policy, test_human_ring_unstable, test_cp)]
    print(f"{sum(results)}/{len(results)} checks passed")
