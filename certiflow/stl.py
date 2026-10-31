from __future__ import annotations
import numpy as np

class Formula:
    def rob(self, sig: dict) -> np.ndarray:            
        raise NotImplementedError

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)

    def robustness(self, sig: dict) -> np.ndarray:     
        return self.rob(sig)[:, 0]

class Pred(Formula):
    def __init__(self, fn, name="mu"):
        self.fn, self.name = fn, name

    def rob(self, sig):
        return np.asarray(self.fn(sig), dtype=float)

    def __repr__(self):
        return self.name

class Not(Formula):
    def __init__(self, a):
        self.a = a

    def rob(self, sig):
        return -self.a.rob(sig)

    def __repr__(self):
        return f"!({self.a})"

class And(Formula):
    def __init__(self, *xs):
        self.xs = xs

    def rob(self, sig):
        return np.minimum.reduce([x.rob(sig) for x in self.xs])

    def __repr__(self):
        return " & ".join(f"({x})" for x in self.xs)

class Or(Formula):
    def __init__(self, *xs):
        self.xs = xs

    def rob(self, sig):
        return np.maximum.reduce([x.rob(sig) for x in self.xs])

    def __repr__(self):
        return " | ".join(f"({x})" for x in self.xs)

def _window(r, a, b, op, empty):
    """out[t] = op over r[t+a : t+b+1] (truncated); empty windows get `empty`."""
    B, T = r.shape
    b = min(b, T - 1) if b is not None else T - 1
    out = np.full((B, T), empty, dtype=float)
    W = b - a + 1
    if W <= 0:
        return out
    pad = np.concatenate([r, np.full((B, b + 1), empty)], axis=1)
    acc = pad[:, a:a + T + W].copy()
    width = 1
    while width < W:
        step = min(width, W - width)
        acc = op(acc[:, :-step] if step > 0 else acc, acc[:, step:])
        width += step
    out = acc[:, :T]
    return out

class Always(Formula):
    def __init__(self, a, interval=(0, None)):
        self.a, (self.lo, self.hi) = a, interval

    def rob(self, sig):
        return _window(self.a.rob(sig), self.lo, self.hi, np.minimum, np.inf)

    def __repr__(self):
        return f"G[{self.lo},{self.hi}]({self.a})"


class Eventually(Formula):
    def __init__(self, a, interval=(0, None)):
        self.a, (self.lo, self.hi) = a, interval

    def rob(self, sig):
        return _window(self.a.rob(sig), self.lo, self.hi, np.maximum, -np.inf)

    def __repr__(self):
        return f"F[{self.lo},{self.hi}]({self.a})"

def traffic_specs(dt: float, d_min=1.0, ttc_min=1.5, sigma_star=0.75, t_settle=60.0, t_hold=20.0,
                  harsh=-4.0):
    n_settle, n_hold = int(t_settle / dt), int(t_hold / dt)
    safe = Always(Pred(lambda s: s["gap_min"] - d_min, "gap>=d_min"))
    ttc = Always(Pred(lambda s: s["ttc_marg"], f"TTC>={ttc_min}s"))
    damp = Eventually(Always(Pred(lambda s: sigma_star - s["std_v"], f"std_v<={sigma_star}"),
                             (0, n_hold)), (0, n_settle))
    comfort = Always(Pred(lambda s: s["a_cav_min"] - harsh, f"a_cav>={harsh}"))
    return {"phi_safe": safe, "phi_ttc": ttc, "phi_damp": damp, "phi_comfort": comfort,
            "phi_all": And(safe, ttc, damp, comfort)}


def signals_from_recording(x, v, a, env, ttc_min=1.5):
    """Build STL signals from env.recording() arrays of shape (T, E, N) -> dict of (E, T)."""
    L, ell, lead = env.L[None, :, None], env.cfg.veh_length, env.lead
    gap = np.mod(x[:, :, lead] - x, L) - ell                           
    closing = np.maximum(0.0, v - v[:, :, lead])
    sig = {
        "gap_min": gap.min(axis=2).T,
        "ttc_marg": (gap - ttc_min * closing).min(axis=2).T,
        "std_v": v.std(axis=2).T,
        "a_cav_min": (a[:, :, env.cav_idx].min(axis=2).T if env.K > 0 else np.zeros(gap.shape[:2]).T + 9.0),
    }
    return sig
