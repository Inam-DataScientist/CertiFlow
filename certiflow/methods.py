from __future__ import annotations
import copy
from .env import EnvConfig
from .shield import ConformalCBFShield

METHODS = {
    "PPO":             ({"w_headway": 1.0}, None),
    # PPO + classic (nominal-model, ego-only) CBF safety filter
    "PPO+CBF":         ({"w_headway": 1.0}, dict(conformal=False, cooperative=False, adaptive=False)),
    "CertiFlow":       ({"w_stl": 1.0, "w_interv": 0.2},
                        dict(conformal=True, cooperative=True, adaptive=True)),
    "CertiFlow-noSTL": ({"w_headway": 1.0}, dict(conformal=True, cooperative=True, adaptive=True)),
    "CertiFlow-noCoop": ({"w_stl": 1.0, "w_interv": 0.2},
                         dict(conformal=True, cooperative=False, adaptive=True)),
    "CertiFlow-noACI": ({"w_stl": 1.0, "w_interv": 0.2},
                        dict(conformal=True, cooperative=True, adaptive=False)),
}
def make_cfg(method: str, base: EnvConfig | None = None, **over) -> EnvConfig:
    cfg = copy.deepcopy(base) if base is not None else EnvConfig()
    for k, v in METHODS.get(method, ({}, None))[0].items():
        setattr(cfg, k, v)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def make_shield(method: str, cfg: EnvConfig, scores):
    kw = METHODS.get(method, ({}, None))[1]
    if kw is None:
        return None
    lo, up = scores
    return ConformalCBFShield(cfg, scores_low=lo, scores_up=up, **kw)
