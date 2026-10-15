"""CertiFlow: certified multi-agent RL for mixed-autonomy traffic.

Modules
-------
env      : vectorised ring-road mixed-traffic simulator (IDM human drivers + CAVs)
shield   : conformal, cooperative discrete-time control-barrier-function safety shield
stl      : Signal Temporal Logic quantitative semantics (robustness) + traffic specs
ppo      : NumPy PPO with parameter sharing (centralised training, decentralised execution)
baselines: human-only IDM and PI-with-saturation (Stern et al., 2018) controllers
certify  : linearised spectral stability certificate + statistical (Clopper-Pearson) bounds
"""
__version__ = "1.0.0"
