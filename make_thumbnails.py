"""Small inset plots embedded in the framework diagram (Fig. 2) -> figures/thumb_*.pdf

Usage:  python make_thumbnails.py
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import BLUE, BAND, RED, GREY, save
from certiflow.env import EnvConfig
from certiflow.shield import calibrate_scores, ConformalCBFShield

plt.rcParams.update({"font.size": 6, "axes.labelsize": 6, "xtick.labelsize": 5, "ytick.labelsize": 5,
                     "axes.linewidth": 0.5})


def thumb_stl():
    tr = np.load("results/traces_S3.npz")
    x = tr["CertiFlow-DR-FT|x"]; T, N = x.shape; t = np.arange(T) * 0.1
    gmin = (np.mod(x[:, np.r_[1:N, 0]] - x, 260) - 5).min(1)
    fig, ax = plt.subplots(figsize=(1.55, 0.95))
    ax.plot(t, gmin, color=BLUE, lw=0.8)
    ax.axhline(1.0, color=RED, lw=0.7, ls="-.")
    i = int(np.argmin(gmin)); ax.annotate("", (t[i], 1.0), (t[i], gmin[i]), arrowprops=dict(arrowstyle="<->", lw=0.5))
    ax.text(t[i] + 3, (1 + gmin[i]) / 2, r"$\rho_{\mathrm{safe}}$", fontsize=6)
    ax.set_xlabel("t (s)", labelpad=1); ax.set_ylabel(r"$g_{\min}$ (m)", labelpad=1); ax.set_xlim(0, 100)
    ax.tick_params(length=1.5, pad=1)
    save(fig, "thumb_stl")


def thumb_conformal():
    lo, _ = calibrate_scores(EnvConfig(), seconds=60)
    q = np.quantile(lo, 0.99)
    fig, ax = plt.subplots(figsize=(1.55, 0.95))
    ax.hist(lo, bins=np.linspace(-2, 2, 60), color=BAND, edgecolor=BLUE, lw=0.2)
    ax.axvline(q, color=RED, lw=0.8, ls="-.")
    ax.text(q + 0.08, ax.get_ylim()[1] * 0.75, r"$q_{0.99}$", color=RED, fontsize=6)
    ax.set_xlabel(r"score $\hat a_l-a_l$", labelpad=1); ax.set_yticks([]); ax.tick_params(length=1.5, pad=1)
    save(fig, "thumb_conformal")


def thumb_shield():
    cfg = EnvConfig(); sh = ConformalCBFShield(cfg)
    s, v, vl, al = 6.0, 6.0, 4.5, -0.9; sf, vf, af = 5.0, 6.4, 0.6
    a = np.linspace(-7, 3, 300)
    h = sh._next_h(a, s, v, vl, al, sh.tau) - (1 - sh.gamma) * sh.barrier(s, v, vl, sh.tau)
    hf = sh._next_h(af, sf, vf, v, a, sh.tau_f) - (1 - sh.gamma) * sh.barrier(sf, vf, v, sh.tau_f)
    up = a[h >= 0].max(); lo = a[hf >= 0].min()
    fig, ax = plt.subplots(figsize=(1.55, 0.95))
    ax.axvspan(lo, up, color=BAND, lw=0)
    ax.plot(a, h, color=BLUE, lw=0.8); ax.plot(a, hf, color=GREY, lw=0.8, ls="--"); ax.axhline(0, color="k", lw=0.4)
    ax.plot([1.5], [-0.3], "o", color=RED, ms=2.5); ax.annotate("", (up, -0.3), (1.5, -0.3),
                                                              arrowprops=dict(arrowstyle="-|>", lw=0.5, color=RED))
    ax.set_xlabel(r"$a$ (m/s$^2$)", labelpad=1); ax.set_ylabel("slack", labelpad=1); ax.set_xlim(-7, 3)
    ax.tick_params(length=1.5, pad=1)
    save(fig, "thumb_shield")


def thumb_spectrum():
    from certiflow.certify import spectral_certificate
    hu = spectral_certificate(EnvConfig(n_cav=0))
    fig, ax = plt.subplots(figsize=(1.2, 1.0))
    th = np.linspace(0, 2 * np.pi, 300); ax.plot(np.cos(th), np.sin(th), "k-", lw=0.6)
    ax.plot(hu["eigs"].real, hu["eigs"].imag, "o", mfc="none", mec=RED, ms=2)
    ax.set_xlim(0.78, 1.04); ax.set_ylim(-0.25, 0.25)
    ax.text(0.80, 0.17, r"$\rho(J)<1$?", fontsize=6); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(r"Re$\,\lambda$", labelpad=1); ax.set_ylabel(r"Im$\,\lambda$", labelpad=1)
    save(fig, "thumb_spectrum")


if __name__ == "__main__":
    for f in (thumb_stl, thumb_conformal, thumb_shield, thumb_spectrum):
        f()
