"""Figures of the paper generated from the result files -> figures/*.pdf, figures/*.png

  Fig. 4   fig_learning         training curves and safe fine-tuning in SUMO
  Fig. 5   fig_spacetime        space-time diagrams under emergency braking (S3)
  Fig. 6   fig_timeseries       speed, acceleration, gap and TTC of the first CAV in S3
  Fig. 7   fig_phase            safe-set phase portrait with and without the shield
  Fig. 8   fig_robustness       STL robustness per episode
  Fig. 9   fig_overall          collision rate and penetration-rate study
  Fig. 10  fig_certificate      collective-stability certificate
  Fig. 11  fig_residuals        predictor residuals on SUMO calibration data
  Fig. 12  fig_sumo_spacetime   space-time diagrams on the SUMO ring
  Fig. 13  fig_sumo             transfer to SUMO
  Fig. 14  fig_uncertainty      performance under increasing human-behaviour uncertainty
  Fig. 15  fig_sensitivity      sensitivity to the shield parameters
  Fig. 16  fig_coverage         coverage of the human-acceleration bound under shift

Usage:  python make_figures.py [learning spacetime ...]   (default: all)
"""
from __future__ import annotations

import glob, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = "figures"; os.makedirs(OUT, exist_ok=True)
BLUE, BAND, RED, GREY, DGREY = "#1f77b4", "#c6dbef", "#d62728", "#9e9e9e", "#4d4d4d"
ORANGE, GREEN = "#ff7f0e", "#2ca02c"
LIGHT = "#6baed6"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "axes.spines.top": True, "axes.spines.right": True, "axes.grid": False,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": True, "legend.edgecolor": "#cccccc", "legend.fancybox": False, "legend.framealpha": 1.0,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "lines.linewidth": 1.1})
COLW, FULLW = 3.5, 7.2
SPEED_CMAP = LinearSegmentedColormap.from_list("spd", ["#67000d", "#d7301f", "#fc8d59", "#fee08b", "#91cf60",
                                                       "#1a9850", "#00441b"])
NAME = {"Human (IDM)": "Human (IDM)", "PI-Saturation": "PI-Saturation", "PPO": "PPO", "PPO-DR": "PPO-DR",
        "PPO+CBF": "PPO+CBF", "PPO+CF-shield (post-hoc)": "PPO+shield (post-hoc)", "CertiFlow": "CertiFlow w/o DR",
        "CertiFlow-DR": "CertiFlow w/o FT", "CertiFlow-DR-FT": "CertiFlow (ours)", "Human (EIDM)": "Human (EIDM)"}


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=200)
    plt.close(fig)
    print("saved", name)


def band_plot(ax, x, groups, title, ylabel, nominal_x=1.0, fmt="{:.3g}", xlabel=None, color=BLUE, show_ref=True):
    """groups: list of arrays (episode values per x)."""
    mu = np.array([np.mean(g) for g in groups])
    p10 = np.array([np.percentile(g, 10) for g in groups]); p90 = np.array([np.percentile(g, 90) for g in groups])
    ax.fill_between(x, p10, p90, color=BAND, alpha=0.75, lw=0, label="10–90th pct.")
    ax.plot(x, mu, "-o", color=color, ms=2.8, label="mean")
    i = int(np.argmin(np.abs(np.array(x) - nominal_x)))
    if show_ref:
        ax.axhline(mu[i], color=RED, ls="-.", lw=0.9, label=f"Ours (nominal): {fmt.format(mu[i])}")
        ax.axvline(nominal_x, color="k", ls="--", lw=0.8)
        ax.plot([x[i]], [mu[i]], "o", ms=6, mfc=RED, mec="k", mew=0.7, zorder=5)
        ax.annotate(fmt.format(mu[i]), (x[i], mu[i]), xytext=(4, 4), textcoords="offset points",
                    fontsize=7, fontweight="bold")
    ax.set_title(title); ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    return mu


# ------------------------------------------------------------------ Fig. 14: human-behaviour uncertainty
def fig_uncertainty():
    u = pd.read_csv("results/sweep_uncertainty.csv")
    c = u[u.method == "CertiFlow"]; p = u[u.method == "PPO"]
    ks = sorted(c.k.unique())
    sig = [0.3 * k for k in ks]
    def G(df, col):
        return [df[df.k == k][col].values for k in ks]
    fig, axs = plt.subplots(3, 3, figsize=(FULLW, 5.3))
    xl = r"human acceleration noise $\sigma$ (m/s$^2$)"
    spec = [("(a) Collision rate", "Collision rate", "collision", "{:.2f}"),
            ("(b) Min gap (m)", "Min gap (m)", "min_gap", "{:.2f}"),
            ("(c) Min TTC (s)", "Min TTC (s)", "min_ttc", "{:.2f}"),
            ("(d) Wave amplitude (m/s)", "Wave amplitude (m/s)", "std_v_ss", "{:.3f}"),
            ("(e) Mean speed (m/s)", "Mean speed (m/s)", "mean_v", "{:.2f}"),
            ("(f) Tractive energy (MJ/km)", "Energy (MJ/km)", "energy_MJkm", "{:.3f}"),
            ("(g) CAV mean |jerk| (m/s$^3$)", "Mean |jerk| (m/s$^3$)", "cav_mean_abs_jerk", "{:.3f}"),
            ("(h) Shield intervention rate", "Intervention rate", "interv_rate", "{:.3f}"),
            ("(i) Conformal miscoverage", "Miscoverage", "miscoverage", "{:.4f}")]
    for ax, (t, yl, col, fmt) in zip(axs.flat, spec):
        groups = [g.astype(float) for g in G(c, col)]
        band_plot(ax, sig, groups, t, yl, nominal_x=0.3, fmt=fmt, xlabel=xl)
        if col in ("collision", "min_gap", "min_ttc", "std_v_ss"):
            ax.plot(sig, [np.mean(g.astype(float)) for g in G(p, col)], "s--", color=GREY, ms=2.6, lw=0.9,
                    label="PPO (no shield)")
        if col == "miscoverage":
            ax.axhline(0.01, color=GREEN, ls=":", lw=0.9, label=r"target $\alpha$ = 0.01")
        if col == "min_ttc":
            ax.set_yscale("log"); ax.set_yticks([2, 5, 10, 20, 50]); ax.set_yticklabels(["2", "5", "10", "20", "50"])
            ax.minorticks_off()
        h_, l_ = ax.get_legend_handles_labels()
        keep = [(a, b) for a, b in zip(h_, l_) if b.startswith("Ours")]
        ax.legend(*zip(*keep), loc="upper left" if col not in ("min_gap", "min_ttc", "mean_v") else "lower left",
                  fontsize=6)
    handles = [Patch(color=BAND, label="CertiFlow (ours): 10\u201390th percentile over 60 episodes"),
               Line2D([], [], color=BLUE, marker="o", ms=3, label="CertiFlow (ours): mean"),
               Line2D([], [], color=GREY, marker="s", ls="--", ms=3, label="PPO (no shield): mean"),
               Line2D([], [], color=GREEN, ls=":", label=r"target $\alpha$ = 0.01"),
               Line2D([], [], color="k", ls="--", label=r"nominal training noise $\sigma$ = 0.3 m/s$^2$")]
    fig.tight_layout(h_pad=0.9, w_pad=0.9, rect=(0, 0, 1, 0.955))
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=6.6, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_uncertainty")


# ------------------------------------------------------------------ Fig. 15: shield-parameter sensitivity
def fig_sensitivity():
    s = pd.read_csv("results/sweep_sensitivity.csv")
    params = [("gamma", r"CBF decay rate $\gamma$", 0.3), ("alpha", r"conformal level $\alpha$", 0.01),
              ("d0", r"standstill margin $d_0$ (m)", 2.0), ("tau", r"reaction allowance $\tau$ (s)", 0.3)]
    metrics = [("std_v_ss", "Wave amplitude (m/s)", "{:.3f}"), ("mean_v", "Mean speed (m/s)", "{:.2f}"),
               ("min_gap", "Min gap (m)", "{:.2f}"), ("interv_rate", "Intervention rate", "{:.3f}")]
    fig, axs = plt.subplots(4, 4, figsize=(FULLW, 5.9))
    letters = "abcd"
    for r, (pn, plabel, nom) in enumerate(params):
        d = s[s.param == pn]; vals = sorted(d.value.unique())
        for c_, (col, ml, fmt) in enumerate(metrics):
            ax = axs[r, c_]
            groups = [d[d.value == v][col].values for v in vals]
            band_plot(ax, vals, groups, f"({letters[r]}{c_+1}) {ml}", ml, nominal_x=nom, fmt=fmt, xlabel=plabel)
            if pn == "alpha":
                ax.set_xscale("log")
    handles = [Patch(color=BAND, label="10\u201390th percentile over 60 episodes (S2, distribution shift)"),
               Line2D([], [], color=BLUE, marker="o", ms=3, label="mean"),
               Line2D([], [], color="k", ls="--", label="nominal value used in the paper"),
               Line2D([], [], color=RED, ls="-.", label="Ours at the nominal value (bold)")]
    fig.tight_layout(h_pad=0.7, w_pad=0.7, rect=(0, 0, 1, 0.965))
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=6.4, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_sensitivity")


# ------------------------------------------------------------------ Fig. 5: space-time diagrams (S3)
def fig_spacetime():
    tr = np.load("results/traces_S3.npz")
    methods = [("Human (IDM)", "(a) Human (IDM)"), ("PI-Saturation", "(b) PI-Saturation"), ("PPO", "(c) PPO (no shield)"),
               ("CertiFlow-DR-FT", "(d) CertiFlow (ours)")]
    fig, axs = plt.subplots(1, 4, figsize=(FULLW, 2.25), sharey=True, gridspec_kw={"wspace": 0.07})
    for ax, (m, title) in zip(axs, methods):
        x, v = tr[f"{m}|x"], tr[f"{m}|v"]; T, N = v.shape; t = np.arange(T) * 0.1; L = 260.0
        gap = np.mod(x[:, np.r_[1:N, 0]] - x, L) - 5.0; hit = np.where((gap < 0).any(1))[0]
        Tm = hit[0] + 1 if len(hit) else T
        pos = np.mod(x, L)
        sc = None
        for i in range(N):
            sc = ax.scatter(t[:Tm:3], pos[:Tm:3, i], c=v[:Tm:3, i], cmap=SPEED_CMAP, vmin=0, vmax=8, s=1.1, lw=0,
                            rasterized=True)
        if len(hit):
            i0 = int(np.argmin(gap[hit[0]]))
            ax.plot(t[Tm - 1], pos[Tm - 1, i0], "X", color="k", ms=7, mec="w", mew=0.8)
            ax.annotate(f"Collision\n{t[Tm-1]:.1f} s", (t[Tm - 1], pos[Tm - 1, i0]), xytext=(8, -2),
                        textcoords="offset points", fontsize=7)
        for t0 in (20, 55):
            ax.axvline(t0, color="k", ls="--", lw=0.6)
        ax.set_title(title); ax.set_xlim(0, 100); ax.set_ylim(0, 260); ax.set_xlabel("Time (s)")
        ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", ""])
    axs[0].set_ylabel("Position on ring (m)")
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SPEED_CMAP, norm=plt.Normalize(0, 8)), ax=axs, fraction=0.016, pad=0.012)
    cb.set_label("Speed (m/s)"); cb.outline.set_linewidth(0.5)
    save(fig, "fig_spacetime")


# ------------------------------------------------------------------ Fig. 6: time series of the first CAV (S3)
def fig_timeseries():
    tr = np.load("results/traces_S3.npz")
    fig, axs = plt.subplots(4, 1, figsize=(COLW, 4.3), sharex=True, gridspec_kw={"hspace": 0.12})
    cfgs = [("Human (IDM)", "Human driver (no CAV)", GREY, "-"), ("PPO", "PPO (no shield)", ORANGE, "--"),
            ("CertiFlow-DR-FT", "CertiFlow (ours)", BLUE, "-")]
    for m, lab, c, ls in cfgs:
        x, v, a = tr[f"{m}|x"], tr[f"{m}|v"], tr[f"{m}|a"]; T, N = v.shape; t = np.arange(T) * 0.1
        gap = np.mod(x[:, 1] - x[:, 0], 260.0) - 5.0
        allgap = np.mod(x[:, np.r_[1:N, 0]] - x, 260.0) - 5.0; hit = np.where((allgap < 0).any(1))[0]
        Tm = hit[0] + 1 if len(hit) else T
        clo = v[:, 0] - v[:, 1]; ttc = np.where(clo > 0.05, np.maximum(gap, 0) / np.maximum(clo, 0.05), np.nan)
        for ax, y in zip(axs, (v[:Tm, 0], a[:Tm, 0], gap[:Tm], np.clip(ttc[:Tm], 0, 20))):
            ax.plot(t[:Tm], y, color=c, ls=ls, lw=0.9 if m != "CertiFlow-DR-FT" else 1.2, label=lab)
        if len(hit):
            for ax in axs:
                ax.axvline(t[Tm - 1], color=ORANGE, lw=0.6, ls=":")
            axs[2].annotate("collision", (t[Tm - 1], 0), xytext=(4, 6), textcoords="offset points", fontsize=6.5,
                            color=ORANGE)
    axs[0].plot(t, tr["CertiFlow-DR-FT|v"][:, 1], color=RED, lw=0.8, ls=":", label="Braking leader")
    for ax, yl in zip(axs, ("Speed (m/s)", r"Accel. (m/s$^2$)", "Gap (m)", "TTC (s)")):
        ax.set_ylabel(yl)
        for t0 in (20, 55):
            ax.axvspan(t0, t0 + 3, color="#f2f2f2", lw=0)
    axs[2].axhline(2.0, color="k", lw=0.6, ls="--"); axs[2].text(96, 2.4, r"$d_0$", fontsize=7, ha="right")
    axs[3].axhline(1.5, color="k", lw=0.6, ls="--"); axs[3].text(96, 2.2, "1.5 s", fontsize=7, ha="right")
    axs[1].axhline(-4, color="k", lw=0.5, ls=":")
    axs[0].legend(loc="lower left", ncol=2, fontsize=5.8)
    axs[-1].set_xlabel("Time (s)"); axs[-1].set_xlim(0, 100)
    save(fig, "fig_timeseries")


# ------------------------------------------------------------------ Fig. 9: overall results
def fig_overall():
    ep = pd.concat([pd.read_csv("results/episodes.csv"), pd.read_csv("results/episodes_dr_numpy.csv")], ignore_index=True)
    s4 = pd.concat([pd.read_csv("results/episodes.csv"), pd.read_csv("results/episodes_s4_ft.csv")], ignore_index=True)
    s4 = s4[s4.scenario == "S4-penetration"]
    fig, axs = plt.subplots(1, 3, figsize=(FULLW, 2.35), gridspec_kw={"width_ratios": [1.45, 1, 1], "wspace": 0.38})
    order = ["PPO", "PPO-DR", "PPO+CBF", "PPO+CF-shield (post-hoc)", "CertiFlow", "CertiFlow-DR", "CertiFlow-DR-FT"]
    lab = ["PPO", "PPO-DR", "PPO+CBF", "PPO+shield", "w/o DR", "w/o FT", "Ours"]
    ax = axs[0]; w = 0.26
    for j, (scen, c) in enumerate((("S1-nominal", "#c6dbef"), ("S2-shift", "#6baed6"), ("S3-emergency", "#08519c"))):
        r = [100 * ep[(ep.method == m) & (ep.scenario == scen)].collision.mean() for m in order]
        ax.bar(np.arange(len(order)) + (j - 1) * w, r, w, color=c, edgecolor="k", lw=0.3,
               label=scen.replace("-", " ").replace("S1 nominal", "S1 nominal"))
    ax.set_xticks(range(len(order))); ax.set_xticklabels(lab, fontsize=6.4, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Episodes with collision (%)"); ax.set_title("(a) Collision rate (60 episodes each)")
    ax.legend(fontsize=6, loc="upper right"); ax.set_ylim(0, 110)
    for i in range(2, len(order)):
        ax.text(i, 3, "0", ha="center", fontsize=6.5, color=BLUE, fontweight="bold")
    hum = s4[s4.method == "Human (IDM)"].groupby("n_cav")
    for ax, col, t in ((axs[1], "std_v_ss", "(b) Wave reduction (%)"), (axs[2], "energy_MJkm", "(c) Energy reduction (%)")):
        for m, c, mk, ls in (("PI-Saturation", GREY, "^", "--"), ("PPO", DGREY, "s", "--"), ("CertiFlow", LIGHT, "d", "-."),
                             ("CertiFlow-DR-FT", BLUE, "o", "-")):
            g = s4[s4.method == m].groupby("n_cav")[col].mean(); h = hum[col].mean()
            red = 100 * (1 - g / h)
            ax.plot(np.array(red.index) / 22 * 100, red.values, ls=ls, marker=mk, ms=3.2, color=c,
                    label={"CertiFlow": "w/o DR", "CertiFlow-DR-FT": "CertiFlow (ours)"}.get(m, m))
        ax.set_xlabel("CAV penetration (%)"); ax.set_title(t); ax.set_xticks([4.5, 9.1, 13.6, 18.2])
    axs[1].set_ylabel("Reduction vs human-only (%)"); axs[1].legend(fontsize=5.8, loc="lower right")
    m1 = s4[(s4.method == "CertiFlow-DR-FT") & (s4.n_cav == 1)].std_v_ss.mean() / hum.std_v_ss.mean().loc[1]
    axs[1].annotate(f"{100*(1-m1):.0f}%", (4.5, 100 * (1 - m1)), xytext=(6, -10), textcoords="offset points",
                    fontsize=7, fontweight="bold", color=BLUE)
    save(fig, "fig_overall")



# ------------------------------------------------------------------ Fig. 10: stability certificate
def fig_certificate():
    from certiflow.env import EnvConfig
    from certiflow.certify import spectral_certificate
    from certiflow.shield import calibrate_scores
    from certiflow.methods import make_cfg, make_shield
    from evaluate import load_agent
    scores = calibrate_scores(EnvConfig())
    hu = spectral_certificate(EnvConfig(n_cav=0))
    cfg = make_cfg("CertiFlow", EnvConfig(n_cav=2))
    cf = spectral_certificate(cfg, load_agent("CertiFlow-DR-FT", 0, required=True), make_shield("CertiFlow", cfg, scores))
    fig, axs = plt.subplots(1, 2, figsize=(FULLW, 2.25), gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.42})
    ax = axs[0]; th = np.linspace(0, 2 * np.pi, 500)
    ax.plot(np.cos(th), np.sin(th), "k-", lw=0.8, label="unit circle")
    ax.plot(hu["eigs"].real, hu["eigs"].imag, "o", mfc="none", mec=RED, ms=4, label=fr"Human only ($\rho$={hu['rho']:.4f})")
    ax.plot(cf["eigs"].real, cf["eigs"].imag, "o", color=BLUE, ms=3.2, label=fr"CertiFlow ($\rho$={cf['rho']:.4f})")
    ax.set_xlim(0.78, 1.05); ax.set_ylim(-0.26, 0.26); ax.set_xlabel(r"Re($\lambda$)"); ax.set_ylabel(r"Im($\lambda$)")
    ax.set_title("(a) Eigenvalues near the unit circle"); ax.legend(fontsize=5.8, loc="lower left")
    certs = {**json.load(open("results/certificates.json")), **json.load(open("results/certificates_dr.json"))}
    groups = [("Human only", ["Human (IDM), 0 CAV"]), ("PPO", [f"PPO s{s}" for s in range(3)]),
              ("PPO+CBF", [f"PPO+CBF s{s}" for s in range(3)]), ("PPO-DR", [f"PPO-DR s{s}" for s in range(3)]),
              ("CertiFlow w/o DR", [f"CertiFlow s{s}" for s in range(3)]),
              ("CertiFlow w/o FT", [f"CertiFlow-DR s{s}" for s in range(3)]),
              ("Ours, 1 CAV", [f"CertiFlow-DR-FT s{s} K=1" for s in range(3)]),
              ("Ours, 2 CAVs", [f"CertiFlow-DR-FT s{s}" for s in range(3)]),
              ("Ours, 3 CAVs", [f"CertiFlow-DR-FT s{s} K=3" for s in range(3)]),
              ("Ours, 4 CAVs", [f"CertiFlow-DR-FT s{s} K=4" for s in range(3)])]
    ax = axs[1]
    for y, (lab, keys) in enumerate(groups):
        vals = [certs[k]["rho"] for k in keys if k in certs]
        c = RED if lab == "Human only" else (BLUE if lab.startswith("Ours") else DGREY)
        ax.plot(vals, [y] * len(vals), "o", color=c, ms=4)
    ax.axvline(1.0, color="k", ls="--", lw=0.8); ax.text(1.002, 9.3, "stability limit", fontsize=6.5)
    ax.set_yticks(range(len(groups))); ax.set_yticklabels([g[0] for g in groups], fontsize=6.6); ax.invert_yaxis()
    ax.set_xlim(0.94, 1.02); ax.set_xlabel(r"Spectral radius $\rho(J)$ per 0.5 s step")
    ax.set_title("(b) Certificate of every trained policy (3 seeds)")
    save(fig, "fig_certificate")


# ------------------------------------------------------------------ Fig. 16: conformal coverage
def fig_coverage():
    from certiflow.env import RingEnv, EnvConfig
    from certiflow.shield import ConformalCBFShield, calibrate_scores
    from evaluate import load_agent
    lo, up = calibrate_scores(EnvConfig()); ag = load_agent("CertiFlow-DR-FT", 0, required=True)
    out = {}
    for name, adaptive in (("Static conformal", False), ("Adaptive conformal (ACI)", True)):
        cfg = EnvConfig(n_cav=2, idm_spread=0.30, accel_noise=0.6, seed=4242); env = RingEnv(cfg, n_envs=20)
        sh = ConformalCBFShield(cfg, scores_low=lo, scores_up=up, adaptive=adaptive); env.shield = sh; env.reset()
        errs, alphas = [], []; orig = sh.update
        def upd(e, vb, sh=sh, errs=errs, alphas=alphas, adaptive=adaptive, orig=orig):
            i = e.cav_idx; L = e.lead[i]; a_l = (e.v[:, L] - vb[:, L]) / e.cfg.dt
            q = sh._quantile(sh.scores_low, sh.alpha_t if adaptive else np.full_like(sh.alpha_t, sh.alpha))
            errs.append(((a_l < sh.pred_l - q - 1e-9) & (e.v[:, L] > 0)).mean()); alphas.append(sh.alpha_t.mean())
            if adaptive:
                orig(e, vb)
        sh.update = upd
        for _ in range(cfg.horizon):
            a, _ = ag.act(env.obs(), deterministic=True); env.step(a[..., 0] * ag.act_scale)
        out[name] = (np.array(errs), np.array(alphas))
    fig, axs = plt.subplots(2, 1, figsize=(COLW, 3.3), sharex=True, gridspec_kw={"hspace": 0.32})
    t = np.arange(len(out["Static conformal"][0])) * 0.1
    for name, c, ls in (("Static conformal", ORANGE, "--"), ("Adaptive conformal (ACI)", BLUE, "-")):
        e = out[name][0]; run = 100 * np.cumsum(e) / np.arange(1, len(e) + 1)
        axs[0].plot(t, run, color=c, ls=ls, label=name)
        axs[0].annotate(f"{run[-1]:.1f}%", (t[-1], run[-1]), xytext=(-22, 4 if run[-1] < 5 else -11), textcoords="offset points",
                        fontsize=7, fontweight="bold", color=c)
    axs[0].axhline(1.0, color=RED, ls="-.", lw=0.9, label=r"Target $\alpha$ = 1%")
    axs[0].set_ylabel("Running miscoverage (%)"); axs[0].set_title("(a) Coverage of the human-acceleration bound (S2)")
    axs[0].legend(fontsize=6, loc="center right"); axs[0].set_ylim(0, 16)
    axs[1].plot(t, out["Adaptive conformal (ACI)"][1], color=BLUE)
    axs[1].axhline(0.01, color=RED, ls="-.", lw=0.9)
    axs[1].set_ylabel(r"Adapted level $\alpha_t$"); axs[1].set_xlabel("Time (s)"); axs[1].set_title(r"(b) ACI level $\alpha_t$")
    save(fig, "fig_coverage")


# ------------------------------------------------------------------ Fig. 13: transfer to SUMO
def fig_sumo():
    r = pd.read_csv("results/sumo_ring_episodes.csv"); r["collision"] = r.collision.astype(bool)
    h = pd.read_csv("results/sumo_highway_episodes.csv")
    order = ["Human (EIDM)", "PI-Saturation", "PPO", "PPO-DR", "PPO+CBF", "PPO+CF-shield (post-hoc)", "CertiFlow",
             "CertiFlow-DR", "CertiFlow-DR-FT"]
    lab = ["Human (EIDM)", "PI-Saturation", "PPO", "PPO-DR", "PPO+CBF", "PPO+shield (post-hoc)", "CertiFlow w/o DR",
           "CertiFlow w/o FT", "CertiFlow (ours)"]
    fig, axs = plt.subplots(1, 3, figsize=(FULLW, 2.7), gridspec_kw={"width_ratios": [1.3, 1, 1.05], "wspace": 0.12})
    y = np.arange(len(order))
    ax = axs[0]
    for k, (scen, mk, c) in enumerate((("SUMO-ring nominal", "o", BLUE), ("SUMO-ring emergency", "s", ORANGE))):
        for i, m in enumerate(order):
            g = r[(r.method == m) & (r.scenario == scen)]; ok = g[~g.collision]
            yy = i + (k - 0.5) * 0.32
            if len(ok) == 0:
                ax.text(1.25, yy, "emergency: all collided", fontsize=6, color=RED, va="center"); continue
            ax.plot(ok.std_v_ss.mean(), yy, mk, color=c, ms=4, mec="k", mew=0.3,
                    label=("Nominal" if k == 0 else "Emergency braking") if i == 0 else None)
            if g.collision.any():
                ax.text(ok.std_v_ss.mean() + 0.1, yy, f"{int(g.collision.sum())}/10 collided", fontsize=6, color=RED,
                        va="center")
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=6.6); ax.invert_yaxis(); ax.set_xlim(0, 3.4)
    ax.set_xlabel("Wave amplitude (m/s)"); ax.set_title("(a) SUMO ring: waves"); ax.legend(fontsize=6, loc="lower right")
    ax = axs[1]
    hv = r[(r.method == "Human (EIDM)") & (r.scenario == "SUMO-ring nominal")].mean_v.mean()
    for i, m in enumerate(order):
        g = r[(r.method == m) & (r.scenario == "SUMO-ring nominal")]; ok = g[~g.collision]
        if len(ok) == 0:
            continue
        val = 100 * (ok.mean_v.mean() / hv - 1)
        ax.barh(i, val, color=BLUE if m == "CertiFlow-DR-FT" else ("#c6dbef" if "CertiFlow" in m or "CBF" in m or "shield" in m
                                                                   else GREY), edgecolor="k", lw=0.3, height=0.6)
        if g.collision.any():
            ax.text(-1 if val >= 0 else 1, i, "unsafe", fontsize=6, color=RED, va="center", ha="right" if val >= 0 else "left")
    ax.axvline(0, color="k", lw=0.6); ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis()
    ax.set_xlabel("Mean speed vs human-only (%)"); ax.set_title("(b) SUMO ring: speed (nominal)")
    ax = axs[2]
    hw = ["Human (EIDM)", "PI-Saturation", "PPO", "PPO+CF-shield (post-hoc)", "CertiFlow-DR", "CertiFlow-DR-FT"]
    hl = ["Human (EIDM)", "PI-Saturation", "PPO", "PPO+shield (post-hoc)", "CertiFlow w/o FT", "CertiFlow (ours)"]
    for i, m in enumerate(hw):
        g = h[h.method == m]; ce = g.coll_events.mean(); b = g.harsh_brake_human_per_h.mean()
        if ce > 0:
            ax.text(10, i, f"{ce:.0f} crashes / episode", fontsize=6.5, va="center", color=RED); continue
        ax.barh(i, b, color=BLUE if m == "CertiFlow-DR-FT" else ("#c6dbef" if "CertiFlow" in m or "shield" in m else GREY),
                edgecolor="k", lw=0.3, height=0.6)
        ax.text(b + 12, i, f"{b:.0f}", fontsize=6.5, va="center")
    ax.set_yticks(range(len(hw))); ax.set_yticklabels(hl, fontsize=6.6); ax.yaxis.tick_right(); ax.invert_yaxis()
    ax.set_xlim(0, 820); ax.set_xlabel("Human harsh braking (events/h)"); ax.set_title("(c) SUMO highway merge: safety")
    save(fig, "fig_sumo")


# ------------------------------------------------------------------ Fig. 7: safe-set phase portrait
def fig_phase():
    d = np.load("results/phase.npz")
    fig, axs = plt.subplots(1, 4, figsize=(FULLW, 2.55), gridspec_kw={"wspace": 0.3, "width_ratios": [1, 1, 0.05, 0.95]})
    cax = axs[2]; axs = [axs[0], axs[1], axs[3]]
    dv_grid = np.linspace(-6, 6, 300)
    for ax, lab, title in ((axs[0], "noshield", "(a) Policy without shield"), (axs[1], "shield", "(b) CertiFlow (with shield)")):
        g, dv, v, h = d[f"{lab}|gap"], d[f"{lab}|dv"], d[f"{lab}|v"], d[f"{lab}|h"]
        rng = np.random.default_rng(0); idx = rng.choice(len(g), size=min(len(g), 30000), replace=False)
        sc = ax.scatter(dv[idx], g[idx], c=np.clip(h[idx], -6, 6), cmap="RdBu", vmin=-6, vmax=6, s=0.6, lw=0,
                        rasterized=True)
        for vv, ls in ((3.0, ":"), (6.0, "--"), (9.0, "-.")):
            vl = vv - dv_grid
            s0 = 2.0 + 0.3 * vv + (vv ** 2 - np.maximum(vl, 0) ** 2) / 14.0
            ok = vl >= 0
            ax.plot(dv_grid[ok], s0[ok], "k", ls=ls, lw=0.8, label=f"$h=0$, $v$={vv:.0f} m/s")
        ax.axhline(0, color=RED, lw=0.9); ax.text(-5.8, 0.4, "collision", color=RED, fontsize=6.5)
        ax.set_xlim(-6, 6); ax.set_ylim(-5, 30)
        ax.set_xlabel(r"Closing speed $v-v_l$ (m/s)"); ax.set_title(title)
        frac = (h < 0).mean(); n_col = (g < 0).mean()
        ax.text(0.97, 0.95, f"$h<0$: {100*frac:.1f}% of states\ngap$<0$: {100*n_col:.2f}% of states", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.3, bbox=dict(fc="white", ec="#cccccc", lw=0.5))
    axs[0].set_ylabel("Gap to leader (m)")
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.36, 1.06))
    cb = fig.colorbar(sc, cax=cax); cb.set_label(r"Barrier value $h$ (m)", labelpad=1)
    pos = axs[2].get_position(); axs[2].set_position([pos.x0 + 0.075, pos.y0, pos.width - 0.075, pos.height])
    ax = axs[2]
    bins = np.linspace(-15, 25, 81)
    ax.hist(d["noshield|h"], bins=bins, color=GREY, alpha=0.7, density=True, label="w/o shield")
    ax.hist(d["shield|h"], bins=bins, color=BLUE, alpha=0.6, density=True, label="CertiFlow")
    ax.axvline(0, color=RED, ls="-.", lw=0.9)
    ax.set_xlabel(r"Barrier value $h$ (m)"); ax.set_ylabel("Density"); ax.set_title("(c) Histogram of $h$")
    ax.legend(fontsize=6)
    save(fig, "fig_phase")


# ------------------------------------------------------------------ Fig. 4: learning curves and SUMO fine-tuning
def fig_learning():
    fig, axs = plt.subplots(1, 4, figsize=(FULLW, 2.15), gridspec_kw={"wspace": 0.42})
    spec = [("PPO", "PPO", DGREY, "--"), ("PPO+CBF", "PPO+CBF", GREY, ":"), ("CertiFlow", "w/o DR", LIGHT, "-."),
            ("PPO-DR", "PPO-DR", ORANGE, "--"), ("CertiFlow-DR", "CertiFlow (DR stage)", BLUE, "-")]
    for m, lab, c, ls in spec:
        logs = [pd.read_csv(f) for f in sorted(glob.glob(f"runs/{m}_s[0-9]/log.csv"))]
        n = min(len(l) for l in logs); k = 10; x = logs[0].steps.values[:n] / 1e3
        for ax, col in zip(axs[:3], ("ep_return", "collisions_per_ep", "interv_rate")):
            arr = np.stack([np.convolve(np.nan_to_num(l[col].values[:n], nan=np.nanmean(l[col].values[:n])),
                                        np.ones(k) / k, mode="valid") for l in logs])
            mu = arr.mean(0); lo_, hi_ = arr.min(0), arr.max(0); xs = x[k - 1:]
            ax.plot(xs, mu, color=c, ls=ls, lw=1.0, label=lab)
            ax.fill_between(xs, lo_, hi_, color=c, alpha=0.12, lw=0)
    axs[0].set_title("(a) Episode return"); axs[0].set_ylabel("Return")
    axs[1].set_title("(b) Collisions per episode"); axs[1].set_ylabel("Collisions / episode")
    axs[2].set_title("(c) Shield intervention rate"); axs[2].set_ylabel("Intervention rate")
    for ax in axs[:3]:
        ax.set_xlabel("Control steps ($\\times10^3$)")
    axs[1].legend(fontsize=5.6, loc="upper right")
    ax = axs[3]
    for s in range(3):
        l = pd.read_csv(f"runs/CertiFlow-DR-FT_s{s}/log.csv")
        ax.plot(l.steps / 1e3, l.mean_v, color=BLUE, lw=0.9, alpha=0.4 + 0.2 * s, label=f"seed {s}")
    ax.set_title("(d) Safe fine-tuning in SUMO"); ax.set_xlabel("Control steps ($\\times10^3$)"); ax.set_ylabel("Mean speed (m/s)")
    tot = sum(int(pd.read_csv(f"runs/CertiFlow-DR-FT_s{s}/log.csv").collisions.iloc[-1]) for s in range(3))
    ax.text(0.97, 0.05, f"collisions during\nfine-tuning: {tot}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.3, bbox=dict(fc="white", ec="#cccccc", lw=0.5))
    ax.legend(fontsize=5.6, loc="upper left")
    save(fig, "fig_learning")


# ------------------------------------------------------------------ Fig. 8: STL robustness
def fig_robustness():
    E = pd.read_csv("results/episodes_main.csv")
    methods = ["Human (IDM)", "PI-Saturation", "PPO", "PPO-DR", "PPO+CBF", "PPO+CF-shield (post-hoc)", "CertiFlow-DR-FT"]
    labs = ["Human", "PI-Sat.", "PPO", "PPO-DR", "PPO+CBF", "PPO+shield", "Ours"]
    specs = [("rho_phi_safe", r"(a) $\rho(\varphi_{\mathrm{safe}})$ (m)"), ("rho_phi_ttc", r"(b) $\rho(\varphi_{\mathrm{ttc}})$ (m)"),
             ("rho_phi_damp", r"(c) $\rho(\varphi_{\mathrm{damp}})$ (m/s)")]
    fig, axs = plt.subplots(1, 3, figsize=(FULLW, 2.35), gridspec_kw={"wspace": 0.3})
    cols = {"S1-nominal": "#c6dbef", "S2-shift": "#6baed6", "S3-emergency": "#08519c"}
    for ax, (col, title) in zip(axs, specs):
        for j, (sc, c) in enumerate(cols.items()):
            data = [E[(E.method == m) & (E.scenario == sc)][col].clip(-6, 6).values for m in methods]
            pos = np.arange(len(methods)) + (j - 1) * 0.26
            bp = ax.boxplot(data, positions=pos, widths=0.22, patch_artist=True, showfliers=False,
                            medianprops=dict(color="k", lw=0.7), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                            boxprops=dict(lw=0.5))
            for b in bp["boxes"]:
                b.set_facecolor(c)
        ax.axhline(0, color=RED, ls="-.", lw=0.9)
        ax.set_xticks(range(len(methods))); ax.set_xticklabels(labs, rotation=35, ha="right", rotation_mode="anchor", fontsize=6.4)
        ax.set_title(title)
    axs[0].set_ylabel("Robustness (clipped to $\\pm$6)")
    axs[0].legend(handles=[Patch(fc=c, ec="k", lw=0.4, label=s.replace("-", " ")) for s, c in cols.items()], fontsize=5.8,
                  loc="lower left")
    save(fig, "fig_robustness")


# ------------------------------------------------------------------ Fig. 11: predictor residuals
def fig_residuals():
    r = np.load("results/residuals.npz")
    fig, axs = plt.subplots(1, 2, figsize=(COLW, 1.9), sharey=False, gridspec_kw={"wspace": 0.38})
    bins = np.linspace(-4, 4, 121)
    for ax, key, title in ((axs[0], "ring", "(a) SUMO ring"), (axs[1], "hw", "(b) SUMO highway")):
        for k, c, lab in (("idm", GREY, "IDM predictor"), ("persist", BLUE, "Persistence")):
            x = r[f"{key}_{k}"]
            ax.hist(x, bins=bins, density=True, histtype="stepfilled", alpha=0.45 if k == "idm" else 0.6, color=c, label=lab)
            lo, hi = np.quantile(x, 0.01), np.quantile(x, 0.99)
            for q in (lo, hi):
                ax.axvline(q, color=c, ls="-.", lw=0.8)
            ax.text(0.97, 0.93 if k == "idm" else 0.78, f"width {hi-lo:.2f}", transform=ax.transAxes, ha="right",
                    fontsize=6, color=c)
        ax.set_yscale("log"); ax.set_xlabel(r"Residual $\hat a-a$ (m/s$^2$)"); ax.set_title(title); ax.set_xlim(-4, 4)
    axs[0].set_ylabel("Density"); axs[0].legend(fontsize=5.5, loc="lower left")
    save(fig, "fig_residuals")


# ------------------------------------------------------------------ Fig. 12: SUMO space-time diagrams
def fig_sumo_spacetime():
    d = np.load("results/sumo_traces.npz"); L = float(d["L"])
    panels = [("Human", "nom", "(a) Human, nominal"), ("CertiFlow-DR-FT", "nom", "(b) Ours, nominal"),
              ("Human", "em", "(c) Human, emergency"), ("PPO", "em", "(d) PPO, emergency"),
              ("CertiFlow-DR-FT", "em", "(e) Ours, emergency")]
    fig, axs = plt.subplots(1, 5, figsize=(FULLW, 2.15), sharey=True, gridspec_kw={"wspace": 0.07})
    for ax, (m, tag, title) in zip(axs, panels):
        pos, v = d[f"{m}|{tag}|pos"], d[f"{m}|{tag}|v"]; T, N = v.shape; t = np.arange(T) * 0.1
        p = np.mod(pos, L)
        gap = np.mod(pos[:, np.r_[1:N, 0]] - pos, L) - 5.0; hit = np.where((gap < 0).any(1))[0]
        Tm = hit[0] + 1 if len(hit) else T
        for i in range(N):
            ax.scatter(t[:Tm:3], p[:Tm:3, i], c=v[:Tm:3, i], cmap=SPEED_CMAP, vmin=0, vmax=8, s=0.9, lw=0, rasterized=True)
        if len(hit):
            i0 = int(np.argmin(gap[hit[0]])); ax.plot(t[Tm - 1], p[Tm - 1, i0], "X", color="k", ms=6, mec="w", mew=0.7)
            ax.annotate(f"Collision at {t[Tm-1]:.1f} s", (t[Tm - 1], p[Tm - 1, i0]), xytext=(8, -14), textcoords="offset points",
                        fontsize=6.3)
        if tag == "em":
            for t0 in (20, 55):
                ax.axvline(t0, color="k", ls="--", lw=0.6)
        ax.set_title(title, fontsize=7.6); ax.set_xlim(0, 100); ax.set_ylim(0, L); ax.set_xlabel("Time (s)")
        ax.set_xticks([0, 50, 100]); ax.set_xticklabels(["0", "50", ""])
    axs[0].set_ylabel("Position on ring (m)")
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SPEED_CMAP, norm=plt.Normalize(0, 8)), ax=axs, fraction=0.014, pad=0.01)
    cb.set_label("Speed (m/s)"); cb.outline.set_linewidth(0.5)
    save(fig, "fig_sumo_spacetime")


FIGURES = ["learning", "spacetime", "timeseries", "phase", "robustness", "overall", "certificate", "residuals",
           "sumo_spacetime", "sumo", "uncertainty", "sensitivity", "coverage"]

if __name__ == "__main__":
    for n in (sys.argv[1:] or FIGURES):
        globals()[f"fig_{n}"]()
