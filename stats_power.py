"""Statistical power and uncertainty of the ring-road comparison (response to review).

  * paired comparisons CertiFlow vs each collision-free baseline (common random numbers: the same evaluation
    seeds give the same initial states, human parameters and noise to every method), with a two-level
    (seed -> episode) bootstrap 95% CI, per-seed mean differences and a Wilcoxon signed-rank test;
  * exact Clopper-Pearson bounds and detection power for collisions as a function of the number of episodes;
  * exact Poisson bounds for the SUMO highway merge (3 episodes per method).

Usage:  python stats_power.py
Output: tables/tab_stats.tex, figures/fig_power.{pdf,png}, results/stats_power.json
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import make_figures as MF                       # shared style
from certiflow.certify import clopper_pearson_upper as cp

E = pd.read_csv("results/episodes_main.csv")
OURS = "CertiFlow-DR-FT"
BASE = [("PI-Saturation", "PI-Saturation"), ("PPO+CBF", "PPO + nominal CBF"), ("PPO+CF-shield (post-hoc)", "PPO + our shield")]
SCEN = [("S1-nominal", "S1"), ("S2-shift", "S2"), ("S3-emergency", "S3")]
METRICS = [("std_v_ss", "Wave (m/s)"), ("mean_v", "Speed (m/s)"), ("energy_MJkm", "Energy (MJ/km)")]
rng = np.random.default_rng(2026)
B = 4000


def paired(scen, base, col):
    a = E[(E.method == OURS) & (E.scenario == scen)].sort_values(["seed"], kind="stable")
    b = E[(E.method == base) & (E.scenario == scen)].sort_values(["seed"], kind="stable")
    d = a[col].to_numpy() - b[col].to_numpy()
    seeds = a.seed.to_numpy()
    groups = [d[seeds == s] for s in np.unique(seeds)]
    # two-level bootstrap: resample seeds, then episodes within each resampled seed
    boots = np.empty(B)
    for i in range(B):
        gs = [groups[j] for j in rng.integers(0, len(groups), len(groups))]
        boots[i] = np.mean(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in gs]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    per_seed = [float(g.mean()) for g in groups]
    p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), per_seed=per_seed, p=float(p), sd=float(d.std(ddof=1)),
                n=len(d))


def sg(x, d):
    """Signed number with a typographic minus; values that round to zero are printed without sign."""
    if round(x, d) == 0:
        return f"{0:.{d}f}"
    return ("$-$" if x < 0 else "$+$") + f"{abs(x):.{d}f}"


def fmt_p(p):
    return "$<10^{-4}$" if p < 1e-4 else f"{p:.3f}"


def tab_stats(res):
    lines = []
    for sk, sn in SCEN:
        first = True
        for bk, bn in BASE:
            cells = []
            for col, _ in METRICS:
                r = res[(sk, bk, col)]
                better = (r["hi"] < 0) if col != "mean_v" else (r["lo"] > 0)
                worse = (r["lo"] > 0) if col != "mean_v" else (r["hi"] < 0)
                d = 2 if col == "mean_v" else 3
                t = f"{sg(r['mean'], d)} [{sg(r['lo'], d)}, {sg(r['hi'], d)}]"
                if better:
                    t = f"\\textbf{{{t}}}"
                cells.append(t)
                cells.append(f"{sum((x < 0) if col != 'mean_v' else (x > 0) for x in r['per_seed'])}/3")
            lines.append((f"\\multirow{{3}}{{*}}{{{sn}}}" if first else "") + f" & {bn} & " + " & ".join(cells) + "\\\\")
            first = False
        if sk != "S3-emergency":
            lines.append("\\addlinespace[2pt]")
    body = r"""\begin{table*}[!t]
\caption{Paired comparison of CertiFlow with the collision-free baselines (difference CertiFlow $-$ baseline; 300 paired episodes per scenario, 3 training seeds $\times$ 100 evaluation episodes that share initial states, human parameters and noise across methods). Brackets: 95\% two-level bootstrap interval (seeds, then episodes within seeds); ``Seeds'': number of training seeds whose mean difference favours CertiFlow. Bold: interval excludes zero in favour of CertiFlow.}\label{tab:stats}
\centering\footnotesize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll cc cc cc@{}}
\toprule
 & & \multicolumn{2}{c}{$\Delta$Wave (m/s), lower is better} & \multicolumn{2}{c}{$\Delta$Speed (m/s), higher is better} & \multicolumn{2}{c}{$\Delta$Energy (MJ/km), lower is better}\\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}
Sc. & Baseline & Mean [95\% CI] & Seeds & Mean [95\% CI] & Seeds & Mean [95\% CI] & Seeds\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""
    open("tables/tab_stats.tex", "w").write(body)
    print("wrote tab_stats.tex")


def fig_power(res, hw):
    fig, axs = plt.subplots(1, 3, figsize=(MF.FULLW, 2.3), gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1, 1.25]})
    # (a) Clopper-Pearson bound with zero failures
    ax = axs[0]
    n = np.unique(np.logspace(1, 4.3, 200).astype(int))
    ax.loglog(n, [100 * cp(0, k) for k in n], color=MF.BLUE, label="exact, $k=0$")
    ax.loglog(n, 100 * 3 / n, color=MF.GREY, ls="--", lw=0.8, label="rule of three, $3/n$")
    for k, lab, off in ((60, "60", (6, 1)), (300, "300", (6, 1)), (900, "900", (6, 1)), (9900, "9900", (-8, -12))):
        ax.plot(k, 100 * cp(0, k), "o", color=MF.RED if k == 60 else MF.BLUE, ms=3.5, zorder=3)
        ax.annotate(lab, (k, 100 * cp(0, k)), xytext=off, textcoords="offset points", fontsize=5.6)
    ax.set_xlabel("Collision-free test episodes $n$"); ax.set_ylabel("95% upper bound on $p$ (%)")
    ax.set_title("(a) Bound from zero collisions"); ax.legend(fontsize=5.6, loc="upper right")
    # (b) probability of observing at least one collision
    ax = axs[1]
    p = np.logspace(-4, -0.3, 300)
    for k, c, ls in ((60, MF.RED, ":"), (300, MF.LIGHT, "--"), (900, MF.BLUE, "-"), (9900, MF.DGREY, "-.")):
        ax.semilogx(100 * p, 100 * (1 - (1 - p) ** k), color=c, ls=ls, label=f"$n={k}$")
    ax.axhline(95, color=MF.GREY, lw=0.6)
    ax.text(0.012, 88, "95%", fontsize=5.8, color=MF.DGREY)
    ax.set_xlabel("True collision probability per episode (%)"); ax.set_ylabel("P(at least one collision) (%)")
    ax.set_title("(b) Power to reveal collisions"); ax.legend(fontsize=5.6, loc="center right")
    ax.set_xlim(0.01, 50)
    # (c) paired wave differences, per seed and pooled
    ax = axs[2]
    cols = {"PI-Saturation": MF.GREY, "PPO+CBF": MF.DGREY, "PPO+CF-shield (post-hoc)": MF.LIGHT}
    labs = {"PI-Saturation": "vs PI-Saturation", "PPO+CBF": "vs PPO + nominal CBF", "PPO+CF-shield (post-hoc)": "vs PPO + our shield"}
    for i, (sk, sn) in enumerate(SCEN):
        for j, (bk, _) in enumerate(BASE):
            r = res[(sk, bk, "std_v_ss")]
            ref = E[(E.method == bk) & (E.scenario == sk)].std_v_ss.mean() / 100.0
            y = i + (j - 1) * 0.24
            ax.plot([r["lo"] / ref, r["hi"] / ref], [y, y], color=cols[bk], lw=1.8, solid_capstyle="butt")
            ax.plot(r["mean"] / ref, y, "o", color=cols[bk], ms=3.8, mec="k", mew=0.4, label=labs[bk] if i == 0 else None)
            ax.plot(np.array(r["per_seed"]) / ref, [y] * 3, "|", color="k", ms=5, mew=0.7)
    ax.axvline(0, color=MF.RED, lw=0.8, ls="-.")
    ax.set_yticks(range(3)); ax.set_yticklabels(["S1", "S2", "S3"]); ax.invert_yaxis()
    ax.set_xlim(-75, 8)
    ax.set_xlabel(r"$\Delta$wave, CertiFlow vs baseline (% of baseline)")
    ax.set_title("(c) Paired wave difference (95% CI)")
    from matplotlib.lines import Line2D
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], marker="|", color="k", ls="none", ms=5, mew=0.7)); l.append("per-seed mean")
    ax.legend(h, l, fontsize=5.5, loc="upper left")
    MF.save(fig, "fig_power")


def main():
    res = {}
    for sk, _ in SCEN:
        for bk, _ in BASE:
            for col, _ in METRICS:
                res[(sk, bk, col)] = paired(sk, bk, col)
    tab_stats(res)
    # SUMO highway: exact Poisson analysis of crash counts (3 episodes of 10 min per method)
    h = pd.read_csv("results/sumo_highway_episodes.csv")
    hw = {}
    for m, g in h.groupby("method"):
        k = int(g.coll_events.sum()); n = len(g)
        up = stats.chi2.ppf(0.95, 2 * (k + 1)) / 2 / n
        lo = stats.chi2.ppf(0.025, 2 * k) / 2 / n if k > 0 else 0.0
        hw[m] = dict(events=k, episodes=n, rate=k / n, rate_upper95=up, rate_lower95=lo,
                     throughput=g.throughput_vph.tolist(), harsh=g.harsh_brake_human_per_h.tolist())
    # exact conditional test: unshielded (PPO) vs shielded (CertiFlow), equal exposure
    k1 = hw["PPO"]["events"]; k2 = hw[OURS]["events"]
    hw["test_PPO_vs_ours_p"] = float(stats.binomtest(k2, k1 + k2, 0.5, alternative="less").pvalue)
    fig_power(res, hw)
    # summary numbers used in the text
    out = {f"{a}|{b}|{c}": v for (a, b, c), v in res.items()}
    out["highway"] = hw
    out["cp"] = {str(k): 100 * cp(0, k) for k in (60, 180, 300, 900, 1980, 9900)}
    out["detect"] = {f"{k}@{p}": 100 * (1 - (1 - p) ** k) for k in (60, 300, 900) for p in (0.01, 0.005, 0.0033)}
    # seed-level spread of the main metrics of CertiFlow
    g = E[E.method == OURS].groupby(["scenario", "seed"])[["std_v_ss", "mean_v", "energy_MJkm"]].mean()
    out["ours_per_seed"] = {f"{s}|{sd}": r.to_dict() for (s, sd), r in g.iterrows()}
    json.dump(out, open("results/stats_power.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k in ("cp", "detect")}, indent=1))
    print(json.dumps({m: {kk: vv for kk, vv in d.items()} if isinstance(d, dict) else d for m, d in hw.items()}, indent=1))


if __name__ == "__main__":
    main()
