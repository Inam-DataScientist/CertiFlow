"""LaTeX tables of the paper generated from the result files -> tables/

  tab_main.tex      Table 6   performance comparison on the ring road
  tab_verif.tex     Table 7   statistical verification of the STL specifications
  tab_sumo.tex      Table 8   transfer to SUMO
  tab_ablation.tex  Table 10  ablation study

Bold = best, underline = second best, ranked among methods that were collision-free in that scenario.
Usage:  python make_tables.py
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from certiflow.certify import clopper_pearson_upper as cp

OUT = "tables"; os.makedirs(OUT, exist_ok=True)
E = pd.read_csv("results/episodes_main.csv")
NAME = {"Human (IDM)": "Human drivers (IDM)", "PI-Saturation": "PI-Saturation", "PPO": "PPO", "PPO-DR": "PPO-DR",
        "PPO+CBF": "PPO + nominal CBF", "PPO+CF-shield (post-hoc)": "PPO + our shield (post hoc)",
        "CertiFlow-DR-FT": "CertiFlow (ours)"}
SCEN = [("S1-nominal", "S1 nominal"), ("S2-shift", "S2 distribution shift"), ("S3-emergency", "S3 emergency braking")]


def write(name, body):
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write(body)
    print("wrote", name)


def mark(vals, higher, eligible):
    """Return formatted strings with bold best / underline second among eligible entries."""
    idx = [i for i, (v, e) in enumerate(zip(vals, eligible)) if e and not pd.isna(v)]
    order = sorted(idx, key=lambda i: -vals[i] if higher else vals[i])
    best = order[0] if order else None
    second = next((i for i in order[1:] if not np.isclose(vals[i], vals[best])), None) if order else None
    ties = [i for i in order if best is not None and np.isclose(vals[i], vals[best])]
    return ties, ([second] if second is not None else [])


def fmt(v, d):
    return "--" if pd.isna(v) else f"{v:.{d}f}"


# ------------------------------------------------------------------ Table 6: main comparison
def tab_main():
    methods = ["Human (IDM)", "PI-Saturation", "PPO", "PPO-DR", "PPO+CBF", "PPO+CF-shield (post-hoc)", "CertiFlow-DR-FT"]
    cols = [("Coll.", "collision", "sum", False, 0), ("Gap$_{\\min}$", "min_gap", "mean", True, 2),
            ("TTC$_{\\min}$", "min_ttc", "mean", True, 2), ("Wave", "std_v_ss", "ok", False, 3),
            ("Speed", "mean_v", "ok", True, 2), ("Energy", "energy_MJkm", "ok", False, 3),
            ("$|$Jerk$|$", "cav_mean_abs_jerk", "mean", False, 2), ("$\\varphi_{\\mathrm{all}}$", "rho_phi_all", "sat", True, 2),
            ("Interv.", "interv_rate", "mean", False, 3)]
    lines = []
    for sk, sname in SCEN:
        lines.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{{sname}}}}}\\\\")
        vals = {c[1]: [] for c in cols}; elig = []
        for m in methods:
            g = E[(E.method == m) & (E.scenario == sk)]; ok = g[~g.collision]
            elig.append(int(g.collision.sum()) == 0)
            for _, col, how, _, _ in cols:
                if how == "sum":
                    v = g[col].sum()
                elif how == "ok":
                    v = ok[col].mean() if len(ok) else np.nan
                elif how == "sat":
                    v = (g[col] > 0).mean()
                else:
                    v = g[col].mean()
                if col == "interv_rate" and m in ("Human (IDM)", "PI-Saturation", "PPO", "PPO-DR"):
                    v = np.nan
                if col == "cav_mean_abs_jerk" and m == "Human (IDM)":
                    v = np.nan
                vals[col].append(v)
        cells = {m: [] for m in methods}
        for lab, col, how, higher, d in cols:
            if col == "collision":
                for m, v in zip(methods, vals[col]):
                    cells[m].append(f"{int(v)}/60" if v == 0 else f"\\textcolor{{red}}{{{int(v)}/60}}")
                continue
            b, s = mark(vals[col], higher, elig)
            for i, (m, v) in enumerate(zip(methods, vals[col])):
                t = fmt(v, d)
                if i in b and t != "--":
                    t = f"\\textbf{{{t}}}"
                elif i in s and t != "--":
                    t = f"\\underline{{{t}}}"
                cells[m].append(t)
        for m in methods:
            lines.append(f"\\quad {NAME[m]} & " + " & ".join(cells[m]) + "\\\\")
        if sk != "S3-emergency":
            lines.append("\\addlinespace[2pt]")
    head = " & ".join(c[0] for c in cols)
    body = r"""\begin{table*}[t]
\caption{Performance comparison on the ring road (2 CAVs out of 22; 3 training seeds $\times$ 20 test episodes per scenario). Coll.: episodes with a collision; Gap$_{\min}$ (m) and TTC$_{\min}$ (s): smallest gap and time-to-collision among all vehicles; Wave (m/s): steady-state speed standard deviation; Speed (m/s); Energy (MJ/km); $|$Jerk$|$ (m/s$^3$): mean absolute CAV jerk; $\varphi_{\mathrm{all}}$: share of episodes satisfying all STL specifications; Interv.: share of steps altered by the shield.}\label{tab:main}
\centering\footnotesize
\setlength{\tabcolsep}{4.2pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l ccccccccc@{}}
\toprule
Method & """ + head + r"""\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\par\smallskip\raggedright\scriptsize Note: Bold and underlined values are the best and second-best among methods that were collision-free in that scenario; metrics of colliding methods are computed over their collision-free episodes.
\end{table*}
"""
    write("tab_main.tex", body)


# ------------------------------------------------------------------ Table 10: ablation
def tab_ablation():
    rows = [("Full model (CertiFlow)", "CertiFlow-DR-FT", ""),
            ("\\textit{Training-time ablations}", None, ""),
            ("(i) w/o SUMO fine-tuning", "CertiFlow-DR", ""),
            ("(ii) w/o domain randomisation", "CertiFlow", "a"),
            ("(iii) (ii) + w/o STL reward", "CertiFlow-noSTL", "a"),
            ("(iv) (ii) + w/o cooperative term", "CertiFlow-noCoop", "a"),
            ("(v) (ii) + w/o ACI", "CertiFlow-noACI", "a"),
            ("\\textit{Test-time ablations of the full model}", None, ""),
            ("(vi) w/o shield", "Ours w/o shield", ""),
            ("(vii) w/o conformal bound (nominal CBF)", "Ours w/o conformal (nominal CBF)", ""),
            ("(viii) w/o cooperative term", "Ours w/o cooperative term", ""),
            ("(ix) w/o ACI (static conformal)", "Ours w/o ACI", "")]
    data = []
    for lab, m, _ in rows:
        if m is None:
            data.append(None); continue
        g = E[E.method == m]
        def S(s):
            return g[g.scenario == s]
        s1, s2, s3 = S("S1-nominal"), S("S2-shift"), S("S3-emergency")
        ok2 = s2[~s2.collision]; ok3 = s3[~s3.collision]
        data.append([int(g.collision.sum()), s1[~s1.collision].mean_v.mean(), ok2.std_v_ss.mean() if len(ok2) else np.nan,
                     (s2.rho_phi_all > 0).mean(), ok3.std_v_ss.mean() if len(ok3) else np.nan, s2.interv_rate.mean()])
    spec = [(0, False, 0), (1, True, 2), (2, False, 3), (3, True, 2), (4, False, 3), (5, False, 3)]
    elig = [d is not None and d[0] == 0 for d in data]
    marks = {}
    for j, higher, d in spec[1:]:
        vals = [np.nan if dd is None else dd[j] for dd in data]
        marks[j] = mark(vals, higher, elig)
    lines = []
    for i, ((lab, m, a), dd) in enumerate(zip(rows, data)):
        if dd is None:
            lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{{lab}}}\\\\"); continue
        cells = [f"{dd[0]}/180" if dd[0] == 0 else f"\\textcolor{{red}}{{{dd[0]}/180}}"]
        for j, higher, d in spec[1:]:
            t = fmt(dd[j], d)
            if m == "Ours w/o shield" and j == 5:
                t = "--"
            b, s = marks[j]
            if i in b and t != "--":
                t = f"\\textbf{{{t}}}"
            elif i in s and t != "--":
                t = f"\\underline{{{t}}}"
            cells.append(t)
        sup = "$^{a}$" if a else ""
        lines.append(f"{lab}{sup} & " + " & ".join(cells) + "\\\\")
    body = r"""\begin{table*}[t]
\caption{Ablation study (ring road, 2 CAVs; 60 episodes per scenario). Speed in S1; wave amplitude, STL satisfaction and intervention rate under distribution shift (S2); wave amplitude under emergency braking (S3).}\label{tab:ablation}
\centering\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l cccccc@{}}
\toprule
Configuration & Coll. (S1--S3) & Speed S1 (m/s) & Wave S2 (m/s) & $\varphi_{\mathrm{all}}$ S2 & Wave S3 (m/s) & Interv. S2\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\par\smallskip\raggedright\scriptsize Note: Bold and underlined values are the best and second-best among collision-free configurations. $^{a}$Trained without domain randomisation and without SUMO fine-tuning (i.e., on the nominal human model only).
\end{table*}
"""
    write("tab_ablation.tex", body)


# ------------------------------------------------------------------ Table 8: SUMO
def tab_sumo():
    r = pd.read_csv("results/sumo_ring_episodes.csv"); r["collision"] = r.collision.astype(bool)
    h = pd.read_csv("results/sumo_highway_episodes.csv")
    order = ["Human (EIDM)", "PI-Saturation", "PPO", "PPO-DR", "PPO+CBF", "PPO+CF-shield (post-hoc)", "CertiFlow",
             "CertiFlow-DR", "CertiFlow-DR-FT"]
    nm = {"Human (EIDM)": "Human drivers (EIDM)", "PI-Saturation": "PI-Saturation", "PPO": "PPO", "PPO-DR": "PPO-DR",
          "PPO+CBF": "PPO + nominal CBF", "PPO+CF-shield (post-hoc)": "PPO + our shield (post hoc)",
          "CertiFlow": "CertiFlow w/o DR", "CertiFlow-DR": "CertiFlow w/o fine-tuning", "CertiFlow-DR-FT": "CertiFlow (ours)"}
    lines = []
    for m in order:
        c = []
        for scen in ("SUMO-ring nominal", "SUMO-ring emergency"):
            g = r[(r.method == m) & (r.scenario == scen)]; ok = g[~g.collision]
            n = int(g.collision.sum())
            c.append(f"{n}/10" if n == 0 else f"\\textcolor{{red}}{{{n}/10}}")
            c.append(fmt(ok.std_v_ss.mean(), 3) if len(ok) else "--")
            if scen == "SUMO-ring nominal":
                c.append(fmt(ok.mean_v.mean(), 2) if len(ok) else "--")
                c.append(fmt(ok.energy_MJkm.mean(), 3) if len(ok) else "--")
        g = h[h.method == m]
        if len(g):
            ce = g.coll_events.mean()
            c.append("0" if ce == 0 else f"\\textcolor{{red}}{{{ce:.1f}}}")
            c.append(f"{g.harsh_brake_human_per_h.mean():.0f}" if ce == 0 else "--")
            c.append(f"{g.throughput_vph.mean():.0f}" if ce == 0 else "--")
        else:
            c += ["n/e"] * 3
        lines.append(f"{nm[m]} & " + " & ".join(c) + "\\\\")
    body = r"""\begin{table*}[t]
\caption{Zero-shot transfer to SUMO with EIDM human drivers (reaction time 0.8\,s, perception errors). Ring: 10 episodes per scenario; highway on-ramp merge: 3 episodes of 10\,min, 10\% CAVs. Only ``CertiFlow (ours)'' was fine-tuned in SUMO (ring only).}\label{tab:sumo}
\centering\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l cccc cc ccc@{}}
\toprule
 & \multicolumn{4}{c}{SUMO ring, nominal} & \multicolumn{2}{c}{SUMO ring, emergency} & \multicolumn{3}{c}{SUMO highway merge}\\
\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(l){8-10}
Method & Coll. & Wave & Speed & Energy & Coll. & Wave & Crashes/ep. & Harsh/h & Thr. (veh/h)\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\par\smallskip\raggedright\scriptsize Note: Harsh/h: human braking samples below $-4$\,m/s$^2$ per hour. Traffic metrics of crashing controllers are omitted because SUMO removes crashed vehicles. n/e: not evaluated.
\end{table*}
"""
    write("tab_sumo.tex", body)



# ------------------------------------------------------------------ Table 7: statistical verification
def tab_verification():
    methods = ["Human (IDM)", "PI-Saturation", "PPO", "PPO-DR", "PPO+CBF", "PPO+CF-shield (post-hoc)", "CertiFlow-DR-FT"]
    specs = [("rho_phi_safe", "$\\varphi_{\\mathrm{safe}}$"), ("rho_phi_ttc", "$\\varphi_{\\mathrm{ttc}}$"),
             ("rho_phi_damp", "$\\varphi_{\\mathrm{damp}}$"), ("rho_phi_comfort", "$\\varphi_{\\mathrm{comf}}$"),
             ("rho_phi_all", "$\\varphi_{\\mathrm{all}}$")]
    lines = []
    for m in methods:
        g = E[E.method == m]; n = len(g); cells = []
        for col, _ in specs:
            k = int((g[col] <= 0).sum()); ub = 100 * cp(k, n)
            txt = f"{k} ({ub:.1f})"
            cells.append(f"\\textbf{{{txt}}}" if k == 0 else txt)
        lines.append(f"{NAME[m]} & " + " & ".join(cells) + "\\\\")
    body = r"""\begin{table*}[!t]
\caption{Statistical verification over all 180 test episodes (S1--S3) per method: number of episodes violating each STL specification and, in parentheses, the exact 95\% Clopper--Pearson upper bound on the violation probability (\%).}\label{tab:verif}
\centering\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l ccccc@{}}
\toprule
Method & """ + " & ".join(s[1] for s in specs) + r"""\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\par\smallskip\raggedright\scriptsize Note: Bold marks specifications never violated. Almost all $\varphi_{\mathrm{ttc}}$ violations occur in S3, where the 6\,m/s$^2$ brakes create closing speeds that a stopping-distance barrier does not exclude (Section~\ref{sec:lim}).
\end{table*}
"""
    write("tab_verif.tex", body)


if __name__ == "__main__":
    tab_main(); tab_verification(); tab_sumo(); tab_ablation()
