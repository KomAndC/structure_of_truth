"""Figures for sim/power.py, rebuilt from the saved CSVs (no simulation).

Usage: python sim/power_figures.py
Reads results/power/{p1,p2,p2_reps}.csv and power.json; writes power.png, p1_gradient.png,
p2_rank.png, p2_false_low.png into results/power/.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from structure_of_truth.io import save_figure  # noqa: E402

NAME = "power"
OUT = Path(__file__).resolve().parents[1] / "results" / NAME
RANKS = ("3", "5", "10", "full")
METHODS = ("var80", "scree", "parallel", "cv", "retest_cv", "retest_share")


def _load(p):
    rows = []
    for r in csv.DictReader(open(p)):
        d = {}
        for k, v in r.items():
            if k in ("rank", "method", "scenario"):
                d[k] = v
                continue
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
        rows.append(d)
    return rows


def heat(ax, M, rows, cols, title, ylabel=None, xlabel="judgments per concept per subject (m)"):
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", origin="lower", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if M[i, j] > 0.6 else "white")
    ax.set_xticks(range(len(cols)), [str(int(c)) for c in cols])
    ax.set_yticks(range(len(rows)), [str(int(r)) for r in rows])
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    return im


def p2_class(r):
    rh = int(r["rhat"])
    if r["method"] == "retest_share":
        if r["inconclusive"] == "True":
            return "inconclusive"
        truth_low = r["rank"] in ("3", "5")
        sep = r["separated"] == "True"
        return ("low" if sep else "high") if truth_low else ("high" if sep else "low")
    return "low" if 1 <= rh <= 6 else ("high" if rh >= 8 else "inconclusive")


def main():
    cfg = json.load(open(OUT / "power.json"))["config"]
    noise, ms, S1, S2 = cfg["noise"], cfg["m"], cfg["S1"], cfg["S2"]
    p1 = _load(OUT / "p1.csv")
    p2 = _load(OUT / "p2.csv")

    def p1cell(sc, nz, met):
        return np.array([[next(r[met] for r in p1 if r["scenario"] == sc and r["noise"] == nz and r["S"] == S and r["m"] == m)
                          for m in ms] for S in S1])

    def p2sep(meth, nz, ranks=RANKS):
        return np.array([[min(next(r["separated"] for r in p2 if r["method"] == meth and r["noise"] == nz and r["S"] == S
                                   and r["m"] == m and r["rank"] == k) for k in ranks) for m in ms] for S in S2])

    # --- power overview
    fig, axes = plt.subplots(len(noise), 4, figsize=(19, 4.3 * len(noise)), squeeze=False, constrained_layout=True)
    for i, nz in enumerate(noise):
        heat(axes[i, 0], p1cell("gradient", nz, "sig_corr"), S1, ms, f"P1 power, strong gradient\n(corrected test), noise {nz}", "subjects per culture")
        heat(axes[i, 1], p1cell("weak_gradient", nz, "sig_corr"), S1, ms, f"P1 power, weak gradient\n(corrected test), noise {nz}", "subjects per culture")
        heat(axes[i, 2], p1cell("equal", nz, "sig_raw"), S1, ms, f"P1 false positives, NO true gradient\n(RAW test), noise {nz}", "subjects per culture")
        im = heat(axes[i, 3], p2sep("retest_share", nz), S2, ms, f"P2: all 4 ranks separated low/high\n(retest_share, 2 sessions), noise {nz}", "subjects (total)")
    fig.colorbar(im, ax=axes, shrink=0.5, label="proportion of replicates")
    fig.suptitle("Power at 80%: target cells ≥ 0.80 (P1 false positives should be ≤ 0.05)")
    save_figure(NAME, fig)

    # --- P2 all methods
    fig, axes = plt.subplots(len(noise), len(METHODS), figsize=(3.6 * len(METHODS), 3.9 * len(noise)), squeeze=False, constrained_layout=True)
    for i, nz in enumerate(noise):
        for j, meth in enumerate(METHODS):
            im = heat(axes[i, j], p2sep(meth, nz), S2, ms, f"{meth}, noise {nz}", "subjects" if j == 0 else None, "m")
    fig.colorbar(im, ax=axes, shrink=0.5)
    fig.suptitle("P2: P(all four true ranks 3, 5, 10, full classified correctly as low / high)")
    save_figure(NAME, fig, stem="p2_rank")

    # --- P2 false "low-rank" verdicts for a rank-10 truth (false support for P2)
    reps = list(csv.DictReader(open(OUT / "p2_reps.csv")))
    fl = defaultdict(list)
    for r in reps:
        if r["rank"] == "10":
            fl[(r["method"], float(r["noise"]), int(r["S"]), int(r["m"]))].append(p2_class(r) == "low")
    meths = ("scree", "parallel", "cv", "retest_share")
    fig, axes = plt.subplots(len(noise), len(meths), figsize=(3.8 * len(meths), 3.9 * len(noise)), squeeze=False, constrained_layout=True)
    for i, nz in enumerate(noise):
        for j, meth in enumerate(meths):
            M = np.array([[np.mean(fl[(meth, nz, S, m)]) for m in ms] for S in S2])
            im = heat(axes[i, j], M, S2, ms, f"{meth}, noise {nz}", "subjects" if j == 0 else None, "m")
    fig.colorbar(im, ax=axes, shrink=0.5)
    fig.suptitle("P2 hazard: P(true rank 10 called 'low-rank (≤ 6)') — false support for prediction 2")
    save_figure(NAME, fig, stem="p2_false_low")

    # --- P1 gradient profile
    scen = ("equal", "weak_gradient", "gradient")
    fig, axes = plt.subplots(len(noise), len(scen), figsize=(4.6 * len(scen), 3.5 * len(noise)), squeeze=False, sharey=True, constrained_layout=True)
    for i, nz in enumerate(noise):
        for j, sc in enumerate(scen):
            ax = axes[i, j]
            for S, ls in ((min(S1), "-"), (max(S1), ":")):
                r = next(x for x in p1 if x["scenario"] == sc and x["noise"] == nz and x["S"] == S and x["m"] == max(ms))
                ax.plot([0, 1, 2], [r["raw_phys"], r["raw_inter"], r["raw_value"]], "o", ls=ls, color="tab:red", label=f"raw, S={int(S)}")
                ax.plot([0, 1, 2], [r["corr_phys"], r["corr_inter"], r["corr_value"]], "s", ls=ls, color="tab:blue", label=f"corrected, S={int(S)}")
            r = next(x for x in p1 if x["scenario"] == sc and x["noise"] == nz and x["S"] == max(S1) and x["m"] == max(ms))
            ax.plot([0, 1, 2], [r["true_phys"], r["true_inter"], r["true_value"]], "k^-", lw=2, label="true")
            ax.set_xticks([0, 1, 2], ["physical", "intersubj.", "value"])
            ax.set_title(f"{sc} τ, noise {nz}, m = {max(ms)} (full matrix)", fontsize=9)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel("cross-cultural convergence r")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("P1: raw convergence shows a layer gradient even with none (left); disattenuation removes it but over-corrects when reliability is low")
    save_figure(NAME, fig, stem="p1_gradient")
    print("figures written")


if __name__ == "__main__":
    main()
