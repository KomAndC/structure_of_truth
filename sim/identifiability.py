"""Can the pipeline tell rotation (A) from rescaling (B), and is prediction 3 testable?

Part 1 — identifiability sweep
    For every noise level x number of subjects x generator in {null, A, B, C} and
    ``--reps`` replicates: simulate one measurement per subject (sim/generator.py), estimate
    per-subject rotational (θ̂) and rescaling (b̂) magnitudes by the INDSCAL and GPA routes
    (structure_of_truth.decompose), and score against ground truth:
      * r_rot / r_resc: Spearman correlation across subjects of estimate vs truth;
      * leak_rot_from_resc: corr(θ̂, b_true) — rotation estimate contaminated by rescaling
        (computed where b_true varies); leak_resc_from_rot: corr(b̂, θ_true);
      * detection: population mean θ̂ (b̂) exceeds a threshold calibrated on the *null*
        generator in the same noise x S cell (mean + 1.645 sd over null replicates,
        leave-one-out for the null replicates themselves); the detected set is compared
        with the true set, giving a generator-classification accuracy.

Part 2 — clarification intervention (prediction 3)
    Generator C (both components present). Every subject is measured twice. Between the
    measurements a "clarification" multiplies the rotation angle and/or the log weights by
    (1 - f), f = ``--reduction`` (default 0.5):
      rotation_only (the P3 prediction), rescaling_only, both_equal, none (control).
    INDSCAL/GPA are fitted jointly to the 2S matrices. Per replicate, one-sided Wilcoxon
    signed-rank tests over subjects:
      p_rot: θ̂_pre > θ̂_post;   p_resc: b̂_pre > b̂_post;
      p_diff: relative drop of θ̂ > relative drop of b̂ (per subject, each drop divided by
              the component's mean pre value).
    "P3 signature" = p_rot < .05 and p_diff < .05. "Naive signature" = p_rot < .05 and
    p_resc >= .05 (the literal "rotation drops, weights don't", which rewards low power).

Usage:  python sim/identifiability.py --seed 20261103 [--quick] [--workers 4]
Outputs: results/identifiability/{identifiability,identifiability_reps,intervention,
intervention_reps}.csv, identifiability.json, identifiability.png, intervention.png,
tables.md.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

from generator import SimConfig, make_population, measure, true_magnitudes  # noqa: E402
from structure_of_truth.decompose import estimate_components  # noqa: E402
from structure_of_truth.io import save_figure, write_csv, write_json  # noqa: E402
from structure_of_truth.seeds import make_rng  # noqa: E402

NAME = "identifiability"
GENS = ("null", "A", "B", "C")
SCENARIOS = ("rotation_only", "rescaling_only", "both_equal", "none")
ESTIMATORS = ("indscal", "gpa")
INDSCAL_KW = dict(tol=1e-7, max_iter=500)


def cell_seed(master: int, *key: int) -> int:
    return int(np.random.SeedSequence(master, spawn_key=tuple(key)).generate_state(1)[0])


def _spear(x, y) -> float:
    if np.ptp(y) == 0 or np.ptp(x) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _reliability(D: np.ndarray, D_true: np.ndarray) -> tuple[float, float]:
    """Mean single-subject r with its own noise-free matrix (oracle) and Nili lower bound."""
    iu = np.triu_indices(D.shape[1], 1)
    V, T = D[:, iu[0], iu[1]], D_true[:, iu[0], iu[1]]
    oracle = np.mean([np.corrcoef(v, t)[0, 1] for v, t in zip(V, T)])
    tot = V.sum(axis=0)
    lower = np.mean([np.corrcoef(v, (tot - v) / (len(V) - 1))[0, 1] for v in V])
    return float(oracle), float(lower)


def _estimates(c, est: str):
    return (c.theta_indscal, c.b_indscal) if est == "indscal" else (c.theta_gpa, c.b_gpa)


# ---------------------------------------------------------------- part 1
def run_identifiability_task(args):
    master, i_n, noise, i_s, S, i_g, gen, rep, layer_noise = args
    seed = cell_seed(master, 1, i_n, i_s, i_g, rep)
    rng = make_rng(seed)
    pop = make_population(SimConfig(n_subjects=S, generator=gen, noise=noise, layer_noise=layer_noise), rng)
    D, D_true = measure(pop, rng)
    th, b = true_magnitudes(pop)
    c = estimate_components(D, pop.config.k, block=pop.value_idx, anchors=pop.anchor_idx, indscal_kwargs=INDSCAL_KW)
    oracle, lower = _reliability(D, D_true)
    rows = []
    for est in ESTIMATORS:
        th_hat, b_hat = _estimates(c, est)
        rows.append(dict(
            noise=noise, S=S, generator=gen, rep=rep, estimator=est, task_seed=seed,
            mean_theta_hat=float(th_hat.mean()), mean_b_hat=float(b_hat.mean()),
            mean_theta_true=float(th.mean()), mean_b_true=float(b.mean()),
            r_rot=_spear(th_hat, th), r_resc=_spear(b_hat, b),
            leak_rot_from_resc=_spear(th_hat, b), leak_resc_from_rot=_spear(b_hat, th),
            single_subject_r=oracle, nili_lower=lower,
            indscal_converged=bool(c.indscal.converged), indscal_iter=int(c.indscal.n_iter),
        ))
    return rows


def add_detection(rows: list[dict]) -> None:
    """Null-calibrated detection of each component; modifies rows in place."""
    from collections import defaultdict

    null = defaultdict(list)
    for r in rows:
        if r["generator"] == "null":
            null[(r["noise"], r["S"], r["estimator"])].append(r)
    for r in rows:
        ref = [x for x in null[(r["noise"], r["S"], r["estimator"])] if x is not r]
        for comp, key in (("rot", "mean_theta_hat"), ("resc", "mean_b_hat")):
            vals = np.array([x[key] for x in ref])
            thr = vals.mean() + 1.645 * vals.std(ddof=1)
            r[f"thr_{comp}"] = float(thr)
            r[f"detect_{comp}"] = bool(r[key] > thr)
        truth = {"null": (False, False), "A": (True, False), "B": (False, True), "C": (True, True)}[r["generator"]]
        r["classified_correctly"] = (r["detect_rot"], r["detect_resc"]) == truth


def summarise(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    groups: dict = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for g, rs in groups.items():
        row = dict(zip(keys, g))
        row["n_reps"] = len(rs)
        for m in metrics:
            v = np.array([float(x[m]) for x in rs], dtype=float)
            v = v[~np.isnan(v)]
            row[m] = float(v.mean()) if len(v) else float("nan")
            if m.startswith(("r_", "leak_")):
                row[m + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        out.append(row)
    return out


# ---------------------------------------------------------------- part 2
def run_intervention_task(args):
    master, i_n, noise, i_s, S, i_c, scen, rep, reduction, layer_noise = args
    seed = cell_seed(master, 2, i_n, i_s, i_c, rep)
    rng = make_rng(seed)
    pop = make_population(SimConfig(n_subjects=S, generator="C", noise=noise, layer_noise=layer_noise), rng)
    f_rot = reduction if scen in ("rotation_only", "both_equal") else 0.0
    f_resc = reduction if scen in ("rescaling_only", "both_equal") else 0.0
    theta_post, logw_post = pop.theta * (1 - f_rot), pop.log_w * (1 - f_resc)
    D_pre, _ = measure(pop, rng)
    D_post, _ = measure(pop, rng, theta=theta_post, log_w=logw_post)
    c = estimate_components(
        np.concatenate([D_pre, D_post]), pop.config.k,
        block=pop.value_idx, anchors=pop.anchor_idx, indscal_kwargs=INDSCAL_KW,
    )
    th_pre_true, b_pre_true = true_magnitudes(pop)
    th_post_true, b_post_true = true_magnitudes(pop, theta_post, logw_post)
    rows = []
    for est in ESTIMATORS:
        th_hat, b_hat = _estimates(c, est)
        tp, tq, bp, bq = th_hat[:S], th_hat[S:], b_hat[:S], b_hat[S:]
        drot = (tp - tq) / tp.mean()
        dresc = (bp - bq) / bp.mean()
        p_rot = float(wilcoxon(tp - tq, alternative="greater").pvalue)
        p_resc = float(wilcoxon(bp - bq, alternative="greater").pvalue)
        p_diff = float(wilcoxon(drot - dresc, alternative="greater").pvalue)
        p_diff_two = float(wilcoxon(drot - dresc).pvalue)
        rows.append(dict(
            noise=noise, S=S, scenario=scen, rep=rep, estimator=est, task_seed=seed, reduction=reduction,
            rel_drop_rot_hat=float(drot.mean()), rel_drop_resc_hat=float(dresc.mean()),
            rel_drop_rot_true=float(1 - th_post_true.mean() / th_pre_true.mean()),
            rel_drop_resc_true=float(1 - b_post_true.mean() / b_pre_true.mean()),
            p_rot=p_rot, p_resc=p_resc, p_diff=p_diff,
            rot_drop_sig=p_rot < 0.05, resc_drop_sig=p_resc < 0.05,
            p3_signature=(p_rot < 0.05) and (p_diff < 0.05),
            naive_signature=(p_rot < 0.05) and (p_resc >= 0.05),
            both_drop_equal=(p_rot < 0.05) and (p_resc < 0.05) and (p_diff_two >= 0.05),
        ))
    return rows


# ---------------------------------------------------------------- figures / tables
def plot_identifiability(summary, noises, Ss):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    cmap = plt.get_cmap("viridis")
    for ei, est in enumerate(ESTIMATORS):
        panels = [
            ("r_rot", "C", "rotation recovery r(θ̂, θ)  [gen C]"),
            ("r_resc", "C", "rescaling recovery r(b̂, b)  [gen C]"),
            ("classified_correctly", None, "generator classification accuracy (mean over A,B,C,null)"),
        ]
        for pi, (metric, gen, title) in enumerate(panels):
            ax = axes[ei, pi]
            for ni, nz in enumerate(noises):
                ys = []
                for S in Ss:
                    rs = [r for r in summary if r["estimator"] == est and r["noise"] == nz and r["S"] == S
                          and (gen is None or r["generator"] == gen)]
                    ys.append(np.nanmean([r[metric] for r in rs]))
                ax.plot(Ss, ys, "o-", color=cmap(ni / max(len(noises) - 1, 1)), label=f"noise {nz}")
            ax.set_xscale("log")
            ax.minorticks_off()
            ax.set_xticks(Ss, [str(s) for s in Ss])
            ax.set_ylim(-0.05 if metric != "classified_correctly" else 0, 1.02)
            if metric == "classified_correctly":
                ax.axhline(0.25, color="grey", lw=0.8, ls=":")
            ax.set_title(f"{est.upper()}: {title}", fontsize=9)
            ax.grid(alpha=0.3)
            if ei == 1:
                ax.set_xlabel("subjects S")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Identifiability of rotational vs rescaling components (mean over replicates)")
    fig.tight_layout()
    return fig


def plot_intervention(summary, noises, Ss):
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(15, 3.8), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ci, scen in enumerate(SCENARIOS):
        ax = axes[ci]
        for ni, nz in enumerate(noises):
            ys = [next(r["p3_signature"] for r in summary
                       if r["estimator"] == "indscal" and r["scenario"] == scen and r["noise"] == nz and r["S"] == S)
                  for S in Ss]
            ax.plot(Ss, ys, "o-", color=cmap(ni / max(len(noises) - 1, 1)), label=f"noise {nz}")
        ax.axhline(0.05, color="grey", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(Ss, [str(s) for s in Ss])
        ax.set_ylim(0, 1.02)
        ax.set_title(f"clarification: {scen}", fontsize=9)
        ax.set_xlabel("subjects S")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("P(P3 signature detected)  [INDSCAL]")
    axes[0].legend(fontsize=8)
    fig.suptitle("Prediction 3 signature (rotation drops more than weights): want high only for rotation_only")
    fig.tight_layout()
    return fig


def _fmt(x) -> str:
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, float):
        return "nan" if np.isnan(x) else f"{x:.2f}"
    return str(x)


def md_table(rows, cols) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--S", type=int, nargs="+", default=[20, 50, 100, 200])
    ap.add_argument("--layer-noise", type=float, nargs=3, default=[0.5, 1.0, 1.5])
    ap.add_argument("--reduction", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (overrides grid)")
    ap.add_argument("--skip-intervention", action="store_true")
    a = ap.parse_args(argv)
    if a.quick:
        a.reps, a.noise, a.S = 4, [0.1, 0.5], [20, 50]
    ln = tuple(a.layer_noise)
    t0 = time.time()

    tasks = [(a.seed, i_n, nz, i_s, S, i_g, g, rep, ln)
             for (i_n, nz), (i_s, S), (i_g, g), rep in product(enumerate(a.noise), enumerate(a.S), enumerate(GENS), range(a.reps))]
    tasks.sort(key=lambda t: -t[4])  # big jobs first for load balance
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(run_identifiability_task, tasks, chunksize=1) for r in rs]
    add_detection(rows)
    rows.sort(key=lambda r: (r["estimator"], r["noise"], r["S"], GENS.index(r["generator"]), r["rep"]))
    metrics = ("r_rot", "r_resc", "leak_rot_from_resc", "leak_resc_from_rot", "detect_rot", "detect_resc",
               "classified_correctly", "mean_theta_hat", "mean_b_hat", "mean_theta_true", "mean_b_true",
               "single_subject_r", "nili_lower", "indscal_converged")
    summary = summarise(rows, ("estimator", "noise", "S", "generator"), metrics)
    write_csv(NAME, rows, seed=a.seed, stem="identifiability_reps")
    write_csv(NAME, summary, seed=a.seed)
    save_figure(NAME, plot_identifiability(summary, a.noise, a.S))
    t1 = time.time()

    isummary = []
    if not a.skip_intervention:
        itasks = [(a.seed, i_n, nz, i_s, S, i_c, sc, rep, a.reduction, ln)
                  for (i_n, nz), (i_s, S), (i_c, sc), rep in product(enumerate(a.noise), enumerate(a.S), enumerate(SCENARIOS), range(a.reps))]
        itasks.sort(key=lambda t: -t[4])
        with ProcessPoolExecutor(a.workers) as ex:
            irows = [r for rs in ex.map(run_intervention_task, itasks, chunksize=1) for r in rs]
        irows.sort(key=lambda r: (r["estimator"], r["noise"], r["S"], SCENARIOS.index(r["scenario"]), r["rep"]))
        imetrics = ("rot_drop_sig", "resc_drop_sig", "p3_signature", "naive_signature", "both_drop_equal",
                    "rel_drop_rot_hat", "rel_drop_resc_hat", "rel_drop_rot_true", "rel_drop_resc_true")
        isummary = summarise(irows, ("estimator", "noise", "S", "scenario"), imetrics)
        write_csv(NAME, irows, seed=a.seed, stem="intervention_reps")
        write_csv(NAME, isummary, seed=a.seed, stem="intervention")
        save_figure(NAME, plot_intervention(isummary, a.noise, a.S), stem="intervention")

    rel = {nz: (np.mean([r["single_subject_r"] for r in summary if r["noise"] == nz]),
                np.mean([r["nili_lower"] for r in summary if r["noise"] == nz])) for nz in a.noise}
    write_json(NAME, dict(
        config=dict(reps=a.reps, noise=a.noise, S=a.S, layer_noise=ln, reduction=a.reduction,
                    sim_defaults=SimConfig().__dict__, indscal=INDSCAL_KW, quick=a.quick),
        reliability_by_noise={str(k): dict(single_subject_r=v[0], nili_lower=v[1]) for k, v in rel.items()},
        runtime_s=dict(identifiability=t1 - t0, intervention=time.time() - t1),
    ), seed=a.seed)

    lines = [f"# Auto-generated tables (seed {a.seed}, reps {a.reps})", "",
             "## Measurement reliability by noise level", "",
             md_table([dict(noise=k, single_subject_r=v[0], nili_lower=v[1]) for k, v in rel.items()],
                      ["noise", "single_subject_r", "nili_lower"])]
    for est in ESTIMATORS:
        lines += ["", f"## Recovery and classification — {est}", "",
                  md_table([r for r in summary if r["estimator"] == est],
                           ["noise", "S", "generator", "r_rot", "r_resc", "leak_rot_from_resc", "leak_resc_from_rot",
                            "detect_rot", "detect_resc", "classified_correctly"])]
    for est in ESTIMATORS:
        if isummary:
            lines += ["", f"## Intervention (reduction {a.reduction}) — {est}", "",
                      md_table([r for r in isummary if r["estimator"] == est],
                               ["noise", "S", "scenario", "rot_drop_sig", "resc_drop_sig", "p3_signature",
                                "naive_signature", "both_drop_equal", "rel_drop_rot_hat", "rel_drop_resc_hat"])]
    (Path(__file__).resolve().parents[1] / "results" / NAME / "tables.md").write_text("\n".join(lines) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
