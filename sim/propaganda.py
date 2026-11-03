"""Prediction 6: propaganda as a rigid local deformation with tension at the edge.

World: whitened shared space X (3 layers x 60 concepts, k = 5; sim/generator.py).
A "rewritten cluster" C = the n_c nearest neighbours (in X) of a seed concept drawn from
the value-laden layer. Observed group dissimilarities ``D' = d(Y) + noise`` where Y is the
deformed configuration and noise s.d. = ``noise * mean_dist * layer factor`` (0.5/1/1.5) —
``noise`` here is the noise of the *group-mean* matrix (≈ per-subject noise / sqrt(S)).

Conditions (all with the same total squared displacement ``n_c * (mag * mean_dist)^2``):
  * ``rigid``: C rotated about its centroid (random plane, angle θ ≤ π/2) and translated;
    half of the squared displacement from rotation, half from translation (translation
    takes the remainder when a small cluster cannot reach it with θ ≤ π/2);
  * ``nonrigid_subset``: every member of C displaced by an independent random vector;
  * ``global_spread``: every concept displaced by an independent random vector.

Global fit: the model is the reference space X (the analyst cannot re-fit the deformed
world freely — any configuration, rigidly rewritten or not, is exactly Euclidean and would
fit with zero stress). Distances: ratio-level Kruskal stress of D' against d(X), decomposed
per pair (structure_of_truth.stress). Configurations: classical MDS of D' aligned to X by
one global similarity Procrustes; per-concept residuals.

Interface statistics per replicate (pairs: CC within cluster, CR cross, RR rest):
  auc_cr_cc   AUC of pair stress, CR > CC   (rigid -> high: rigid moves keep CC intact)
  auc_cr_rr   AUC of pair stress, CR > RR
  conc_cr     (share of total stress on CR pairs) / (share of pairs that are CR)
  knn_conc_cr same as conc_cr on the 10-nearest-neighbour graph of X only (local edge)
  rigidity    mean CC stress / mean CR stress
  proc_auc    AUC of per-concept Procrustes residual, members of C > others
The "interface signature" for a rigid rewrite = auc_cr_cc AND auc_cr_rr both exceed the
95th percentile of the same statistic under the controls (non-rigid subset for auc_cr_cc,
global spread for auc_cr_rr) at the same n_c, noise, magnitude. Its power is the fraction
of rigid replicates where this holds. In ``global_spread`` the "cluster" is just the label
set (same seed rule), so its statistics show what an arbitrary concept set looks like.

Interface rewriting (ii): the rigid transform is additionally applied to a shell of the
``round(rho * n_c)`` nearest non-cluster concepts with linearly tapering weight
(w = 1 - rank/(shell+1)), i.e. ``y = x + w (T(x) - x)``; rho ∈ {0, 0.5, 1, 2}. Tension =
global stress-1, local (10-NN graph) stress-1 and peak per-concept stress, all against X.

Internal consistency (M2, alternative reading of (ii)): judgments mix two geometries —
within-cluster pairs from the rewritten world, rest-rest pairs unchanged, and each
cross (interface) pair rewritten with probability q, else still judged from X. A *free*
metric SMACOF fit in k dims (no reference) is then strained only by this inconsistency.
q = 0 and q = 1 are both consistent (old / new world), so tension can only fall with q
from its maximum, not from q = 0. Detectability: stress at q exceeds the 95th percentile
of stress at q = 1 for the same condition, n_c, noise, magnitude.

Usage: python sim/propaganda.py --seed 20261105 [--quick] [--workers N]
Outputs: results/propaganda/{propaganda_reps,propaganda,signature_power,taper_reps,taper}.csv,
consistency_reps.csv, consistency.csv, propaganda.json, propaganda.png, taper.png, consistency.png
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import expm  # noqa: E402

from generator import latent_space  # noqa: E402
from structure_of_truth.decompose import classical_mds, similarity_align  # noqa: E402
from structure_of_truth.io import save_figure, write_csv, write_json  # noqa: E402
from structure_of_truth.seeds import make_rng  # noqa: E402
from structure_of_truth.stress import stress  # noqa: E402

NAME = "propaganda"
N_PER_LAYER, K, KNN = 60, 5, 10
LAYER_NOISE = np.array([0.5, 1.0, 1.5])
CONDITIONS = ("rigid", "nonrigid_subset", "global_spread")


def cell_seed(master, *key):
    return int(np.random.SeedSequence(master, spawn_key=tuple(key)).generate_state(1)[0])


def world(rng):
    N = 3 * N_PER_LAYER
    X = latent_space(rng, N, K)
    layer = np.repeat(np.arange(3), N_PER_LAYER)
    D0 = np.linalg.norm(X[:, None] - X[None], axis=-1)
    mean_dist = float(D0[np.triu_indices(N, 1)].mean())
    return X, layer, D0, mean_dist


def pick_cluster(rng, D0, layer, n_c):
    seed = rng.choice(np.flatnonzero(layer == 2))
    return np.argsort(D0[seed])[:n_c]  # includes the seed itself


def plane_generator(rng, k):
    """Skew generator of a rotation in a random 2-plane: expm(θ G) rotates by angle θ."""
    Q, _ = np.linalg.qr(rng.normal(size=(k, 2)))
    u, v = Q[:, 0], Q[:, 1]
    return np.outer(u, v) - np.outer(v, u)


def rigid_transform(rng, Xc, target_rms):
    """Rotation about the centroid + translation with total RMS displacement target_rms.

    Half the squared displacement goes to rotation (angle found by bisection, capped at
    π/2), the rest to a translation in a random direction.
    """
    c = Xc.mean(0)
    A = Xc - c
    G = plane_generator(rng, Xc.shape[1])

    def rms_rot(th):
        return float(np.sqrt(((A @ expm(th * G) - A) ** 2).sum(1).mean()))

    rot_target = target_rms / np.sqrt(2)
    if rms_rot(np.pi / 2) <= rot_target:
        th = np.pi / 2
    else:
        lo, hi = 0.0, np.pi / 2
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if rms_rot(mid) < rot_target else (lo, mid)
        th = (lo + hi) / 2
    R = expm(th * G)
    t_len = np.sqrt(max(target_rms**2 - rms_rot(th) ** 2, 0.0))
    t = rng.normal(size=Xc.shape[1])
    t *= t_len / np.linalg.norm(t)

    def T(P):
        return (P - c) @ R + c + t

    return T, float(th), float(t_len)


def random_displacement(rng, n, k, rms):
    E = rng.normal(size=(n, k))
    return E * (rms / np.sqrt((E**2).sum(1).mean()))


def observe(rng, Y, layer, noise, mean_dist):
    N = len(Y)
    D = np.linalg.norm(Y[:, None] - Y[None], axis=-1)
    f = LAYER_NOISE[layer]
    sd = noise * mean_dist * (f[:, None] + f[None, :]) / 2
    E = np.triu(rng.normal(size=(N, N)), 1)
    D = np.abs(D + sd * (E + E.T))
    np.fill_diagonal(D, 0.0)
    return D


def auc(pos, neg):
    from scipy.stats import mannwhitneyu

    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def knn_mask(D0, k=KNN):
    N = len(D0)
    idx = np.argsort(D0, axis=1)[:, 1 : k + 1]
    M = np.zeros((N, N), bool)
    M[np.repeat(np.arange(N), k), idx.ravel()] = True
    return M | M.T


def interface_stats(Dobs, X, D0, C, knn):
    N = len(X)
    st = stress(Dobs, D0, level="ratio")
    inC = np.zeros(N, bool)
    inC[C] = True
    iu = np.triu_indices(N, 1)
    ps = st.pair[iu]
    a, b = inC[iu[0]], inC[iu[1]]
    cc, cr, rr = a & b, a ^ b, ~a & ~b
    kn = knn[iu]
    share = lambda mask: ps[mask].sum() / ps.sum() / mask.mean()
    out = dict(
        stress1=st.stress1,
        auc_cr_cc=auc(ps[cr], ps[cc]), auc_cr_rr=auc(ps[cr], ps[rr]),
        conc_cr=float(share(cr)),
        knn_conc_cr=float((ps[kn & cr].mean()) / ps[kn & ~cr].mean()) if (kn & cr).any() else float("nan"),
        rigidity=float(ps[cc].mean() / ps[cr].mean()) if cc.any() else float("nan"),
    )
    Y = classical_mds(Dobs, K)
    Ya = similarity_align(Y, X, np.arange(N))
    res = np.linalg.norm(Ya - X, axis=1)
    out["proc_auc"] = auc(res[inC], res[~inC])
    return out


def run_main(args):
    master, idx, cond, n_c, noise, mag, rep = args
    seed = cell_seed(master, 1, *idx, rep)
    rng = make_rng(seed)
    X, layer, D0, md = world(rng)
    C = pick_cluster(rng, D0, layer, n_c)
    target = mag * md
    Y = X.copy()
    if cond == "rigid":
        T, th, tl = rigid_transform(rng, X[C], target)
        Y[C] = T(X[C])
    elif cond == "nonrigid_subset":
        Y[C] += random_displacement(rng, n_c, K, target)
    else:  # equal total squared displacement spread over all concepts
        Y += random_displacement(rng, len(X), K, target * np.sqrt(n_c / len(X)))
    Dobs = observe(rng, Y, layer, noise, md)
    row = dict(condition=cond, n_c=n_c, noise=noise, mag=mag, rep=rep, task_seed=seed)
    row.update(interface_stats(Dobs, X, D0, C, knn_mask(D0)))
    return row


def run_taper(args):
    master, idx, n_c, noise, mag, rho, rep = args
    seed = cell_seed(master, 2, *idx, rep)
    rng = make_rng(seed)
    X, layer, D0, md = world(rng)
    C = pick_cluster(rng, D0, layer, n_c)
    T, _, _ = rigid_transform(rng, X[C], mag * md)
    rest = np.setdiff1d(np.arange(len(X)), C)
    shell_n = min(int(round(rho * n_c)), len(np.setdiff1d(np.arange(len(X)), C)))  # cannot exceed the rest
    w = np.zeros(len(X))
    w[C] = 1.0
    if shell_n:
        dist_to_C = D0[np.ix_(rest, C)].min(1)
        shell = rest[np.argsort(dist_to_C)[:shell_n]]
        w[shell] = 1.0 - (np.arange(1, shell_n + 1) / (shell_n + 1))
    Y = X + w[:, None] * (T(X) - X)
    Dobs = observe(rng, Y, layer, noise, md)
    st = stress(Dobs, D0, level="ratio")
    knn = knn_mask(D0)
    iu = np.triu_indices(len(X), 1)
    kn = knn[iu]
    dh, d = st.disparities[iu], D0[iu]
    knn_stress = float(np.sqrt(((dh - d)[kn] ** 2).sum() / (d[kn] ** 2).sum()))
    return dict(n_c=n_c, noise=noise, mag=mag, rho=rho, rep=rep, task_seed=seed, n_moved=int((w > 0).sum()),
                stress1=st.stress1, knn_stress1=knn_stress, peak_concept=float(st.concept.max()),
                peak_concept_stress1=float(np.nanmax(st.concept_stress1)))


def run_consistency(args):
    """M2: internally inconsistent judgments, free (SMACOF) fit, tension vs interface update q."""
    from sklearn.manifold import smacof

    master, idx, cond, n_c, noise, mag, q, rep = args
    seed = cell_seed(master, 3, *idx, rep)
    rng = make_rng(seed)
    X, layer, D0, md = world(rng)
    C = pick_cluster(rng, D0, layer, n_c)
    Y = X.copy()
    if cond == "rigid":
        T, _, _ = rigid_transform(rng, X[C], mag * md)
        Y[C] = T(X[C])
    else:
        Y[C] += random_displacement(rng, n_c, K, mag * md)
    D1 = np.linalg.norm(Y[:, None] - Y[None], axis=-1)
    N = len(X)
    inC = np.zeros(N, bool)
    inC[C] = True
    cross = inC[:, None] ^ inC[None, :]
    upd = np.triu(rng.random((N, N)) < q, 1)
    upd = upd | upd.T
    J = np.where(cross & ~upd, D0, D1)  # within-cluster: new; rest: unchanged; cross: new w.p. q
    f = LAYER_NOISE[layer]
    E = np.triu(rng.normal(size=(N, N)), 1)
    J = np.abs(J + noise * md * (f[:, None] + f[None, :]) / 2 * (E + E.T))
    np.fill_diagonal(J, 0.0)
    Yfit, _ = smacof(J, n_components=K, init=classical_mds(J, K), n_init=1, max_iter=300, metric=True,
                     normalized_stress=False)
    F = np.linalg.norm(Yfit[:, None] - Yfit[None], axis=-1)
    st = stress(J, F, level="ratio")
    iu = np.triu_indices(N, 1)
    ps = st.pair[iu]
    a, b = inC[iu[0]], inC[iu[1]]
    cc, cr, rr = a & b, a ^ b, ~a & ~b
    return dict(condition=cond, n_c=n_c, noise=noise, mag=mag, q=q, rep=rep, task_seed=seed,
                stress1=st.stress1, conc_cr=float(ps[cr].sum() / ps.sum() / cr.mean()),
                auc_cr_cc=auc(ps[cr], ps[cc]), auc_cr_rr=auc(ps[cr], ps[rr]))


def summarise(rows, keys, metrics):
    g = {}
    for r in rows:
        g.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, rs in g.items():
        row = dict(zip(keys, key), n_reps=len(rs))
        for m in metrics:
            v = np.array([r[m] for r in rs], float)
            row[m] = float(np.nanmean(v))
            row[m + "_sd"] = float(np.nanstd(v, ddof=1))
        out.append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--n_c", type=int, nargs="+", default=[3, 5, 10, 20, 40, 80])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.4])
    ap.add_argument("--mag", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--rho", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--q", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    if a.quick:
        a.n_c, a.noise, a.mag, a.reps = [5, 20], [0.0, 0.2], [0.2], 6
    t0 = time.time()
    tasks = [(a.seed, (i_c, i_n, i_s, i_m), cond, n_c, nz, mg, rep)
             for (i_c, cond), (i_n, n_c), (i_s, nz), (i_m, mg), rep in
             product(enumerate(CONDITIONS), enumerate(a.n_c), enumerate(a.noise), enumerate(a.mag), range(a.reps))]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run_main, tasks, chunksize=8))
    mets = ("stress1", "auc_cr_cc", "auc_cr_rr", "conc_cr", "knn_conc_cr", "rigidity", "proc_auc")
    summ = summarise(rows, ("condition", "mag", "noise", "n_c"), mets)

    power = []
    for n_c, nz, mg in product(a.n_c, a.noise, a.mag):
        get = lambda c, m: np.array([r[m] for r in rows if r["condition"] == c and r["n_c"] == n_c and r["noise"] == nz and r["mag"] == mg], float)
        rig_cc, rig_rr = get("rigid", "auc_cr_cc"), get("rigid", "auc_cr_rr")
        thr_cc = np.nanpercentile(get("nonrigid_subset", "auc_cr_cc"), 95)
        thr_rr = np.nanpercentile(get("global_spread", "auc_cr_rr"), 95)
        vs_nr = rig_cc > thr_cc
        vs_gl = rig_rr > thr_rr
        # false alarms: control replicates that would pass the same two thresholds
        nr_pass = (get("nonrigid_subset", "auc_cr_cc") > thr_cc) & (get("nonrigid_subset", "auc_cr_rr") > thr_rr)
        gl_pass = (get("global_spread", "auc_cr_cc") > thr_cc) & (get("global_spread", "auc_cr_rr") > thr_rr)
        power.append(dict(n_c=n_c, noise=nz, mag=mg, power_vs_nonrigid=float(np.mean(vs_nr)),
                          power_vs_global=float(np.mean(vs_gl)), power_signature=float(np.mean(vs_nr & vs_gl)),
                          false_alarm_nonrigid=float(np.mean(nr_pass)), false_alarm_global=float(np.mean(gl_pass)),
                          thr_auc_cr_cc=float(thr_cc), thr_auc_cr_rr=float(thr_rr)))
    write_csv(NAME, rows, seed=a.seed, stem="propaganda_reps")
    write_csv(NAME, summ, seed=a.seed)
    write_csv(NAME, power, seed=a.seed, stem="signature_power")

    t1 = time.time()
    ttasks = [(a.seed, (i_c, i_n, i_m, i_r), n_c, nz, mg, rho, rep)
              for (i_c, n_c), (i_n, nz), (i_m, mg), (i_r, rho), rep in
              product(enumerate(a.n_c), enumerate(a.noise), enumerate(a.mag), enumerate(a.rho), range(a.reps))]
    with ProcessPoolExecutor(a.workers) as ex:
        trows = list(ex.map(run_taper, ttasks, chunksize=8))
    tsumm = summarise(trows, ("mag", "noise", "n_c", "rho"), ("stress1", "knn_stress1", "peak_concept", "peak_concept_stress1", "n_moved"))
    write_csv(NAME, trows, seed=a.seed, stem="taper_reps")
    write_csv(NAME, tsumm, seed=a.seed, stem="taper")

    t2 = time.time()
    ctasks = [(a.seed, (i_c, i_n, i_s, i_m, i_q), cond, n_c, nz, mg, q, rep)
              for (i_c, cond), (i_n, n_c), (i_s, nz), (i_m, mg), (i_q, q), rep in
              product(enumerate(("rigid", "nonrigid_subset")), enumerate(a.n_c), enumerate(a.noise), enumerate(a.mag),
                      enumerate(a.q), range(a.reps))]
    with ProcessPoolExecutor(a.workers) as ex:
        crows = list(ex.map(run_consistency, ctasks, chunksize=8))
    csumm = summarise(crows, ("condition", "mag", "noise", "n_c", "q"), ("stress1", "conc_cr", "auc_cr_cc", "auc_cr_rr"))
    # detectability of tension: stress at q exceeds the 95th percentile of the fully consistent (q = 1) fit
    for r in csumm:
        ref = np.array([x["stress1"] for x in crows if x["condition"] == r["condition"] and x["n_c"] == r["n_c"]
                        and x["noise"] == r["noise"] and x["mag"] == r["mag"] and x["q"] == 1.0])
        cur = np.array([x["stress1"] for x in crows if x["condition"] == r["condition"] and x["n_c"] == r["n_c"]
                        and x["noise"] == r["noise"] and x["mag"] == r["mag"] and x["q"] == r["q"]])
        r["power_tension_vs_consistent"] = float(np.mean(cur > np.percentile(ref, 95))) if len(ref) else float("nan")
    write_csv(NAME, crows, seed=a.seed, stem="consistency_reps")
    write_csv(NAME, csumm, seed=a.seed, stem="consistency")

    # figures
    mg = a.mag[-1]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    cols = {"rigid": "tab:red", "nonrigid_subset": "tab:blue", "global_spread": "tab:grey"}
    nz0 = a.noise[min(1, len(a.noise) - 1)]
    for ax, met, title in zip(axes[:3], ("auc_cr_cc", "auc_cr_rr", "knn_conc_cr"),
                              ("AUC pair stress: cross > within-cluster", "AUC pair stress: cross > within-rest",
                               "local (10-NN) stress concentration on cross pairs")):
        for c in CONDITIONS:
            ys = [next(r[met] for r in summ if r["condition"] == c and r["n_c"] == n and r["noise"] == nz0 and r["mag"] == mg) for n in a.n_c]
            es = [next(r[met + "_sd"] for r in summ if r["condition"] == c and r["n_c"] == n and r["noise"] == nz0 and r["mag"] == mg) for n in a.n_c]
            ax.errorbar(a.n_c, ys, yerr=es, fmt="o-", color=cols[c], label=c, capsize=2)
        ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(a.n_c, [str(n) for n in a.n_c])
        ax.set_xlabel("cluster size n_c"); ax.set_title(f"{title}\n(noise {nz0}, mag {mg})", fontsize=8); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    ax = axes[3]
    M = np.array([[next(r["power_signature"] for r in power if r["n_c"] == n and r["noise"] == z and r["mag"] == mg) for n in a.n_c] for z in a.noise])
    im = ax.imshow(M, vmin=0, vmax=1, origin="lower", aspect="auto", cmap="viridis")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black" if M[i, j] > 0.6 else "white")
    ax.set_xticks(range(len(a.n_c)), [str(n) for n in a.n_c]); ax.set_yticks(range(len(a.noise)), [str(z) for z in a.noise])
    ax.set_xlabel("cluster size n_c"); ax.set_ylabel("group-matrix noise")
    ax.set_title(f"power of interface signature vs both controls (mag {mg})", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    save_figure(NAME, fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, met, title in zip(axes, ("stress1", "knn_stress1", "peak_concept_stress1"),
                              ("global stress-1", "local (10-NN) stress-1", "peak per-concept stress-1")):
        for n, ls in zip(a.n_c, ("-", "--", ":", "-.", "-", "--")):
            ys = [next(r[met] for r in tsumm if r["n_c"] == n and r["noise"] == nz0 and r["mag"] == mg and r["rho"] == rho) for rho in a.rho]
            ax.plot(a.rho, ys, "o", ls=ls, label=f"n_c={n}")
        ax.set_xlabel("rewritten interface shell (ρ × n_c concepts, tapered)"); ax.set_title(f"{title} (noise {nz0}, mag {mg})", fontsize=8); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save_figure(NAME, fig, stem="taper")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (met, title) in zip(axes, (("stress1", "free-fit stress-1 (tension)"), ("conc_cr", "stress concentration on cross pairs"),
                                       ("power_tension_vs_consistent", "P(tension > fully-consistent 95th pct)"))):
        for n, ls in zip(a.n_c, ("-", "--", ":", "-.", "-", "--")):
            for cond, col in (("rigid", "tab:red"), ("nonrigid_subset", "tab:blue")):
                ys = [next(r[met] for r in csumm if r["condition"] == cond and r["n_c"] == n and r["noise"] == nz0 and r["mag"] == mg and r["q"] == q) for q in a.q]
                ax.plot(a.q, ys, "o", ls=ls, color=col, alpha=0.8, label=f"{cond} n_c={n}" if met == "stress1" else None)
        ax.set_xlabel("fraction q of interface pairs rewritten"); ax.set_title(f"{title} (noise {nz0}, mag {mg})", fontsize=8); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    save_figure(NAME, fig, stem="consistency")

    write_json(NAME, dict(config=dict(n_c=a.n_c, noise=a.noise, mag=a.mag, rho=a.rho, q=a.q, reps=a.reps, knn=KNN,
                                      n_per_layer=N_PER_LAYER, k=K, layer_noise=LAYER_NOISE.tolist(), quick=a.quick),
                          runtime_s=dict(main=t1 - t0, taper=t2 - t1, consistency=time.time() - t2)), seed=a.seed)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
