"""Power simulations for prediction 1 (layer gradient) and prediction 2 (residual rank).

Shared machinery from sim/generator.py: whitened 3-layer latent space (3 x 60 concepts,
k = 5), layer-specific additive noise with s.d. ``noise * mean_dist * (f_i + f_j)/2``,
f = 0.5 / 1.0 / 1.5 (physical / intersubjective / value-laden).

Judgment design: every subject rates the same fixed set of pairs in which every concept
occurs exactly ``m`` times ("judgments per concept per subject"); ``m = N - 1`` is the full
matrix. The graph is a union of random circulants (exactly m-regular for even m).

P1 — cross-cultural convergence gradient
    ``n_cultures`` (3) cultures x ``S`` subjects each. Culture c's configuration is
    ``X + τ_layer(i) * E_c`` (concept-wise Gaussian displacement); subject s in c adds
    individual deviation ``ι * E_s`` (ι = 0.2) and measurement noise.
    Convergence of concept i between cultures a, b = Pearson r of the two culture-mean
    dissimilarity profiles of i (its m pairs); averaged over culture pairs.
    Noise-ceiling corrected: ``r_ab,i / sqrt(rel_a,i rel_b,i)`` (Spearman's
    disattenuation), rel = split-half Spearman–Brown reliability of the culture-mean profile
    within each culture (``n_splits`` = 50 seeded splits; reliabilities floored at 0.05).
    Scenarios: ``equal`` τ = (.3,.3,.3) — no true gradient, only reliability differs by
    layer; ``gradient`` τ = (.1,.3,.5); ``weak_gradient`` τ = (.2,.3,.4). True convergence is computed from noise-free culture
    configurations. Test: one-sided Mann–Whitney, physical > value-laden concepts.

P2 — rank of the individual-differences residual
    Subject configuration ``X + D_s``. Rank r ∈ {3, 5, 10}: ``D_s = Σ_j a_sj U_j`` with fixed
    unit-norm modes U_j and ``a_sj ~ N(0, η²/r)``; ``full``: i.i.d. entries (each subject an
    own direction). ``η² = 0.05 · N k`` for every r (same total individual variance).
    Residual R = subjects x pairs matrix minus its column means. Methods (rank cap 30):
      * ``var80``: smallest r explaining 80% of residual variance (SPEC default);
      * ``scree``: elbow of the log-eigenvalue curve (max distance to the chord);
      * ``parallel``: Horn — leading eigenvalues above the 95th percentile of 20 column-wise
        permutations;
      * ``cv``: bi-cross-validation — 5 folds over subjects; components from training
        subjects; test subjects' scores from a random half of the pairs, prediction error
        on the other half; argmin over r = 0..30;
      * ``retest_cv`` (needs a second session, i.e. twice the judgments): components from
        session 1 of all subjects (r = 0..S-1); each subject's session-2 scores from a
        random half of the pairs, prediction error on the other half. The only method here
        that can, in principle, see reliable idiosyncratic (full-rank) differences, because
        a subject's own session-1 deviation is in the component span.
      * ``retest_share`` (two sessions): reliable individual variance = sum_s <R1_s, R2_s>
        (noise cancels in expectation; one-sided t-test over subjects that it is > 0) and
        the share of it captured by the top-r̂_cv session-1 components. "low" if reliable
        variance is detected, 1 ≤ r̂_cv ≤ 6 and share ≥ 0.5; "high" if detected and
        (r̂_cv ≥ 8 or share < 0.5); otherwise inconclusive. This asks the question P2 needs
        ("is the reliable disagreement low-rank?") rather than "what is the rank?".
    Correct: |r̂-3| ≤ 1, |r̂-5| ≤ 1, |r̂-10| ≤ 2, full: r̂ ≥ min(20, 0.75 (S-1)).
    Low/high separation: "low" if 1 ≤ r̂ ≤ 6, "high" if r̂ ≥ 8, otherwise inconclusive.

Usage:  python sim/power.py --seed 20261104 [--quick] [--workers 4] [--only p1|p2]
Outputs: results/power/{p1_reps,p1,p2_reps,p2,power_table}.csv, power.json,
p1_gradient.png, p2_rank.png, power.png, tables.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations, product
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

from generator import latent_space  # noqa: E402
from structure_of_truth.io import save_figure, write_csv, write_json  # noqa: E402
from structure_of_truth.reliability import spearman_brown  # noqa: E402
from structure_of_truth.seeds import make_rng  # noqa: E402

NAME = "power"
N_PER_LAYER, K = 60, 5
LAYER_NOISE = np.array([0.5, 1.0, 1.5])
P1_TAU = {"equal": (0.3, 0.3, 0.3), "gradient": (0.1, 0.3, 0.5), "weak_gradient": (0.2, 0.3, 0.4)}
IOTA = 0.2
N_CULTURES = 3
P1_SPLITS = 50
ETA_FRAC = 0.05
RANKS = ("3", "5", "10", "full")
METHODS = ("var80", "scree", "parallel", "cv", "retest_cv", "retest_share")
SHARE_LOW = 0.5
R_MAX = 30
N_PERM = 20


def cell_seed(master: int, *key: int) -> int:
    return int(np.random.SeedSequence(master, spawn_key=tuple(key)).generate_state(1)[0])


# ---------------------------------------------------------------- design + measurement
def pair_design(N: int, m: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Pairs (P, 2) with every concept in exactly m pairs, and nbr (N, m) pair indices per concept."""
    if m >= N - 1:
        pairs = np.array(list(combinations(range(N), 2)))
    else:
        if m % 2:
            raise ValueError("m must be even (or N-1)")
        perm = rng.permutation(N)
        offsets = rng.choice(np.arange(1, (N - 1) // 2 + 1), size=m // 2, replace=False)
        pairs = np.array([sorted((perm[i], perm[(i + o) % N])) for o in offsets for i in range(N)])
    nbr = [[] for _ in range(N)]
    for p, (i, j) in enumerate(pairs):
        nbr[i].append(p)
        nbr[j].append(p)
    nbr = np.array(nbr)  # same design for every subject/culture, so profiles align
    return pairs, nbr


def base_world(rng: np.random.Generator):
    N = 3 * N_PER_LAYER
    X = latent_space(rng, N, K)
    layer = np.repeat(np.arange(3), N_PER_LAYER)
    iu = np.triu_indices(N, 1)
    mean_dist = float(np.linalg.norm(X[iu[0]] - X[iu[1]], axis=1).mean())
    return X, layer, mean_dist


def pair_sd(layer, pairs, noise, mean_dist):
    f = LAYER_NOISE[layer]
    return noise * mean_dist * (f[pairs[:, 0]] + f[pairs[:, 1]]) / 2.0


def measure_pairs(Y: np.ndarray, pairs: np.ndarray, sd: np.ndarray, rng) -> np.ndarray:
    """Y (S, N, k) configurations -> (S, P) noisy dissimilarities on the design pairs."""
    d = np.linalg.norm(Y[:, pairs[:, 0]] - Y[:, pairs[:, 1]], axis=-1)
    return np.abs(d + sd * rng.normal(size=d.shape))


# ---------------------------------------------------------------- P1
def rowcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    den = np.sqrt((a**2).sum(-1) * (b**2).sum(-1))
    return np.where(den > 0, (a * b).sum(-1) / np.where(den > 0, den, 1), np.nan)


def profile_reliability(V: np.ndarray, nbr: np.ndarray, rng, n_splits: int) -> np.ndarray:
    """Split-half SB reliability of each concept's culture-mean profile. V: (S, P)."""
    S = V.shape[0]
    rs = np.empty((n_splits, nbr.shape[0]))
    for t in range(n_splits):
        perm = rng.permutation(S)
        a, b = V[perm[: S // 2]].mean(0), V[perm[S // 2:]].mean(0)
        rs[t] = rowcorr(a[nbr], b[nbr])
    return np.asarray(spearman_brown(np.nanmean(rs, axis=0)))


def run_p1(args):
    master, idx, scen, S, m, noise, rep = args
    seed = cell_seed(master, 1, *idx, rep)
    rng = make_rng(seed)
    X, layer, mean_dist = base_world(rng)
    pairs, nbr = pair_design(len(layer), m, rng)
    sd = pair_sd(layer, pairs, noise, mean_dist)
    tau = np.asarray(P1_TAU[scen])[layer][:, None]
    cultures, means, rels, true_prof = [], [], [], []
    for c in range(N_CULTURES):
        Xc = X + tau * rng.normal(size=X.shape)
        Y = Xc[None] + IOTA * rng.normal(size=(S, *X.shape))
        V = measure_pairs(Y, pairs, sd, rng)
        means.append(V.mean(0)[nbr])
        rels.append(profile_reliability(V, nbr, rng, P1_SPLITS))
        true_prof.append(np.linalg.norm(Xc[pairs[:, 0]] - Xc[pairs[:, 1]], axis=1)[nbr])
    raw, corr, true = [], [], []
    for a, b in combinations(range(N_CULTURES), 2):
        r = rowcorr(means[a], means[b])
        raw.append(r)
        corr.append(r / np.sqrt(np.maximum(rels[a], 0.05) * np.maximum(rels[b], 0.05)))
        true.append(rowcorr(true_prof[a], true_prof[b]))
    raw, corr, true = (np.nanmean(x, axis=0) for x in (raw, corr, true))
    rel = np.mean(rels, axis=0)
    row = dict(scenario=scen, S=S, m=m, noise=noise, rep=rep, task_seed=seed)
    for name, v in (("raw", raw), ("corr", corr), ("true", true), ("rel", rel)):
        for L, lname in enumerate(("phys", "inter", "value")):
            row[f"{name}_{lname}"] = float(np.nanmean(v[layer == L]))
        row[f"{name}_gradient"] = row[f"{name}_phys"] - row[f"{name}_value"]
    for name, v in (("raw", raw), ("corr", corr)):
        p = mannwhitneyu(v[layer == 0], v[layer == 2], alternative="greater", nan_policy="omit").pvalue
        row[f"p_{name}"] = float(p)
        row[f"sig_{name}"] = bool(p < 0.05)
    return row


# ---------------------------------------------------------------- P2
def gram_eig(R: np.ndarray):
    """Eigen-decomposition of R R^T (descending): returns eigenvalues, left vectors U."""
    w, U = np.linalg.eigh(R @ R.T)
    return w[::-1].clip(0), U[:, ::-1]


def rank_var80(lam):
    c = np.cumsum(lam) / lam.sum()
    return int(np.searchsorted(c, 0.8) + 1)


def rank_scree(lam):
    lam = lam[: min(R_MAX + 1, len(lam) - 1)]  # last eigenvalue is ~0 by column centring
    y = np.log(lam + 1e-12 * lam[0])
    x = np.arange(len(y))
    p0, p1 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    q = np.column_stack([x, y]) - p0
    u = p1 - p0
    d = np.abs(u[0] * q[:, 1] - u[1] * q[:, 0]) / np.linalg.norm(u)
    return int(np.argmax(d))  # the corner is the first "noise" eigenvalue -> r = its index


def rank_parallel(R, lam, rng):
    perm_l = np.empty((N_PERM, min(len(lam), R_MAX + 1)))
    R32 = R.astype(np.float32)  # thresholds only; float32 halves the cost of the Gram products
    for t in range(N_PERM):
        Rp = rng.permuted(R32, axis=0)
        Rp = Rp - Rp.mean(0)
        perm_l[t] = gram_eig(Rp.astype(np.float64) if R.shape[1] < 2000 else Rp)[0][: perm_l.shape[1]]
    thr = np.percentile(perm_l, 95, axis=0)
    above = lam[: len(thr)] > thr
    return int(np.argmin(above)) if not above.all() else len(thr)


def rank_cv(R, rng, folds=5):
    S, P = R.shape
    rmax = min(R_MAX, S - S // folds - 2)
    press = np.zeros(rmax + 1)
    fold_of = rng.permutation(S) % folds
    for f in range(folds):
        tr, te = R[fold_of != f], R[fold_of == f]
        mu = tr.mean(0)
        tr, te = tr - mu, te - mu
        w, U = gram_eig(tr)
        keep = w > w[0] * 1e-12
        V = (tr.T @ U[:, keep]) / np.sqrt(w[keep])  # (P, r) right singular vectors
        cols = rng.permutation(P)
        A, B = cols[: P // 2], cols[P // 2:]
        press[0] += (te[:, B] ** 2).sum()
        for r in range(1, min(rmax, V.shape[1]) + 1):
            coef, *_ = np.linalg.lstsq(V[A, :r], te[:, A].T, rcond=None)
            press[r] += ((te[:, B].T - V[B, :r] @ coef) ** 2).sum()
    return int(np.argmin(press))


def rank_retest(R1, R2, rng):
    """Components from session 1 (all subjects, up to S-1); each subject's session-2 scores
    from a random half A of the pairs, prediction error on the other half B. Nested
    least squares via one QR of V[A]: with V[A] = QR, the r-component prediction is
    (V[B] R^-1)[:, :r] (Q^T y_A)[:r]."""
    S, P = R1.shape
    w, U = gram_eig(R1)
    keep = w > w[0] * 1e-10
    V = (R1.T @ U[:, keep]) / np.sqrt(w[keep])
    cols = rng.permutation(P)
    A, B = cols[: P // 2], cols[P // 2:]
    rmax = min(V.shape[1], len(A) - 1)
    Q, Rq = np.linalg.qr(V[A, :rmax])
    T = np.linalg.solve(Rq.T, V[B, :rmax].T).T  # V[B] R^-1
    C = Q.T @ R2[:, A].T  # (rmax, S)
    E = R2[:, B].T.copy()
    press = np.empty(rmax + 1)
    press[0] = (E**2).sum()
    for r in range(rmax):
        E -= np.outer(T[:, r], C[r])
        press[r + 1] = (E**2).sum()
    return int(np.argmin(press))


def reliable_share(R1, R2, r):
    """Reliable (cross-session) individual variance, and the share of it inside the top-r
    session-1 components. Returns (one-sided p that reliable variance > 0, share)."""
    from scipy.stats import ttest_1samp

    c = (R1 * R2).sum(axis=1)  # per-subject cross-session product: signal energy + zero-mean noise
    p = float(ttest_1samp(c, 0.0, alternative="greater").pvalue)
    if r == 0:
        return p, 0.0
    w, U = gram_eig(R1)
    V = (R1.T @ U[:, :r]) / np.sqrt(w[:r])
    captured = float(((R1 @ V) * (R2 @ V)).sum())
    return p, captured / max(c.sum(), np.finfo(float).tiny)


def p2_correct(rank: str, rhat: int, S: int) -> bool:
    if rank == "full":
        return rhat >= min(20, 0.75 * (S - 1))
    r = int(rank)
    return abs(rhat - r) <= (2 if r >= 10 else 1)


def p2_class(rhat: int) -> str:
    return "low" if 1 <= rhat <= 6 else ("high" if rhat >= 8 else "inconclusive")


def run_p2(args):
    master, idx, rank, S, m, noise, rep = args
    seed = cell_seed(master, 2, *idx, rep)
    rng = make_rng(seed)
    X, layer, mean_dist = base_world(rng)
    N = len(layer)
    pairs, _ = pair_design(N, m, rng)
    sd = pair_sd(layer, pairs, noise, mean_dist)
    eta2 = ETA_FRAC * N * K
    if rank == "full":
        Dv = np.sqrt(eta2 / (N * K)) * rng.normal(size=(S, N, K))
    else:
        r = int(rank)
        U = rng.normal(size=(r, N, K))
        U /= np.linalg.norm(U, axis=(1, 2), keepdims=True)
        a = rng.normal(scale=np.sqrt(eta2 / r), size=(S, r))
        Dv = np.einsum("sr,rnk->snk", a, U)
    Y = X[None] + Dv
    V1, V2 = measure_pairs(Y, pairs, sd, rng), measure_pairs(Y, pairs, sd, rng)
    R1, R2 = V1 - V1.mean(0), V2 - V2.mean(0)
    lam = gram_eig(R1)[0]
    rhat = dict(
        var80=rank_var80(lam), scree=rank_scree(lam), parallel=rank_parallel(R1, lam, rng),
        cv=rank_cv(R1, rng), retest_cv=rank_retest(R1, R2, rng),
    )
    rows = []
    truth_class = "low" if rank in ("3", "5") else "high"
    for meth, rh in rhat.items():
        rows.append(dict(rank=rank, S=S, m=m, noise=noise, rep=rep, method=meth, task_seed=seed, rhat=rh,
                         correct=p2_correct(rank, rh, S), separated=(p2_class(rh) == truth_class),
                         inconclusive=(p2_class(rh) == "inconclusive"), share=float("nan"), p_reliable=float("nan")))
    # retest_share: is there reliable individual variance, and does the low-rank CV model hold most of it?
    p_rel, share = reliable_share(R1, R2, rhat["cv"])
    rh = rhat["cv"]
    if p_rel >= 0.05:
        cls = "inconclusive"
    elif 1 <= rh <= 6 and share >= SHARE_LOW:
        cls = "low"
    elif rh >= 8 or share < SHARE_LOW:
        cls = "high"
    else:
        cls = "inconclusive"
    rows.append(dict(rank=rank, S=S, m=m, noise=noise, rep=rep, method="retest_share", task_seed=seed, rhat=rh,
                     correct=(cls == truth_class) and (rank == "full" or p2_correct(rank, rh, S)),
                     separated=(cls == truth_class), inconclusive=(cls == "inconclusive"),
                     share=share, p_reliable=p_rel))
    return rows


# ---------------------------------------------------------------- aggregation
def summarise(rows, keys, metrics):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for g, rs in groups.items():
        row = dict(zip(keys, g), n_reps=len(rs))
        for mtr in metrics:
            v = np.array([float(x[mtr]) for x in rs])
            row[mtr] = float(np.nanmean(v))
            if mtr == "rhat":
                row["rhat_median"] = float(np.median(v))
        out.append(row)
    return out


def min_design(cells: dict, Ss, ms, target=0.8):
    """For each m: smallest S with power >= target (power must stay >= target for all larger S)."""
    res = {}
    for m in ms:
        ok = [S for S in Ss if cells.get((S, m), 0) >= target]
        res[m] = next((S for S in Ss if all(cells.get((S2, m), 0) >= target for S2 in Ss if S2 >= S)), None) if ok else None
    return res


def heat(ax, cells, Ss, ms, title):
    M = np.array([[cells.get((S, m), np.nan) for m in ms] for S in Ss])
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", origin="lower", aspect="auto")
    for i in range(len(Ss)):
        for j in range(len(ms)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if M[i, j] > 0.6 else "white")
    ax.set_xticks(range(len(ms)), [str(m) for m in ms])
    ax.set_yticks(range(len(Ss)), [str(s) for s in Ss])
    ax.set_xlabel("judgments per concept per subject (m)")
    ax.set_title(title, fontsize=8)
    return im


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--m", type=int, nargs="+", default=[10, 20, 40, 90, 179])
    ap.add_argument("--S1", type=int, nargs="+", default=[5, 10, 20, 40, 80], help="P1 subjects per culture")
    ap.add_argument("--S2", type=int, nargs="+", default=[20, 50, 100, 200, 400], help="P2 subjects")
    ap.add_argument("--reps1", type=int, default=40)
    ap.add_argument("--reps2", type=int, default=20)
    ap.add_argument("--only", choices=["p1", "p2"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    if a.quick:
        a.noise, a.m, a.S1, a.S2, a.reps1, a.reps2 = [0.5], [20, 179], [10, 40], [20, 100], 6, 3
    out = Path(__file__).resolve().parents[1] / "results" / NAME
    t0 = time.time()
    meta = dict(config=dict(noise=a.noise, m=a.m, S1=a.S1, S2=a.S2, reps1=a.reps1, reps2=a.reps2,
                            n_per_layer=N_PER_LAYER, k=K, layer_noise=LAYER_NOISE.tolist(), p1_tau=P1_TAU,
                            iota=IOTA, n_cultures=N_CULTURES, p1_splits=P1_SPLITS, eta_frac=ETA_FRAC,
                            r_max=R_MAX, n_perm=N_PERM, quick=a.quick))
    power_rows = []

    if a.only in (None, "p1"):
        tasks = [(a.seed, (i_c, i_s, i_m, i_n), sc, S, m, nz, rep)
                 for (i_c, sc), (i_s, S), (i_m, m), (i_n, nz), rep in
                 product(enumerate(P1_TAU), enumerate(a.S1), enumerate(a.m), enumerate(a.noise), range(a.reps1))]
        tasks.sort(key=lambda t: -t[3] * t[4])
        with ProcessPoolExecutor(a.workers) as ex:
            p1 = list(ex.map(run_p1, tasks, chunksize=4))
        p1.sort(key=lambda r: (r["scenario"], r["noise"], r["S"], r["m"], r["rep"]))
        mets = [f"{n}_{l}" for n in ("raw", "corr", "true", "rel") for l in ("phys", "inter", "value", "gradient")] + ["sig_raw", "sig_corr"]
        p1s = summarise(p1, ("scenario", "noise", "S", "m"), mets)
        write_csv(NAME, p1, seed=a.seed, stem="p1_reps")
        write_csv(NAME, p1s, seed=a.seed, stem="p1")
        for nz, gs in product(a.noise, ("gradient", "weak_gradient")):
            pw = {(r["S"], r["m"]): r["sig_corr"] for r in p1s if r["scenario"] == gs and r["noise"] == nz}
            fp = {(r["S"], r["m"]): r["sig_corr"] for r in p1s if r["scenario"] == "equal" and r["noise"] == nz}
            for m, S in min_design(pw, a.S1, a.m).items():
                power_rows.append(dict(prediction="P1", method=f"corrected gradient test ({gs})", noise=nz, m=m,
                                       min_S=S if S is not None else f">{max(a.S1)}", S_unit="per culture",
                                       false_pos_at_min=fp.get((S, m), float("nan")) if S else float("nan")))
        # figure: raw vs corrected vs true layer means at the largest S, full m, each noise
        fig, axes = plt.subplots(1, len(P1_TAU) * len(a.noise), figsize=(3.6 * len(P1_TAU) * len(a.noise), 3.6), sharey=True)
        axes = np.atleast_1d(axes)
        for j, (sc, nz) in enumerate(product(P1_TAU, a.noise)):
            ax = axes[j]
            for S, ls in zip(a.S1, ("-", "--", ":", "-.", (0, (1, 1)))):
                r = next(x for x in p1s if x["scenario"] == sc and x["noise"] == nz and x["S"] == S and x["m"] == max(a.m))
                if S in (min(a.S1), max(a.S1)):
                    ax.plot([0, 1, 2], [r["raw_phys"], r["raw_inter"], r["raw_value"]], "o", ls=ls, color="tab:red", label=f"raw S={S}")
                    ax.plot([0, 1, 2], [r["corr_phys"], r["corr_inter"], r["corr_value"]], "s", ls=ls, color="tab:blue", label=f"corrected S={S}")
            r = next(x for x in p1s if x["scenario"] == sc and x["noise"] == nz and x["S"] == max(a.S1) and x["m"] == max(a.m))
            ax.plot([0, 1, 2], [r["true_phys"], r["true_inter"], r["true_value"]], "k^-", lw=2, label="true")
            ax.set_xticks([0, 1, 2], ["physical", "intersubj.", "value"])
            ax.set_title(f"{sc} τ, noise {nz}, m={max(a.m)}", fontsize=9)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("cross-cultural convergence (r)")
        axes[0].legend(fontsize=7)
        fig.suptitle("P1: apparent layer gradient, raw vs noise-ceiling corrected")
        fig.tight_layout()
        save_figure(NAME, fig, stem="p1_gradient")
    t1 = time.time()

    if a.only in (None, "p2"):
        tasks = [(a.seed, (i_r, i_s, i_m, i_n), rk, S, m, nz, rep)
                 for (i_r, rk), (i_s, S), (i_m, m), (i_n, nz), rep in
                 product(enumerate(RANKS), enumerate(a.S2), enumerate(a.m), enumerate(a.noise), range(a.reps2))]
        tasks.sort(key=lambda t: -t[3] * t[4])
        with ProcessPoolExecutor(a.workers) as ex:
            p2 = [r for rs in ex.map(run_p2, tasks, chunksize=1) for r in rs]
        p2.sort(key=lambda r: (r["method"], r["noise"], r["S"], r["m"], RANKS.index(r["rank"]), r["rep"]))
        p2s = summarise(p2, ("method", "noise", "S", "m", "rank"), ("rhat", "correct", "separated", "inconclusive", "share", "p_reliable"))
        write_csv(NAME, p2, seed=a.seed, stem="p2_reps")
        write_csv(NAME, p2s, seed=a.seed, stem="p2")
        sep = {}
        for meth, nz, S, m in product(METHODS, a.noise, a.S2, a.m):
            rs = {r["rank"]: r for r in p2s if r["method"] == meth and r["noise"] == nz and r["S"] == S and r["m"] == m}
            sep[(meth, nz, S, m)] = min(rs[k]["separated"] for k in RANKS)  # all four truths
            sep[(meth, nz, S, m, "lowonly")] = min(rs[k]["separated"] for k in ("3", "5", "10"))
        for meth, nz in product(METHODS, a.noise):
            for tag, label in (("", "3/5 vs 10/full"), ("lowonly", "3/5 vs 10 (full excluded)")):
                cells = {(S, m): sep[(meth, nz, S, m) + ((tag,) if tag else ())] for S in a.S2 for m in a.m}
                for m, S in min_design(cells, a.S2, a.m).items():
                    power_rows.append(dict(prediction="P2", method=f"{meth}: {label}", noise=nz, m=m,
                                           min_S=S if S is not None else f">{max(a.S2)}", S_unit="total",
                                           false_pos_at_min=float("nan")))
        fig, axes = plt.subplots(len(a.noise), len(METHODS), figsize=(3.2 * len(METHODS), 3.0 * len(a.noise)), squeeze=False)
        for i, nz in enumerate(a.noise):
            for j, meth in enumerate(METHODS):
                cells = {(S, m): sep[(meth, nz, S, m)] for S in a.S2 for m in a.m}
                im = heat(axes[i, j], cells, a.S2, a.m, f"{meth}, noise {nz}\nP(all 4 ranks separated)")
                if j == 0:
                    axes[i, j].set_ylabel("subjects S")
        fig.colorbar(im, ax=axes, shrink=0.6)
        fig.suptitle("P2: worst-case low/high-rank separation over true ranks {3, 5, 10, full}")
        save_figure(NAME, fig, stem="p2_rank")

    if power_rows:
        write_csv(NAME, power_rows, seed=a.seed, stem="power_table")
    # combined power figure: P1 corrected power and best P2 method
    if a.only is None:
        fig, axes = plt.subplots(len(a.noise), 3, figsize=(12, 3.2 * len(a.noise)), squeeze=False)
        for i, nz in enumerate(a.noise):
            heat(axes[i, 0], {(r["S"], r["m"]): r["sig_corr"] for r in p1s if r["scenario"] == "weak_gradient" and r["noise"] == nz},
                 a.S1, a.m, f"P1 power (weak true gradient), corrected, noise {nz}")
            heat(axes[i, 1], {(r["S"], r["m"]): r["sig_raw"] for r in p1s if r["scenario"] == "equal" and r["noise"] == nz},
                 a.S1, a.m, f"P1 false positives (no true gradient), RAW, noise {nz}")
            im = heat(axes[i, 2], {(S, m): sep[("retest_share", nz, S, m)] for S in a.S2 for m in a.m},
                      a.S2, a.m, f"P2 low/high separation, retest_share (2 sessions), noise {nz}")
            axes[i, 0].set_ylabel("subjects per culture")
            axes[i, 2].set_ylabel("subjects (total)")
        fig.colorbar(im, ax=axes, shrink=0.6)
        fig.suptitle("Power: 80% contour is the design target")
        save_figure(NAME, fig, stem="power")

    meta["runtime_s"] = dict(p1=t1 - t0, p2=time.time() - t1)
    write_json(NAME, meta, seed=a.seed)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
