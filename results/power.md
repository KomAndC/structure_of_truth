# Power for prediction 1 (layer gradient) and prediction 2 (residual rank)

Run: `python sim/power.py --seed 20261104 --workers 3` (50 min). Figures are rebuilt from
the CSVs by `python sim/power_figures.py`. P1 uses 40 replicates per cell; P2 uses 20
replicates per cell × 4 true ranks. Files are in `results/power/`:
- tables: `power_table.csv`, `p1.csv`, `p2.csv`, `*_reps.csv`
- figures: `power.png`, `p1_gradient.png`, `p2_rank.png`, `p2_false_low.png`

## Bottom line

**Minimum design for 80% power.** S = subjects; m = judgments per concept per subject.
Each subject rates N·m/2 = 90·m pairs.

| prediction | effect | noise (single-subject r) | m = 10 (900 pairs/subj) | m = 20 (1,800) | m = 40 (3,600) | m = 90 (8,100) | m = 179 (16,110, full) |
|---|---|---|---|---|---|---|---|
| P1 | strong gradient | 0.5 (r ≈ 0.53) | **40**/culture | **10** | **10** | 5 | 5 |
| P1 | weak gradient | 0.5 | >80 | **40** | **20** | 10 | 5 |
| P1 | strong gradient | 1.0 (r ≈ 0.27) | >80 | 80 | 40 | 20 | 20 |
| P1 | weak gradient | 1.0 | >80 | >80 | >80 | >80 | >80 |
| P2 | rank 3/5 vs 10/full, two sessions (`retest_share`) | 0.5 | 400* | >400 | 400* | **100** | **100** |
| P2 | same | 1.0 | >400 | >400 | >400 | 400 | 200 |
| P2 | rank 3/5 vs 10, one session (`cv`; full rank excluded) | 0.5 | >400 | >400 | 400 | 100 | 100 |
| P2 | same, one session (`scree`; all four ranks) | 0.5 | >400 | 400* | 100* | 200* | >400 |

P1 counts are subjects **per culture** (3 cultures). P2 counts are **total** subjects. "*"
marks cells that sit next to non-monotone neighbours; with 20 replicates each power estimate
carries about ±0.1 error, so these are fragile (see the P2 hazard below). Bold marks designs
that fit a plausible human budget of about 1,000–4,000 pair ratings per subject.

1. **P1 is cheap to test, but only after noise-ceiling correction. Raw convergence is
   worthless for P1.** Even when there is no true gradient, *raw* cross-cultural
   convergence shows a significant physical > value-laden gradient in 85–100% of
   replicates for almost every design. The only exception is S = 80/culture with m = 10,
   at 35%. This happens purely because value-laden judgments are less reliable. The
   artificial gradient is as large as or larger than the real effect: at noise 0.5 and
   5–20 subjects per culture it is 0.09–0.20 in r, against a true strong gradient of
   0.125. Within the raw gradient measured under a *true* strong gradient, the share that
   is artifact is:
   - noise 0.5: 77% at S = 5, 50% at S = 20, 21% at S = 80 (full matrix);
   - noise 1.0: 87%, 80% and 48%.

   Disattenuating by each culture's split-half reliability removes the artifact. False
   positives fall to 0–7.5%, and the corrected layer means track the true ones closely
   (e.g. strong gradient, S = 80: corrected 0.86/0.82/0.74 vs true 0.85/0.80/0.72 for
   physical / intersubjective / value).
2. **The correction fails when reliability is low, and then it fails in the conservative
   direction.** When the value-layer reliability of a culture-mean profile is below about
   0.5, dividing by it over-corrects: corrected r exceeds 1, and the corrected gradient is
   pushed *towards the value-laden layer*, which kills power. Below about 0.15
   reliability the correction also stops controlling false positives (0.23 at noise 1.0,
   S = 5, m = 10). **Rule: interpret corrected convergence only for layers whose mean
   split-half reliability is at least about 0.5. Below that, collect more data rather
   than correcting harder.**
3. **P2 needs a second session and far more data than P1.** It is out of reach at
   realistic per-subject budgets unless the concept set is reduced.
   - With one session, **no method can see an idiosyncratic full-rank residual**. CV
     always returns rank 0 for it, and parallel analysis does too apart from a few
     replicates at S = 400. Stable private disagreement is indistinguishable from
     measurement noise, so the refutation branch of P2 ("high-rank and idiosyncratic")
     cannot be established from one session.
   - With two sessions, the reliable individual variance (cross-session covariance) is
     measurable, and the question "does a low-rank model hold most of it?" becomes
     answerable. At noise 0.5 that requires about 100 subjects × 8,100 pair ratings each ×
     2 sessions. At noise 1.0 it requires 200–400 subjects with full matrices.
4. **A partially powered P2 study is dangerous: it confirms P2 falsely.** Some designs
   resolve only about half of 10 true components. In those designs a **true rank-10
   residual is called "low-rank (≤ 6)" most of the time** (`p2_false_low.png`), at
   noise 0.5:
   - cv: 90–95% at (S = 50, m = 90), (S = 100, m = 40) and (S = 200, m = 20);
   - retest_share: up to 80% at (S = 50, m = 90), and 80% at (S = 20, m = 179);
   - scree and parallel analysis: 60–100% at S = 20 with m ≥ 20–40. The band moves diagonally across S × m,
   so adding subjects *or* judgments can move a design into it as well as out of it.
   Because of this, the power criterion in the table requires all four true ranks to be
   classified correctly. A "3–5 axes" result from real data must be accompanied by a
   simulation at the observed noise level showing that a rank-10 truth would have been
   resolved.
5. **Method choice.**
   - `var80` (the SPEC default for residual rank) is useless: noise dominates the
     residual variance, so it always reports rank ≈ 0.75·S. **This default is retired;**
     see SPEC open decision 6.
   - Parallel analysis over-counts at large S. At S = 400 and m = 10 it reports about 27
     components for a true rank 10, because it counts small systematic nonlinear
     components (distance is a nonlinear function of the configuration) as significant.
   - Scree works erratically.
   - Bi-cross-validation (`cv`) is the most accurate single-session estimator of the
     *shared* rank.
   - The two-session reliable-variance share (`retest_share`) is the only method that
     tests P2 as stated.

## P1 detail (full matrix, m = 179; layer means physical / intersubjective / value-laden)

| scenario | noise | S/culture | raw | corrected | true | raw gradient significant | corrected significant |
|---|---|---|---|---|---|---|---|
| no true gradient (equal τ) | 0.5 | 5 | 0.60 / 0.50 / 0.41 | 0.84 / 0.85 / 0.86 | 0.83 / 0.83 / 0.83 | 1.00 | 0.00 |
| no true gradient | 0.5 | 80 | 0.81 / 0.80 / 0.78 | 0.83 / 0.83 / 0.83 | 0.83 / 0.83 / 0.83 | 0.97 | 0.03 |
| no true gradient | 1.0 | 20 | 0.62 / 0.52 / 0.43 | 0.85 / 0.87 / 0.89 | 0.83 / 0.83 / 0.83 | 1.00 | 0.00 |
| strong gradient | 0.5 | 5 | 0.62 / 0.50 / 0.38 | 0.87 / 0.83 / 0.76 | 0.85 / 0.81 / 0.73 | 1.00 | 1.00 |
| strong gradient | 0.5 | 80 | 0.84 / 0.78 / 0.70 | 0.86 / 0.82 / 0.74 | 0.85 / 0.80 / 0.72 | 1.00 | 1.00 |
| strong gradient | 1.0 | 5 | 0.36 / 0.25 / 0.17 | 0.93 / 0.99 / 1.02 | 0.85 / 0.80 / 0.72 | 1.00 | 0.07 |

The last row is the over-correction failure: the value-layer reliability there is 0.20.

## P2 detail: what each method returns (median r̂, noise 0.5, m = 179)

| S | true 3 | true 5 | true 10 | true full |
|---|---|---|---|---|
| cv, 20 | 3 | 4 | 4 | 0 |
| cv, 100 | 3 | 5 | 10 | 0 |
| parallel, 400 | 4 | 6 | 10 | 18 |
| scree, 100 | 4 | 5 | 10 | 10 |
| var80, any S | ≈ 0.75·S for every truth | | | |

retest_share at noise 0.5, m = 179: all four truths are classified correctly in ≥ 95% of
replicates from S = 100. In 100% of replicates the full-rank truth is called "high"
because reliable variance is detected but CV captures none of it. Note that for the
full-rank truth, and for rank 10 at underpowered designs, the "high" verdict means "the
reliable disagreement is not captured by a few strong axes". It does not mean the rank
was recovered.

## Assumptions (all explicit)

- **World:** 180 concepts (3 × 60), latent k = 5, whitened; layer noise factors
  0.5 / 1.0 / 1.5 (physical / intersubjective / value-laden) × `noise` × mean distance;
  Gaussian, i.i.d. per pair and per subject; no rating-scale effects, no missing data,
  no inattentive subjects.
- **Judgment design:** every subject rates the *same* fixed set of pairs in which each
  concept appears exactly m times (m-regular random graph). Designs where different
  subjects see different pairs are not simulated. They would spread coverage further but
  make the P2 rank analysis a missing-data problem.
- **Noise levels:** noise 0.5 corresponds to a single-subject reliability (full matrix)
  of about 0.53 and noise 1.0 to about 0.27 (from `results/identifiability.md`). Compare
  these against the Nili lower-bound noise ceiling of pilot data to choose the relevant
  row.
- **P1 effect:** culture-specific displacement of each concept with s.d. τ per layer:
  no gradient .3/.3/.3, weak .2/.3/.4, strong .1/.3/.5. The resulting true convergence
  gradient (physical − value) is 0 / ≈ 0.07 / ≈ 0.125 in r. Individual deviation within
  culture: s.d. 0.2. Three cultures of equal size.
- **P1 test:** one-sided Mann–Whitney over concepts (physical > value-laden), treating
  concepts as independent. Its calibration was checked empirically with the
  no-gradient scenario rather than assumed. Reliability: 50 split-halves (the real
  pipeline default is 200), floored at 0.05. Pearson r in the simulation (the pipeline
  default is Spearman).
- **P2 effect:** individual deviations hold 5% of the shared configuration's variance,
  split equally over 3, 5 or 10 shared modes, or i.i.d. per subject (full rank).
  Residuals are taken in distance space (alignment-free), so they are only approximately
  low-rank: distance is nonlinear in the configuration.
- **P2 classification:** "low" if r̂ ∈ [1, 6], "high" if r̂ ≥ 8, otherwise
  inconclusive. retest_share adds: reliable variance detected (one-sided t, p < .05) and
  share ≥ 0.5 → low. Power is the minimum over the four true ranks. The minimum S is the
  smallest S at which this and all larger S tested reach ≥ 0.8.
- **Precision:** 20–40 replicates per cell, so power estimates are ±0.07–0.1. Treat
  boundary cells as approximate.

## Practical implications for the study design

- P1 can be tested with about 1,800–3,600 pair ratings per subject (m = 20–40) and
  20–40 subjects per culture if single-subject reliability is about 0.5 or better.
  Always report the corrected statistic together with per-layer reliability.
- P2 as specified (180 concepts) needs two sessions and about 8,000 ratings per subject
  per session at 100+ subjects. That is not realistic for human raters. Options, none of
  them simulated yet:
  - a reduced concept set for the P2 sub-study;
  - spatial-arrangement or triplet tasks, which yield many implied distances per minute;
  - pooling repeated sessions.

  This is the most important design constraint found so far.
