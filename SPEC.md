# SPEC — Structure of Truth

## Experiment design (verbatim)

Minimal setup: ~150-200 concepts across three layers (physical: microorganism, gravity; intersubjective: money, nation, trust; value-laden: capitalism, justice). Similarity judgments from people across cultures, plus embeddings from several LLMs. Generalized Procrustes or INDSCAL, and a noise ceiling (split-half reliability) for each concept.
Predictions, from safest to riskiest:

1. Convergence gradient across layers. High agreement on the physical, medium on the intersubjective, low on the value-laden. Your idea’s specific prediction is different: cross-cultural convergence should be higher for the structure of trust and cooperation than for concrete institutions. If agreement on “money” is high only within a culture, the convergence comes from shared exposure, not from constraints.
2. The residual is low-rank. If the space is shared and only the weights differ, disagreement between people should be explained by 3-5 axes (individual-collective, present-future, authority-liberty). Individual weights should correlate with measures like Moral Foundations. If the residual is high-rank and idiosyncratic, disagreement is irreducible.
3. Rotation yields to clarification, weights don’t. The most testable prediction: give an explicit definition or a short clarifying discussion, then re-measure. The rotational component of the residual should drop, but not the rescaling component. If both drop equally, your definitional/value distinction isn’t real geometrically.
4. LLMs align better with the human centroid than with any individual. An averaging effect, already known. But their centroid will sit closer to educated, Western, English-speaking subjects. So your “core” is really the corpus’s core, and that is measurable: LLM-human distance as a function of subject culture. LLMs should converge with each other more than with humans.
5. Equivocation shows up as geometric drift. A term used in two premises occupies different positions in the contextual embedding. The shift between premises should predict equivocation fallacies. The effect is probably moderate, since many arguments fail for other reasons.
6. Propaganda is a local deformation with tension at the edge. A coherently rewritten cluster moves rigidly, and global fit error (stress) concentrates at the interface with unmodified concepts. Tension should fall as more interfaces are rewritten, which links directly to your experiment.

What would refute the idea: a high-rank residual; equal cross-cultural convergence across layers; the effect in (4) disappearing after controlling for word frequency.
The most important caution: without the noise ceiling, much of the “divergence” on abstract concepts will simply be unreliable measurement. And if the gradient in (1) comes out, it shows where constraints determine structure, not that the core is true.
One elegant test in the spirit of your toy world: train several models on worlds with different constraints and check whether the geometries converge only for the parts determined by the world’s structure, not for the arbitrary parts.

## Terminology

- **Shared space.** A single configuration of the N concepts in a D-dimensional space
  (the *consensus* / *group stimulus space*) from which every subject's (person's or
  LLM's) configuration is assumed to derive by a subject-specific transformation plus
  noise. In GPA it is the mean of the aligned configurations; in INDSCAL it is the group
  stimulus space `X`. The core hypothesis is that such a space exists and is largely
  constrained by the world, not by culture.
- **Per-subject rotation.** The orthogonal transformation `R_k` (rotation, possibly with
  reflection) that best maps subject k's configuration onto the shared space after
  centring and isotropic scaling. Rotation changes *which directions* the subject treats
  as the axes — interpreted here as differences in *definition/framing* of terms. It
  preserves all within-subject distances.
- **Per-axis weights.** Non-negative diagonal scalings `W_k = diag(w_k1..w_kD)` applied to
  the axes of the shared space (INDSCAL's subject weights). They stretch or shrink
  dimensions without rotating them, changing distances — interpreted as differences in
  *how much a subject cares about* an axis (value weighting). Subject k's model is
  `X_k ≈ X W_k R_k` (rotation + rescaling residual decomposition, cf. prediction 3).
- **Noise ceiling.** The best fit any model could achieve given measurement noise,
  estimated per concept by split-half reliability: split subjects randomly in two, average
  each half's dissimilarities for the concept's row, correlate the halves (Spearman),
  average over many seeded splits, and apply the Spearman–Brown correction to full sample
  size. Divergence is only interpretable relative to this ceiling.
- **Stress.** Global goodness-of-fit error between observed dissimilarities `δ_ij` and
  distances `d_ij` in a fitted configuration; we use Kruskal stress-1,
  `sqrt(Σ(δ_ij − d_ij)² / Σ d_ij²)`. *Local stress* of concept i is its share of the
  numerator; prediction 6 concerns where local stress concentrates.
- **Residual rank.** After removing the shared space and per-subject rotations, stack the
  subject residuals (subjects × concept-coordinates) and take its effective rank: the
  number of components needed to explain a fixed fraction (default 80%) of residual
  variance beyond what is expected from noise (parallel analysis against seeded
  permutation/noise-ceiling-matched null). Low (3–5) supports prediction 2; high and
  idiosyncratic refutes it.

## Open decisions

Choices made during setup that a future session may revisit. Change them here, not
silently in code.

1. **Package layout.** Importable code lives in `src/structure_of_truth/`; `sim/` holds
   runnable scripts (not a package); `tests/` uses pytest; results go to
   `results/<analysis_name>/<analysis_name>.{csv|json,png}`.
2. **Python version.** `requires-python = ">=3.11"` (3.11 is the target; newer versions
   are allowed so the code runs on the dev container's interpreter).
3. **Alignment method.** Both are implemented. *GPA* (`procrustes.py`): centring,
   orthogonal rotation (reflections allowed by default; `allow_reflection=False` forces
   det=+1), optional isotropic scaling with the Ten Berge constraint
   `sum_k s_k^2 ||X_k||^2 = K` on unit-norm inputs; deterministic first target = subject 0;
   stop when relative loss decrease < 1e-12. Configurations of different width are
   zero-padded. *INDSCAL* (`indscal.py`): classical double centring of squared
   dissimilarities, CANDECOMP-style ALS with two concept factors A, B (Carroll & Chang),
   NNLS for weights (w >= 0), symmetrised `X = (A+B)/2` at the end with weights refitted.
   Normalisation: columns of X centred with `diag(X^T X)/n = 1`; all scale in the weights;
   dimensions sorted by mean weight, sign fixed by largest-|entry| positive. Stop when
   relative strain decrease < 1e-10. Default init = classical MDS of mean scalar products;
   `init="random"` requires a seed.
4. **Noise ceiling definition.** Per concept: row-wise split-half reliability of the
   concept's dissimilarity profile (diagonal excluded), 200 seeded splits (halves
   floor(K/2)/ceil(K/2)), Spearman by default (Pearson optional), Spearman–Brown factor 2
   applied to the *mean* split-half r; negative values are not clipped. Dataset level:
   same split-half on the full upper triangle, plus Nili et al. (2014) lower
   (leave-one-out) and upper (incl. self) bounds.
5. **Stress variant.** Kruskal stress-1 with disparities at a chosen level: `ratio`
   (default), `raw`, `interval`, `ordinal` (isotonic). Per-pair and per-concept
   contributions are defined to sum to stress-1² (per-concept = half the row sum), plus a
   row-normalised per-concept stress-1.
6. **Residual-rank criterion.** ~~80% variance threshold~~ — **retired**: simulation
   (`results/power.md`) shows it tracks noise (r̂ ≈ 0.75·S for every true rank). Use
   bi-cross-validation (`cv`) for the shared rank and, for the P2 test itself, the
   two-session reliable-variance share (`retest_share`). A low-rank verdict must be
   accompanied by a simulation at the observed noise level showing that a rank-10 truth
   would have been resolved (partially powered designs falsely report low rank).
7. **Similarity format.** Assume full (or NaN-masked partial) N×N dissimilarity matrices
   per subject. The collection method (pairwise ratings vs. spatial arrangement vs. triplet
   odd-one-out) is not yet decided and determines how matrices are built.
8. **LLM representations.** Not yet decided: static word embeddings vs. contextual
   (averaged over template sentences) vs. elicited similarity judgments. Prediction 5
   requires contextual embeddings.
9. **Concept list.** The ~150–200 concepts and their layer assignment are not yet chosen;
   layer assignment must be fixed before any data are seen. Word-frequency covariates
   must be recorded per concept (needed for the refutation test on prediction 4).
10. **Randomness.** All randomness through `structure_of_truth.seeds.make_rng(seed)`;
    seeds are integers passed explicitly and written into every results file.
11. **Figures.** PNG at 150 dpi via a non-interactive matplotlib backend (Agg).
12. **INDSCAL loss vs. stress.** The ALS objective is *strain* (LS on scalar products),
    not Kruskal stress on distances; only strain is guaranteed monotone. The symmetrised
    solution's strain (`symmetric_strain`) can differ from the ALS strain by round-off-
    level amounts when A ≠ B. Stress on distances is computed afterwards via
    `subject_distances` + `stress.stress`.
13. **Missing data.** Reliability and stress accept NaN (pairwise exclusion). INDSCAL and
    GPA currently do not; imputation vs. weighted ALS is undecided.
14. **INDSCAL dimensionality and restarts.** No automatic choice of `dim` and no
    multi-start yet; local minima are possible with `init="random"`.
15. **Rotation vs. rescaling decomposition** (`decompose.py`). Per subject: classical MDS
    in the known dimensionality k; rescaling magnitude `b_s` = row norm of the
    double-centred subjects × axes matrix of log INDSCAL weights; rotation magnitude
    `θ_s = ||R̂_s − I||_F` where `R̂_s` is the proper rotation of the value-laden block
    fitted *before* the subject's axis scaling (Kiers majorization), after aligning the
    subject on anchor concepts (physical + intersubjective) assumed not to rotate. A GPA
    route (no per-axis model) is kept for comparison. Assumptions: k known; anchor and
    target blocks fixed in advance. Rotation is identifiable only relative to the group
    consensus, not to an absolute "unrotated" space.
16. **Detection thresholds.** Both magnitudes have a positive noise floor. In simulation,
    thresholds come from null-generator replicates at the same noise and S. For real data
    this requires a parametric bootstrap from an estimated noise model — not yet built.
17. **P3 test.** Pre/post fitted jointly; one-sided paired Wilcoxon on per-subject drops;
    "P3 signature" = rotation drop significant AND relative rotation drop > relative
    rescaling drop (paired, one-sided). The literal reading "weights don't drop"
    (non-significance) is reported as the *naive* signature and should not be used.
18. **Simulation parameters** (`sim/generator.py`). 3 × 60 concepts, k = 5, whitened
    shared space; rotation angle θ_s ~ U(0, 0.5 rad) of the value block; log-weight spread
    σ_s ~ U(0, 0.6); noise s.d. = noise × mean distance × layer factor (0.5 / 1.0 / 1.5);
    clarification reduces the targeted component(s) by 50%. Conclusions are conditional on
    these effect sizes.
19. **P1/P2 power simulations** (`sim/power.py`). "Judgments per concept" m = number of
    pairs per concept that *every* subject rates (fixed m-regular pair design; m = 179 is
    the full matrix). P1: 3 cultures, culture displacement τ per layer (equal .3/.3/.3;
    weak .2/.3/.4; strong .1/.3/.5), individual deviation 0.2; convergence = Pearson r of
    culture-mean concept profiles, disattenuated by per-culture split-half SB reliability
    (50 splits, floored at 0.05); test = one-sided Mann–Whitney physical > value-laden
    over concepts (concept as unit; calibration checked by the equal-τ scenario). P2:
    individual deviations of rank 3/5/10/full with equal total variance (5% of ||X||²);
    residual = subjects × pairs minus column means (alignment-free). Rank methods: var80,
    scree (log-eigenvalue elbow), Horn parallel analysis (20 permutations), bi-CV, retest
    CV, and the reliable-variance share (two sessions). P2 "power" = all four true ranks
    classified low (3, 5) vs high (10, full), r̂ = 0 counted as failure.
20. **P6 operationalisation** (`sim/propaganda.py`). Interface = pairs between rewritten
    and unmodified concepts; global fit for (i) = the shared space X as fixed model
    (ratio-level stress; global similarity Procrustes for per-concept residuals), because a
    free re-fit absorbs any rigid rewrite. Rigid signature = within-cluster pairs intact
    (AUC cross > within-cluster) AND cross pairs stressed (AUC cross > rest), thresholds
    = 95th percentiles of the non-rigid-subset and global-spread controls with equal total
    squared displacement. "Rewriting more of the interface" has two readings, both
    simulated: a tapered shell of neighbours moved with the cluster (stress vs X), and a
    fraction q of interface pairs re-judged under the new geometry (free SMACOF fit,
    internal-consistency tension).

