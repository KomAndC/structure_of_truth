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
3. **Alignment method.** Generalized Procrustes is implemented first (rotation +
   reflection + isotropic scale, reflections allowed, deterministic init on subject 0).
   INDSCAL (for per-axis weights) is still to be implemented; GPA alone cannot separate
   rotation from rescaling, so prediction 3 needs both.
4. **Noise ceiling definition.** Per-concept, row-wise Spearman split-half reliability over
   random subject halves (default 200 seeded splits), Spearman–Brown corrected (factor 2).
   Alternatives not chosen: lower/upper RSA ceilings (Nili et al. 2014), Pearson.
5. **Stress variant.** Kruskal stress-1 (not raw stress or SSTRESS).
6. **Residual-rank criterion.** 80% variance threshold plus a parallel-analysis null;
   threshold to be confirmed on simulations before applying to real data.
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
