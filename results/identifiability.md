# Identifiability of rotation vs. rescaling, and testability of prediction 3

Run: `python sim/identifiability.py --seed 20261103 --noise 0.1 0.25 0.5 1.0 1.5 --workers 4`
(16 replicates per cell; tables in `results/identifiability/`, auto-generated tables in
`results/identifiability/tables.md`, figures `identifiability.png`, `intervention.png`).
All conclusions are conditional on the simulation assumptions listed at the end.

## Bottom line

**Prediction 3 is testable only when individual subjects' dissimilarity matrices are fairly
reliable, roughly single-subject r ≥ 0.5 against their own noise-free matrix (noise ≤ 0.5
here; Nili lower-bound noise ceiling ≥ ~0.5).** In that regime the INDSCAL-based test
detects "rotation drops, weights don't" in 81–100% of replicates with as few as 20
subjects, and never (0/16) fires when both components drop equally or only the weights drop.
**At single-subject r ≈ 0.27 (noise 1.0) or worse, the rotational component is not
recoverable at any sample size we tried (S ≤ 200): rotation-recovery correlation ≈ 0.0–0.1,
P3 signature detected in ≤ 19% of replicates.** Adding subjects does not help, because the
rotation is a *per-subject* quantity estimated from that subject's single matrix; INDSCAL
pools subjects for the shared space and the weights, not for each subject's rotation.

Two further negative results matter for interpreting a real study:

1. **The refutation branch of P3 ("both drop equally") is not reliably confirmable once
   noise is moderate.** With both components truly reduced by 50%, the estimated relative
   rotation drop is attenuated by the noise floor (0.31 vs 0.47 for the weights at noise
   0.5), so the "both drop equally" verdict is reached in only 0–25% of replicates for
   S ≥ 50 at noise 0.5 (and only 56% at noise 0.25, S = 200): the bias becomes *more*
   significant with more subjects. The bias
   goes against P3, so it does not create false P3 positives, but a real study could
   wrongly conclude "rotation drops *less* than weights". Both magnitudes need noise-floor
   correction (open decision 16) before the equal-drop case can be interpreted.
2. **The two components leak into each other.** Classifying a population as "rotation
   only" or "rescaling only" by null-calibrated detection fails often even at low noise:
   pure rotation (A) produces spurious INDSCAL weight variation (detected in 81–100% of A
   replicates at noise ≤ 0.25; its per-subject correlation with the true rotation is ≈ 0.7
   at noise 0.1), and pure rescaling (B) inflates the rotation estimate at noise 0.25–0.5
   (false rotation detection in 62–100% of B replicates; corr(θ̂, b_true) ≈ 0.15–0.5).
   Only the *within-subject change* design of P3 (pre vs. post, joint fit) is robust to
   this, because the leak is roughly constant between sessions and cancels in the paired
   comparison. Do not use the cross-sectional magnitudes to claim one component exists
   without the other.

## Setup

- 180 concepts (3 layers × 60), latent dimension k = 5, whitened shared space.
- Subject differences: (A) per-subject rotation of the value-laden block by angle
  θ ~ U(0, 0.5 rad) in a random 5-D generator; (B) per-axis log-weights with spread
  σ ~ U(0, 0.6); (C) both; null: neither.
- One noisy dissimilarity matrix per subject; additive noise s.d. = noise × mean distance ×
  layer factor (0.5 / 1.0 / 1.5 for physical / intersubjective / value-laden).
- Estimators (`src/structure_of_truth/decompose.py`): classical MDS per subject; INDSCAL
  for shared space + weights; rotation of the value block fitted in the weighted space
  after anchoring on physical + intersubjective concepts; magnitudes θ̂ = ||R̂ − I||_F and
  b̂ = norm of double-centred log weights. GPA route for comparison.
- Clarification: pre and post measurement of the same subjects; the targeted component(s)
  reduced by 50%; INDSCAL fitted jointly to all 2S matrices; paired one-sided Wilcoxon
  tests. "P3 signature" = rotation drop significant AND relative rotation drop > relative
  weight drop (both p < .05).

How noise maps onto measurable reliability (mean over all cells):

| noise | single-subject r (vs own noise-free matrix) | Nili lower bound (vs mean of others) |
|---|---|---|
| 0.1 | 0.95 | 0.93 |
| 0.25 | 0.78 | 0.76 |
| 0.5 | 0.53 | 0.51 |
| 1.0 | 0.27 | 0.29 |
| 1.5 | 0.15 | 0.25 |

The Nili lower bound is observable in real data, so it is the quantity to compare against.

## Recovery accuracy (INDSCAL route, generator C, mean Spearman r over 16 replicates)

| noise | S=20 rot / resc | S=50 | S=100 | S=200 |
|---|---|---|---|---|
| 0.1 | 0.96 / 0.99 | 0.99 / 1.00 | 0.99 / 1.00 | 0.99 / 1.00 |
| 0.25 | 0.94 / 0.99 | 0.96 / 0.99 | 0.97 / 0.99 | 0.97 / 1.00 |
| 0.5 | 0.65 / 0.97 | 0.71 / 0.97 | 0.69 / 0.98 | 0.73 / 0.98 |
| 1.0 | −0.02 / 0.88 | 0.13 / 0.92 | 0.09 / 0.93 | 0.09 / 0.93 |
| 1.5 | −0.02 / 0.46 | −0.03 / 0.69 | 0.04 / 0.74 | 0.00 / 0.74 |

- Rescaling is far more robust than rotation: b̂ is still recovered at r ≈ 0.9 at noise 1.0.
- Even noise-free, rotation is identifiable only *relative to the group consensus*, not to
  an absolute unrotated space; recovery tops out at ≈ 0.96–0.99.
- GPA is clearly worse for rescaling (it has no per-axis model: its principal axes are not
  the weighting axes in a whitened space) and confounds rescaling with rotation
  (corr(θ̂_GPA, b_true) ≈ 0.5–0.85 under pure B, vs 0.1–0.5 for INDSCAL).

## Prediction 3 intervention (INDSCAL route, P(signature) over 16 replicates)

| noise | S | rotation_only (power) | both_equal (false +) | rescaling_only (false +) | none (false +) |
|---|---|---|---|---|---|
| 0.1 | 20–200 | 1.00 | 0.00 | 0.00 | 0.00–0.06 |
| 0.25 | 20–200 | 1.00 | 0.00 | 0.00 | 0.00–0.06 |
| 0.5 | 20 | 0.81 | 0.00 | 0.00 | 0.00 |
| 0.5 | 50 | 0.88 | 0.00 | 0.00 | 0.06 |
| 0.5 | 100–200 | 1.00 | 0.00 | 0.00 | 0.00–0.06 |
| 1.0 | 20–200 | 0.00–0.19 | 0.00 | 0.00 | 0.00 |
| 1.5 | 20–200 | 0.00–0.06 | 0.00 | 0.00 | 0.00–0.06 |

The *naive* reading ("rotation drop significant, weight drop not significant") has the same
power here but is unprincipled: it is satisfied whenever the weight test is underpowered.
Use the paired relative-drop comparison.

## Where prediction 3 is testable

| single-subject reliability (Nili lower bound) | verdict |
|---|---|
| ≥ ~0.75 | testable with S ≈ 20; both components well recovered |
| ~0.5 | testable: power 81% at S = 20, 88% at S = 50, 100% at S ≥ 100; equal-drop refutation biased |
| ≤ ~0.3 | **not testable** at S ≤ 200: rotation unrecoverable; more subjects do not help — need more reliable *per-subject* matrices (more repetitions per pair, or averaging within subject) |

## Assumptions and limitations

- The dimensionality (k = 5), the block suspected of rotating (value-laden) and the anchor
  set assumed not to rotate (physical + intersubjective) are known to the analyst. A
  misspecified anchor set would bias θ̂ and is not simulated.
- Effect sizes are fixed (θ up to 0.5 rad; log-weight spread up to 0.6; 50% reduction by
  clarification). Smaller real effects need more reliable data.
- Noise is i.i.d. Gaussian per pair with a layer-specific s.d.; real rating noise (scale
  use, ties, bounded scales) is not modelled. Complete matrices are assumed (open
  decision 13).
- Null-calibrated detection thresholds use 15–16 null replicates and a normal
  approximation; empirical null false-positive rates are 0–19% (nominal 5%), so the
  cross-sectional detection rates are rough. Real data needs a parametric bootstrap
  (open decision 16).
- Wilcoxon tests are per replicate at α = .05 with no multiple-comparison correction.
