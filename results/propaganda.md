# Prediction 6 — propaganda as a rigid local deformation with tension at the edge

Run: `python sim/propaganda.py --seed 20261105 --workers 1 --noise 0.0 0.02 0.05 0.1 0.2 0.4`
(30 replicates per cell; tables and figures in `results/propaganda/`:
`signature_power.csv`, `propaganda.csv`, `taper.csv`, `consistency.csv`, `propaganda.png`,
`taper.png`, `consistency.png`). All conclusions are conditional on the assumptions at the end.

## Bottom line

1. **(i) Interface stress — yes, and it does distinguish rigid rewriting from both controls,
   but only through one specific feature.** After a rigid rewrite, the stress against the
   shared space sits on the cross pairs (rewritten–unmodified), while within-cluster pairs
   stay intact. A non-rigid perturbation of the same concepts *also* puts stress on the
   cross pairs (AUC cross > rest ≈ 0.8–0.99 noise-free, about the same as rigid), so
   "stress at the edge" on its own does **not** single out rigid rewriting. What does is
   the *absence* of stress inside the cluster (AUC cross > within-cluster ≈ 0.93–1.0 for
   rigid vs 0.33–0.39 for non-rigid). A perturbation of equal total magnitude spread over
   all concepts shows neither pattern (all AUCs ≈ 0.5). The combined signature (both AUCs
   above the controls' 95th percentiles) has false-alarm rates of 0–7% against both controls.
2. **The signature breaks down with noise much earlier than with cluster size.** At a
   displacement of 0.2 × mean distance it holds (≥ 80% power) up to group-matrix noise 0.05
   for clusters of ≥ 5 concepts and 0.1 for ≥ 10 concepts. At noise 0.2 only an
   80-concept cluster reaches 90%. At noise 0.4 it fails everywhere. Halving the
   displacement (0.1) roughly halves the noise tolerance. Very small clusters (3 concepts)
   are the first to fail, because they have only 3 within-cluster pairs to show intactness.
3. **(ii) "Tension falls as more of the interface is rewritten" is not supported as a
   general rule.** It depends on what "tension" means, and both readings disagree with
   the simple prediction:
   - *Against the shared space* (the reading used for (i)): extending the rewrite to a
     tapered shell of neighbouring concepts **raises** global and local stress for clusters
     of 5–20 concepts (e.g. n_c = 20, noise-free: stress-1 0.036 → 0.047 as the shell grows
     to 2 × n_c). It falls only for an 80-concept cluster (0.056 → 0.040), where the
     rewrite approaches a rigid move of the whole space, which any global alignment
     absorbs. Spreading the deformation makes it smoother, but it also moves more
     concepts away from the shared space.
   - *Internal consistency of the judgments* (each interface pair re-judged under the
     new geometry with probability q, free MDS fit, no reference): tension is an
     **inverted U** in q. It is zero at q = 0, because a rigid rewrite that has not
     reached the interface is invisible: every within-cluster distance is unchanged. It
     is also zero at q = 1 (a consistent new world), and it peaks near q = 0.5. Tension
     therefore falls only *after* about half of the interface has been rewritten. A
     monotone fall from q = 0 is instead the pattern of the **non-rigid** control (for
     large clusters), whose internal distortion clashes with the old interface from the
     start.

## Setup

- Shared space: 180 concepts (3 layers × 60), k = 5, whitened (`sim/generator.py`).
- Rewritten cluster: the n_c ∈ {3, 5, 10, 20, 40, 80} nearest neighbours of a value-laden
  seed concept.
- Rigid rewrite: rotation about the cluster centroid in a random plane plus a translation,
  with RMS displacement = mag × mean distance (mag ∈ {0.1, 0.2}). Half the squared
  displacement comes from the rotation (angle ≤ π/2) and the rest from the translation.
- Controls with **equal total squared displacement**: independent random displacement of
  each cluster member (non-rigid subset), or of every concept (global spread).
- Observation: one group-mean dissimilarity matrix with additive noise s.d. = noise × mean
  distance × layer factor (0.5 / 1 / 1.5). Group noise ≈ per-subject noise / √S. With
  the identifiability run's per-subject noise of 0.5 (single-subject r ≈ 0.53), noise 0.05
  corresponds to S ≈ 100 and noise 0.1 to S ≈ 25.
- Global fit for (i): the shared space X is the model. Ratio-level Kruskal stress of the
  observed matrix against d(X) (one global scale), decomposed per pair. Per-concept
  residuals come from a global similarity Procrustes of classical MDS onto X. A free
  re-fit cannot be used for (i), because any rigidly rewritten world is still exactly
  Euclidean and would fit with zero stress.

## (i) Signature power (rigid rewrites flagged against both controls), mag = 0.2

| group noise \ n_c | 3 | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.02 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.05 | 0.73 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.1 | 0.13 | 0.57 | 0.97 | 1.00 | 1.00 | 1.00 |
| 0.2 | 0.03 | 0.07 | 0.07 | 0.43 | 0.57 | 0.90 |
| 0.4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.17 |

mag = 0.1:

| group noise \ n_c | 3 | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.02 | 0.73 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.05 | 0.27 | 0.17 | 0.93 | 1.00 | 1.00 | 1.00 |
| 0.1 | 0.00 | 0.03 | 0.03 | 0.57 | 0.83 | 0.83 |
| 0.2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.23 | 0.07 |
| 0.4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

False-alarm rates (a control replicate passing both thresholds) are 0–7% in every cell.

Mean statistics at mag = 0.2 (n_c = 20):

| noise | condition | AUC cross > within-cluster | AUC cross > rest | per-concept Procrustes AUC (members) |
|---|---|---|---|---|
| 0 | rigid | 0.97 | 0.95 | 1.00 |
| 0 | non-rigid subset | 0.37 | 0.96 | 1.00 |
| 0 | global spread | 0.50 | 0.50 | 0.50 |
| 0.05 | rigid | 0.70 | 0.71 | 1.00 |
| 0.05 | non-rigid subset | 0.40 | 0.71 | 1.00 |
| 0.05 | global spread | 0.49 | 0.50 | 0.47 |
| 0.1 | rigid | 0.59 | 0.60 | 0.98 |
| 0.1 | non-rigid subset | 0.43 | 0.60 | 0.99 |

A second, local discriminator: on the 10-nearest-neighbour graph of the shared space, the
stress on cross pairs relative to other local pairs stays high for rigid rewrites at every
cluster size (≈ 18–22× at noise 0.02), but for the non-rigid control it shrinks with
cluster size (≈ 18× at n_c = 3, 11× at 10, 5× at 20, 1× at 80). The non-rigid
control's stress spreads over all of the cluster's local pairs, not only the edge, so
for clusters of ≥ 10 the *local* edge concentration also separates rigid from non-rigid.
It is not used in the power table above.

Per-concept alignment residuals **locate** the rewritten concepts (AUC ≈ 1 for rigid and
non-rigid alike, and robust to noise), but they cannot tell a rigid rewrite from a
non-rigid one. For n_c = 80 the global Procrustes partly absorbs the cluster's motion
(AUC 0.77). Rigidity is visible only at the pair level, in the within-cluster pairs.

## (ii) Interface rewriting

Against the shared space (tapered shell of ρ × n_c neighbours, mag 0.2, noise 0), global
stress-1:

| n_c | ρ = 0 | 0.5 | 1 | 2 |
|---|---|---|---|---|
| 5 | 0.019 | 0.021 | 0.023 | 0.029 |
| 20 | 0.036 | 0.039 | 0.041 | 0.047 |
| 80 | 0.056 | 0.052 | 0.043 | 0.040 |

The same direction holds for local (10-NN) stress. At noise 0.1 the changes are at most
0.006 in stress-1 (same signs), small next to a noise-level stress of about 0.10.

Internal consistency (free SMACOF fit, rigid rewrite, mag 0.2): tension = stress-1 in
excess of the fully consistent q = 1 fit. P(tension detectable) = stress above the q = 1
95th percentile:

| noise | n_c | q = 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|---|
| 0 | 20 | 0.000 (0.00) | 0.015 (1.00) | 0.018 (1.00) | 0.016 (1.00) | 0.000 (0.07) |
| 0.05 | 5 | — (0.10) | (0.47) | (0.87) | (0.47) | (0.07) |
| 0.05 | 20 | — (0.17) | (1.00) | (1.00) | (1.00) | (0.07) |
| 0.1 | 20 | — (0.10) | (0.37) | (0.47) | (0.27) | (0.07) |
| 0.1 | 80 | — (0.03) | (0.90) | (0.97) | (0.87) | (0.07) |
| 0.2 | 80 | — (0.10) | (0.33) | (0.43) | (0.30) | (0.07) |

(At the 0.07 cells the q = 1 replicates are compared against their own 95th percentile,
so this is the nominal false-alarm rate.) Inconsistency tension is about as noise-fragile
as the signature in (i). At ≥ 80% power it is detectable for every n_c at noise ≤ 0.02,
for n_c ≥ 10 at 0.05 (q between 0.25 and 0.75), for n_c ≥ 40 at 0.1 (q near 0.25–0.5), and
nowhere at 0.2. While present it is concentrated on
the cross pairs (concentration 2–18× noise-free, 1.2–1.7× at noise 0.05).

## Where it breaks down

- **Noise:** the decisive limit. With mag 0.2, group-matrix noise of 0.05 is fine and
  0.1 is fine for n_c ≥ 10. At 0.2 the signature survives only for very large clusters,
  and at 0.4 it is gone. In design terms this needs a reliable *group* matrix: at
  per-subject reliability ≈ 0.5, about 100 subjects per group for the 0.05 regime.
- **Cluster size:** 3-concept clusters fail first (too few within-cluster pairs). Clusters
  of ≥ 10 are robust down to noise 0.1. Larger clusters are more detectable, not less, up
  to 80 of 180 concepts, the largest size tested. As a cluster grows towards the whole
  space, a rigid rewrite becomes indistinguishable from the shared space's own arbitrary
  orientation and the interface disappears. That limit is reached only beyond the range
  tested here.
- **Magnitude:** a rewrite of 0.1 × mean distance needs about half the noise of a
  0.2 × mean-distance rewrite for the same power.

## Assumptions and limitations

- The analyst knows (or hypothesises) **which concepts were rewritten**. The pair-class
  statistics need the cluster labels. Per-concept residuals can find the cluster blind,
  but blind detection of *rigidity* is not tested.
- The analyst has a **reference shared space** (e.g. the pre-exposure or other-culture
  population) for (i) and for the taper version of (ii). Without one, only the
  internal-consistency version applies.
- Rigid = rotation + translation in the latent space. Real "coherent rewriting" could
  include uniform scaling or be only approximately rigid. A partially rigid case is not
  simulated.
- Equal *total* squared displacement is used to match controls. A per-concept-matched
  global control would be far larger in total and easier to distinguish.
- Group noise is i.i.d. Gaussian per pair with layer factors. Real noise is
  heteroscedastic, bounded and correlated within subjects.
- The 95th-percentile thresholds come from 30 control replicates, so the power estimates
  carry about ±0.1 sampling error.
- My operationalisations of "interface", "rewriting more of the interface" and "tension"
  are listed in SPEC.md, open decision 20. Other readings could give different answers
  for (ii).
