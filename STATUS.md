# STATUS

> Every session: read `SPEC.md` and this file **first**; update this file **at the end**
> (tick items, add a dated log entry, note new open decisions in SPEC.md).

Legend: `[ ]` not started · `[~]` in progress · `[x]` done

## Checklist

- [~] Pipeline — data loading, GPA/INDSCAL alignment, noise ceilings, stress, residual rank
  - [ ] data loading
  - [x] GPA (`procrustes.py`) — tested: noise-free recovery, invariance, monotone loss, matches `scipy.spatial.procrustes` for 2 subjects
  - [x] INDSCAL ALS (`indscal.py`) — tested: noise-free recovery (2-D, 3-D), normalisation, strain never increases (svd + random init, 5 seeds), invariance to rotation/scale of inputs
  - [x] split-half noise ceiling per concept + dataset level (`reliability.py`)
  - [x] stress-1 per concept / per pair (`stress.py`)
  - [~] residual rank — estimators exist in `sim/power.py` (cv, retest_share, parallel, scree); not yet moved into `src/`
  - [x] rotation vs. rescaling decomposition (`decompose.py`) — tested; validated in `results/identifiability.md`
  - [ ] noise-floor correction / parametric bootstrap for θ̂, b̂ (SPEC 16) — needed before real-data P3
- [~] Simulations — generator (`sim/generator.py`); identifiability (P3), power (P1, P2), propaganda (P6) done; P4, P5 not simulated
- [ ] Toy world — models trained on worlds with different constraints; convergence only for constrained parts
- [~] P1. Convergence gradient across layers (physical > intersubjective > value-laden; trust/cooperation > concrete institutions cross-culturally) — *simulated*: raw gradient is a reliability artifact; corrected test needs layer reliability ≥ ~0.5; design table in `results/power.md`. No data. Trust-vs-institutions contrast not simulated.
- [~] P2. Residual is low-rank (3–5 axes; weights correlate with Moral Foundations) — *simulated*: needs 2 sessions, ~100 subjects × ~8,000 ratings at r≈0.5; partially powered designs falsely confirm low rank. No data. Moral-Foundations link not simulated.
- [~] P3. Rotation yields to clarification, weights don't — *simulated*: testable only at single-subject r ≥ ~0.5 (`results/identifiability.md`). No data.
- [ ] P4. LLMs align with human centroid, closer to WEIRD subjects; survives word-frequency control
- [ ] P5. Equivocation shows up as geometric drift
- [~] P6. Propaganda is a local deformation with stress at the edge — *simulated*: rigid signature = intact within-cluster pairs, works at group noise ≤ 0.05–0.1; "tension falls with interface rewriting" not supported as stated (`results/propaganda.md`). No data.

## Log

- 2026-11-03 — Repo scaffolded: pyproject, `src/structure_of_truth` (seeded RNG,
  results I/O, generalized Procrustes, split-half noise ceiling) with tests. No
  pipeline, simulations or analyses yet; all checklist items not started.
- 2026-11-03 — Implemented GPA, INDSCAL (ALS), split-half reliability (per concept +
  dataset, with Nili bounds) and stress-1 decompositions; 48 tests pass (pytest, run on
  Python 3.13 in the container; 3.11 not separately tested). These are library functions
  only — no analysis has been run, so nothing written to `results/` yet. New open
  decisions 12–14 in SPEC.md. Next: residual-rank estimator and rotation/rescaling
  decomposition, then simulations validating them.
- 2026-11-03 — Simulation session (3 requests). Added `decompose.py` (rotation/rescaling
  magnitudes), `sim/generator.py`, and three experiments, each with CSVs + figures +
  write-up: `results/identifiability.md` (P3 testable only at single-subject r ≥ ~0.5;
  rotation unrecoverable at r ≈ 0.27 for any S ≤ 200; equal-drop refutation biased by
  noise floor; cross-sectional component leakage), `results/power.md` (P1 cheap after
  noise-ceiling correction, raw convergence gradient is mostly reliability artifact;
  P2 needs two sessions and ~8,000 ratings/subject — infeasible as specified; partially
  powered P2 designs falsely report low rank; var80 retired), `results/propaganda.md`
  (rigid-rewrite signature distinguishes from controls via intact within-cluster pairs,
  breaks down at group noise ~0.1–0.2 and for 3-concept clusters; tension does not
  simply fall with interface rewriting). INDSCAL internals vectorised (~10×). SPEC open
  decisions 6 (revised), 15–20. 73 tests pass (Python 3.13). Next: noise-floor bootstrap
  (SPEC 16), move rank estimators into `src/`, design a reduced-concept or triplet-based
  P2 study and simulate it; P4/P5 simulations.

