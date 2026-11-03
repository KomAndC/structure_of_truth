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
  - [ ] residual rank
  - [ ] rotation vs. rescaling decomposition of the residual (needed for P3)
- [ ] Simulations — synthetic ground truth to validate each analysis recovers known structure
- [ ] Toy world — models trained on worlds with different constraints; convergence only for constrained parts
- [ ] P1. Convergence gradient across layers (physical > intersubjective > value-laden; trust/cooperation > concrete institutions cross-culturally)
- [ ] P2. Residual is low-rank (3–5 axes; weights correlate with Moral Foundations)
- [ ] P3. Rotation yields to clarification, weights don't
- [ ] P4. LLMs align with human centroid, closer to WEIRD subjects; survives word-frequency control
- [ ] P5. Equivocation shows up as geometric drift
- [ ] P6. Propaganda is a local deformation with stress at the edge

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

