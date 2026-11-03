# STATUS

> Every session: read `SPEC.md` and this file **first**; update this file **at the end**
> (tick items, add a dated log entry, note new open decisions in SPEC.md).

Legend: `[ ]` not started · `[~]` in progress · `[x]` done

## Checklist

- [ ] Pipeline — data loading, GPA/INDSCAL alignment, noise ceilings, stress, residual rank
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
