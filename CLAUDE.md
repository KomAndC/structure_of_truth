# Instructions for every session

1. Read `SPEC.md` and `STATUS.md` before doing anything else.
2. At the end of the session, update `STATUS.md` (checklist + dated log entry). Record any
   new methodological choice under "Open decisions" in `SPEC.md`.

Conventions:
- All randomness via explicit integer seeds (`structure_of_truth.seeds.make_rng`); no global RNG state.
- Every analysis writes a CSV or JSON **and** a figure to `results/` (`structure_of_truth.io`), with the seed recorded.
- No notebooks. Scripts are plain `.py` files (`sim/` for simulations).
- Run `pytest` before committing.
