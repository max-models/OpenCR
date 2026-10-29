---
title: Benchmarks
description: What the reference path, the batched cunumpy path and the JAX backend cost, and how to measure it.
---

`benchmarks/benchmark.py` in the repository times the pieces of a solve on the atomic hydrogen
model with scope-profiler regions, each repeated after one first call, so that the **first**
column holds the JIT compilation of the JAX cases and the **median** the warm time:

```bash
python benchmarks/benchmark.py                 # markdown table
python benchmarks/benchmark.py --json          # the same as JSON
python benchmarks/benchmark.py --n-max 12 --points 100 --repeat 3 --no-jax
```

## Results

Apple M-series CPU (arm64), `hydrogen/atomic/basic` with $n_\mathrm{max} = 20$ (21 states,
630 reactions), scans over a 32 × 32 grid of $(T_e, n_e)$:

| case | first (ms) | median (ms) |
|---|---:|---:|
| construction | 37.28 | 34.55 |
| rates | 15.92 | 15.53 |
| jacobian | 0.28 | 0.08 |
| quasi-static | 19.68 | 19.35 |
| full quasi-neutral | 48.67 | 48.71 |
| transient | 21.51 | 20.47 |
| scan (reference, 1024 points) | 23899.44 | 23933.61 |
| batched scan (cunumpy, 1024 points) | 1542.40 | 1502.39 |
| jax rates (1024 temperatures) | 2334.09 | 508.77 |
| jax reduction (1024 points) | 1739.79 | 516.52 |
| jax jacobian | 903.37 | 0.47 |

What the numbers say:

- **One solve is milliseconds.** Construction (the cross-section fits and the exact Maxwellian
  integrals prepared once), one rate evaluation, a quasi-static solve and a transient to
  $10^{-2}$ s are each 15–50 ms; the sparse Jacobian 0.1 ms.
- **The reference scan costs 23 ms per point**, almost all of it the rate evaluation, which
  loops over the reactions in Python with one closed-form integral each.
- **The batched path is 16× faster on the CPU** (1.5 ms per point): the Maxwellian integrals of
  all cross sections are one padded array evaluated at once, in chunks of 32 temperatures,
  and the reduction is one batched dense solve.
- **JAX compiles in about 2 s** for this model (it was a minute before the integrals were
  vectorised: the trace had one closure per reaction) and then runs the 1024-point scan in
  0.5 s. On the CPU that is 3× faster than cunumpy and 45× faster than the reference; the
  AD Jacobian of the population equations takes 0.5 ms once compiled, 5× the sparse analytic one.
- **No GPU result is claimed.** cunumpy runs the same code on CuPy after
  `cunumpy.set_backend("cupy")` and JAX on its default device; the script measures end to end
  with the copies included. Run it there before expecting a speed-up: the per-point work is
  small (a 19 × 19 solve), so a GPU pays off only for large batches.

Profiling beyond these cases uses scope-profiler's opt-in regions
(`opencr.rate_coefficients`, `opencr.solve_steady`, `opencr.solve_transient`,
`opencr.batched_matrices`, `opencr.batched_reduction`):
`scope-profiler run --enable-tag opencr -o prof.h5 script.py`, then
`scope-profiler inspect prof.h5 --sort total`.
