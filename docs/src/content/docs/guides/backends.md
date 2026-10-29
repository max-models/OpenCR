---
title: Backends
description: The NumPy reference, the batched cunumpy path and where the execution boundaries are.
---

OpenCR has one reference implementation, NumPy and SciPy: sparse rate matrices, sparse LU
solves, stiff integrators. Every result in the documentation and every test is computed there.
The accelerated path does less, in a form that suits a GPU: the Maxwellian rate coefficients
and the quasi-static reduction of **many conditions at once**, as dense batched arrays on an
array namespace chosen by the user.

## What is batched

For a network linear in its states (one heavy reactant per reaction, the case of the hydrogen
model) the rate matrix at one set of conditions is $M = N\,\mathrm{diag}(k\, n_e^{a_e})\, S$,
with $N$ the net stoichiometry, $S$ the selector of each reaction's heavy reactant and $a_e$ the
number of electrons among the reactants. For $B$ pairs $(T_e, n_e)$:

1. the rate coefficients $k$ of every reaction at every $T_e$, shape $(B, R)$, evaluated with
   the same closed-form Maxwellian integrals as the reference, written against the array
   namespace (`RateEvaluator.coefficients_batched`);
2. the matrices $M$, shape $(B, n, n)$, in one `einsum`;
3. the reduction $C = -M_{QQ}^{-1} M_{QP}$, $M^\mathrm{eff} = M_{PP} + M_{PQ} C$ as one batched
   dense solve, and the photon emissivity coefficients from $C$.

`opencr.backends.batched.batched_scan(model, conditions, electron_temperature, electron_density, backend)`
returns the same variables as `opencr.scan` for a quasi-static Maxwellian scan (`effective_rate`,
`population_coefficient`, `pec`), and `BatchedQuasiStatic(model, backend).run(conditions, T_e, n_e)`
the raw arrays for any list of points. The tests check both against the per-point sparse
reference solves to $10^{-9}$.

## The backends and their boundaries

| Backend | Namespace | Where |
| --- | --- | --- |
| `"numpy"` | NumPy | host (the reference) |
| `"cunumpy"` | cunumpy | CuPy when a GPU is available and `cunumpy.set_backend("cupy")` was called, NumPy otherwise |

The execution boundaries are explicit and in two places only: the inputs are copied to the
backend when a batch is assembled (`Backend.asarray`), and the outputs come back as NumPy arrays
(`Backend.to_numpy`); a `BatchedReduction` holds NumPy arrays. Two things always run on the
host: a `Function` rate model (user Python code; its temperatures are copied to the host and
the result back) and a non-Maxwellian EEDF, which plasmacoll integrates per point and which the
batched path refuses.

Not batched: the closed and quasi-neutral steady states, the transient solve, and the
analysis of one result. They stay on the reference path. A JAX backend (jitted rates,
automatic-differentiation Jacobians) and the benchmarks that decide whether a GPU is worth it
are later steps.
