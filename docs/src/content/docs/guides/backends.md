---
title: Backends
description: The NumPy/SciPy reference path, the batched path on cunumpy, and where the execution boundaries are.
---

OpenCR has one reference implementation, NumPy and SciPy: sparse rate matrices, sparse LU
solves, stiff integrators, one set of conditions at a time. Every result in the documentation
and every test is computed there. The batched path does less, in a form that suits a GPU: the
Maxwellian rate coefficients and the quasi-static reduction of **many conditions at once**, as
dense arrays on cunumpy's `xp`.

cunumpy is the array layer: `xp` is NumPy or CuPy for the whole process, chosen with
`cunumpy.set_backend("cupy")` when a GPU is available (and NumPy otherwise). OpenCR's code is
written once against `xp` and does not choose between the two; `opencr.backends.device()` says
where the arrays live.

## What is batched

For a network linear in its states (one heavy reactant per reaction, the case of the hydrogen
model) the rate matrix at one set of conditions is $M = N\,\mathrm{diag}(k\, n_e^{a_e})\, S$,
with $N$ the net stoichiometry, $S$ the selector of each reaction's heavy reactant and $a_e$ the
number of electrons among the reactants. For $B$ pairs $(T_e, n_e)$:

1. the rate coefficients $k$ of every reaction at every $T_e$, shape $(B, R)$, evaluated with
   the same closed-form Maxwellian integrals as the reference
   (`RateEvaluator.coefficients_batched`);
2. the matrices $M$, shape $(B, n, n)$, in one `einsum`;
3. the reduction $C = -M_{QQ}^{-1} M_{QP}$, $M^\mathrm{eff} = M_{PP} + M_{PQ} C$ as one batched
   dense solve, and the photon emissivity coefficients from $C$.

`opencr.backends.batched.batched_scan(model, conditions, electron_temperature, electron_density)`
returns the same variables as `opencr.scan` for a quasi-static Maxwellian scan (`effective_rate`,
`population_coefficient`, `pec`), and `BatchedQuasiStatic(model).run(conditions, T_e, n_e)`
the raw arrays for any list of points. The tests check both against the per-point sparse
reference solves to $10^{-9}$.

## The execution boundaries

They are explicit and in two places only: the inputs are copied to `xp` when a batch is
assembled (`xp.asarray`), and the outputs come back as NumPy arrays (`xp.to_numpy`); a
`BatchedReduction` holds NumPy arrays. Two things always run on the host: a `Function` rate
model (user Python code; its temperatures are copied to the host and the result back) and a
non-Maxwellian EEDF, which plasmacoll integrates per point and which the batched path refuses.

Not batched: the closed and quasi-neutral steady states, the transient solve, and the
analysis of one result. They stay on the reference path.

## The JAX backend (optional)

`opencr.backends.jax_backend.JaxModel(model, conditions)` needs the `jax` extra; nothing in
the core imports JAX. It reuses the reference's prepared rate evaluators with `jax.numpy` as
the array namespace and the structure of the population equations, at the heavy temperatures,
escape factors and prescribed densities of `conditions`:

| Method | What it does |
| --- | --- |
| `rates(T_e)` | the rate coefficients of every reaction at one or many temperatures, jitted |
| `rhs(n, T_e, n_e=None)` | $R(n)$ of the evolving states (quasi-neutral when `n_e` is `None`) |
| `jacobian(n, T_e, n_e=None)` | $\partial R/\partial n$ by forward-mode automatic differentiation, dense |
| `reduction(T_e, n_e)` | the quasi-static reduction of a batch of points with `jax.vmap`, as a `BatchedReduction` |
| `effective_rate(source, target, T_e, n_e)` | a differentiable scalar: `jax.grad` of it is the sensitivity to $T_e$ and $n_e$ |

The AD Jacobian is checked against the analytic sparse one (including the quasi-neutral
terms and reactions between two evolving states) and against finite differences; the
vmapped reduction against the batched path, and the gradient of $S_\mathrm{CR}$ against
finite differences. Two things differ from the reference: a `Function` rate (user Python
code) is tabulated on its `file_temperatures` and interpolated in log-log, as when it is
written to a file (the radiative recombination of the hydrogen model is then within
$10^{-4}$), and a non-Maxwellian EEDF is refused. The first call of each method compiles it;
the [benchmarks](/OpenCR/guides/benchmarks/) separate that from the warm time.
