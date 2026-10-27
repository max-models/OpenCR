---
title: Solvers and troubleshooting
description: How the solves work, what the report says, and what to do when a solve fails.
---

## How a steady state is found

- **Linear systems** (each reaction has at most one evolving reactant and $n_e$ is given, the
  case of every quasi-static hydrogen solve) are solved directly with a sparse LU
  factorisation and one step of iterative refinement.
- **Closed systems**: for each element the evolving states conserve, one equation is replaced
  by the element's total from `Conditions.totals`.
- **Quasi-neutral systems that are linear for a given $n_e$** (atomic hydrogen with
  `electron_density=None`) solve quasi-neutrality as a one-dimensional root: for each trial
  $n_e$ a sparse linear solve gives the populations and their charge, and Brent's method finds
  the $n_e$ that equals it, in $\log n_e$. A root exists between a vanishing $n_e$ and the
  largest possible charge, so this does not depend on a starting point.
- **Other nonlinear systems** (reactions between two evolving states, or a given `initial`) start with
  pseudo-transient continuation — implicit-Euler steps whose time step grows as the residual
  falls — which follows the physical relaxation and keeps the totals, and end in Newton's method.
  Densities are kept non-negative by damping the steps. Pass `initial={...}` to start elsewhere.

Rates are evaluated exactly for Maxwellian electrons: plasmacoll's cross sections are linear
between their table points, so the Maxwellian average is a sum of closed-form integrals. Any
other energy distribution (`Conditions.from_eedf`) is integrated on its own energy grid.

## The report

Every result has `result.report` (`SolverReport`): the method, iterations, the **relative
residual** (the largest net rate of a state divided by its gross gain-plus-loss rate), the
relative error of every conserved total, the smallest density relative to the largest, and any
round-off negatives set to zero. `report.summary()` prints it.

## When a solve fails

| Error | Meaning | What to do |
| --- | --- | --- |
| `UnderdeterminedError` | the evolving states conserve an element without a total | give `Conditions.totals`, or prescribe a state's density |
| `MissingDensityError` | a prescribed state has no density | add it to `Conditions.densities` |
| `ValueError: ... needs the densities of the drivers` | quasi-static solve without the drivers | give them, or use `mode="full"` |
| `SingularSystemError: ... no loss process` | a state can only be filled | prescribe it, or add its losses |
| `SingularSystemError: ... singular` | a group of states has no path out | check the network for missing reactions |
| `ConvergenceError` | a nonlinear solve did not reach the tolerance | start from `initial=`, raise `rtol`, or check that a positive steady state exists (a source with no loss diverges) |
