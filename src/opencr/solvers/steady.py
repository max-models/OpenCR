"""Steady states of the population equations.

:func:`solve_steady` finds ``R(n) = 0`` for the evolving states of a
:class:`~opencr.kinetics.system.PopulationSystem`:

* **Linear systems** (one evolving reactant per reaction, ``n_e`` given) are
  solved directly by sparse LU with one step of iterative refinement.
* **Conserved elements.** When the evolving states exchange an element only
  among themselves (a closed system), ``R = 0`` is singular; one equation per
  conserved element is replaced by its total from ``totals``. A conserved
  element without a total raises :class:`~opencr.solvers.diagnostics.UnderdeterminedError`.
* **Nonlinear systems** (quasi-neutral ``n_e``, two evolving reactants) use
  Newton's method with backtracking on the same equations, started from a
  pseudo-transient continuation: implicit-Euler steps ``(I/dt - J) dx = R``
  whose ``dt`` grows as the residual falls (switched evolution relaxation),
  which follows the physical relaxation and keeps the totals of a closed
  system.

Every solve checks the relative residual, the conservation of the totals and
the positivity of the densities, and returns a
:class:`~opencr.solvers.diagnostics.SolverReport`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import scope_profiler as sp

from opencr._profiling import PROFILE_TAGS
from opencr.kinetics.effective import SingularSystemError
from opencr.kinetics.system import PopulationSystem
from opencr.solvers.diagnostics import (
    ConvergenceError,
    SolverReport,
    UnderdeterminedError,
    relative_residual,
)

__all__ = ["constraint_rows", "solve_steady"]


def constraint_rows(
    system: PopulationSystem, totals: Mapping[str, float] | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return the rows to replace, the constraint matrix and the totals.

    For each conserved element the equation of the state with the most atoms
    of it (the first such state not yet used) is replaced.

    Raises:
        UnderdeterminedError: A conserved element has no total.
    """
    names, matrix = system.conservation()
    totals = dict(totals or {})
    missing = [n for n in names if n not in totals]
    if missing:
        raise UnderdeterminedError(
            f"the evolving states conserve {missing}: the steady state is not unique. Give "
            "the totals in Conditions.totals (m^-3), or prescribe a state's density."
        )
    rows: list[int] = []
    for row in matrix:
        for candidate in np.argsort(-row, kind="stable"):
            if row[candidate] > 0 and candidate not in rows:
                rows.append(int(candidate))
                break
    values = np.array([totals[n] for n in names], dtype=float)
    return np.array(rows, dtype=int), matrix, values, names


def _augmented(system, rows, matrix, values):
    def residual(n: np.ndarray) -> np.ndarray:
        r = system.rhs(n)
        if rows.size:
            r[rows] = matrix @ n - values
        return r

    def jacobian(n: np.ndarray) -> sps.csr_matrix:
        j = system.jacobian(n).tolil()
        for k, row in enumerate(rows):
            j.rows[row] = list(np.flatnonzero(matrix[k]))
            j.data[row] = list(matrix[k][np.flatnonzero(matrix[k])])
        return j.tocsr()

    return residual, jacobian


@sp.profile("opencr.solve_steady", tags=PROFILE_TAGS)
def solve_steady(
    system: PopulationSystem,
    totals: Mapping[str, float] | None = None,
    initial: np.ndarray | None = None,
    mode: str = "quasi-static",
    rtol: float = 1.0e-9,
    max_iterations: int = 200,
) -> tuple[np.ndarray, SolverReport]:
    """Return the steady densities of the evolving states and the report.

    Args:
        system: The population equations.
        totals: The total density of each conserved element (m^-3).
        initial: A starting point for a nonlinear solve; by default the totals
            are put in the lowest-energy state of each element, a tenth of it
            ionized when ``n_e`` is quasi-neutral.
        mode: The name of the solve for the report.
        rtol: The tolerance of the relative residual.
        max_iterations: Newton plus continuation iterations.

    Raises:
        UnderdeterminedError: A conserved element has no total.
        SingularSystemError: The linear system is singular.
        ConvergenceError: A nonlinear solve did not converge.
    """
    rows, matrix, values, names = constraint_rows(system, totals)
    residual, jacobian = _augmented(system, rows, matrix, values)
    if system.is_linear:
        n = _linear(system, residual, jacobian)
        report = SolverReport(mode, "sparse LU", iterations=1)
    else:
        start = _initial(system, matrix, values, names) if initial is None else initial
        n, report = _nonlinear(
            system, residual, jacobian, rows, matrix, values,
            np.asarray(start, dtype=float), mode, max_iterations, rtol,
        )  # fmt: skip
    n = _clip(system, n, report)
    report.residual = relative_residual(system, n)
    report.conservation = {
        name: float(abs(matrix[k] @ n - values[k]) / max(values[k], 1e-300))
        for k, name in enumerate(names)
    }
    report.minimum = float(n.min() / n.max()) if n.size and n.max() > 0 else 0.0
    if report.residual > max(rtol, 1e-6):
        report.converged = False
        report.messages.append(
            f"relative residual {report.residual:.2e} above tolerance"
        )
        raise ConvergenceError("the steady state did not converge", report)
    return n, report


def _linear(system, residual, jacobian) -> np.ndarray:
    zero = np.zeros(system.size)
    a = jacobian(zero).tocsc()
    b = -residual(zero)
    diagonal = a.diagonal()
    lossless = [system.evolving[i] for i in np.flatnonzero(diagonal == 0.0)]
    if lossless:
        raise SingularSystemError(
            f"the states {lossless} have no loss process; their steady state is undefined. "
            "Prescribe their densities or add their losses."
        )
    try:
        lu = spla.splu(a)
    except RuntimeError as error:
        raise SingularSystemError(
            f"the steady-state matrix is singular ({error}): a group of states has no path "
            "out, or a conserved quantity is not constrained"
        ) from error
    n = lu.solve(b)
    return n + lu.solve(b - a @ n)


def _initial(system, matrix, values, names) -> np.ndarray:
    n = np.zeros(system.size)
    energies = np.array([system.network.state(s).energy for s in system.evolving])
    for k, _ in enumerate(names):
        holders = np.flatnonzero(matrix[k])
        neutral = [i for i in holders if system.charges[i] == 0] or list(holders)
        positive = [i for i in holders if system.charges[i] > 0]
        target = min(neutral, key=lambda i: energies[i])
        share = values[k] / matrix[k][target]
        if system.conditions.quasi_neutral and positive:
            ion = min(positive, key=lambda i: energies[i])
            n[ion] += 0.1 * values[k] / matrix[k][ion]
            share *= 0.9
        n[target] += share
    n += 1.0e-12 * max(values.max(initial=0.0), 1.0)
    return n


def _nonlinear(
    system, residual, jacobian, rows, matrix, values, n, mode, max_iterations, rtol
):
    report = SolverReport(mode, "pseudo-transient continuation + Newton")
    free = np.ones(system.size)
    free[rows] = 0.0  # the constraint rows get no time derivative

    def converged(x: np.ndarray) -> float:
        error = relative_residual(system, x)
        if rows.size:
            error = max(error, float(np.max(np.abs(matrix @ x - values) / values)))
        return error

    def norm(x: np.ndarray) -> float:
        # the residual in s^-1 per particle: falls steadily towards the solution
        return float(np.linalg.norm(residual(x)) / max(np.abs(x).sum(), 1e-300))

    current = norm(n)
    dt = 1.0 / max(np.abs(system.jacobian(n).diagonal()).max(initial=0.0), 1.0)
    for iteration in range(1, max_iterations + 1):
        report.iterations = iteration
        if converged(n) < rtol:
            break
        a = (sps.diags(free / dt) - jacobian(n)).tocsc()
        try:
            step = spla.spsolve(a, residual(n))
        except RuntimeError:  # pragma: no cover - a singular continuation step
            step = np.full(system.size, np.nan)
        if not np.all(np.isfinite(step)):
            dt *= 0.1
            continue
        crossing = (step < 0.0) & (n + step < 0.0)
        fraction = 1.0
        if np.any(crossing):
            # damp the step so that no density turns negative
            fraction = float(np.min(0.9 * n[crossing] / -step[crossing]))
        trial = np.maximum(n + max(fraction, 1e-6) * step, 0.0)
        new = norm(trial)
        if new > 10.0 * current and fraction == 1.0:
            dt *= 0.25  # reject a step that blows the residual up
            continue
        # switched evolution relaxation, at least doubling dt after a good step
        dt *= float(np.clip(2.0 * current / max(new, 1e-300), 0.5, 1.0e3))
        n, current = trial, new
    else:
        report.converged = converged(n) < rtol
    return n, report


def _clip(system, n, report) -> np.ndarray:
    largest = np.abs(n).max(initial=0.0)
    small = (n < 0.0) & (n > -1.0e-8 * largest)
    report.clipped = [system.evolving[i] for i in np.flatnonzero(small)]
    n = np.where(small, 0.0, n)
    if np.any(n < 0.0):
        bad = [system.evolving[i] for i in np.flatnonzero(n < 0.0)]
        report.messages.append(f"negative densities: {bad}")
    return n
