"""Time evolution of the populations with stiff integrators.

:func:`solve_transient` integrates ``dn/dt = R(n, t)`` for the evolving states
of a :class:`~opencr.kinetics.system.PopulationSystem` from initial densities
with SciPy's implicit methods (BDF by default, or Radau) and the analytic sparse
Jacobian of the system. The rates span 1e-3 to 1e9 s^-1 in a CR model, so the
problem is stiff; explicit methods are refused.

The conditions are prescribed: constant (one system), or given as a function
of time (``system_at(t)`` returns the system of that moment, built by
:meth:`~opencr.model.Model.solve_transient` from ``conditions(t)``). The
report records the integrator's statistics, the drift of every element total
the evolving states conserve (the particle balance; the charge balance holds
by construction with a given ``n_e`` or quasi-neutrality) and the smallest
density reached.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sps
import scope_profiler as sp
from scipy.integrate import solve_ivp

from opencr._profiling import PROFILE_TAGS
from opencr.kinetics.system import PopulationSystem

__all__ = ["METHODS", "TransientReport", "solve_transient"]

#: The stiff integrators of :func:`scipy.integrate.solve_ivp`.
METHODS = ("BDF", "Radau", "LSODA")


@dataclass
class TransientReport:
    """The diagnostics of a time integration.

    Attributes:
        method: The integrator.
        success: Whether the integration reached the final time.
        evaluations: Right-hand-side evaluations.
        jacobians: Jacobian evaluations.
        factorisations: LU factorisations of the iteration matrix.
        conservation: The largest relative drift of each conserved total over
            the output times.
        minimum: The smallest density reached divided by the largest.
        messages: Anything worth knowing, in sentences.
    """

    method: str
    success: bool = True
    evaluations: int = 0
    jacobians: int = 0
    factorisations: int = 0
    conservation: dict[str, float] = field(default_factory=dict)
    minimum: float = 0.0
    messages: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return the report as a few lines of text."""
        lines = [
            f"transient solve by {self.method}: "
            + ("reached the final time" if self.success else "FAILED")
            + f" with {self.evaluations} evaluations, {self.jacobians} Jacobians "
            f"and {self.factorisations} factorisations",
            f"  min/max density {self.minimum:.2e}",
        ]
        if self.conservation:
            worst = max(self.conservation.values())
            lines.append(
                f"  conservation drift {worst:.2e} ({', '.join(self.conservation)})"
            )
        lines += [f"  {m}" for m in self.messages]
        return "\n".join(lines)


@sp.profile("opencr.solve_transient", tags=PROFILE_TAGS)
def solve_transient(
    system: PopulationSystem,
    initial: np.ndarray,
    times: np.ndarray,
    method: str = "BDF",
    rtol: float = 1.0e-6,
    atol: float | None = None,
    system_at: Callable[[float], PopulationSystem] | None = None,
) -> tuple[np.ndarray, TransientReport]:
    """Return the densities of the evolving states at ``times`` and the report.

    Args:
        system: The population equations (the structure, and the rates when
            ``system_at`` is not given).
        initial: The densities of the evolving states at ``times[0]`` (m^-3).
        times: The output times in s, increasing; the integration runs from
            the first to the last.
        method: One of :data:`METHODS`.
        rtol: The relative tolerance of the integrator.
        atol: The absolute tolerance in m^-3; by default ``1e-12`` times the
            largest density of the problem (initial, prescribed or a total).
        system_at: For time-dependent conditions, the system at a time ``t``
            (same network and evolving states); called at every evaluation.

    Returns:
        The densities with shape ``(len(times), system.size)`` and the report.

    Raises:
        ValueError: An explicit method, or times that do not increase.
        RuntimeError: The integrator failed (the report is in the message).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}: the problem is stiff")
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be increasing, with at least two values")
    initial = np.asarray(initial, dtype=float)
    if initial.shape != (system.size,):
        raise ValueError(f"initial needs {system.size} densities")
    if np.any(initial < 0.0):
        raise ValueError("initial densities must be >= 0")
    scale = (
        max(
            float(initial.max(initial=0.0)),
            max(system.conditions.densities.values(), default=0.0),
            max(system.conditions.totals.values(), default=0.0),
        )
        or 1.0
    )
    if atol is None:
        atol = 1.0e-12 * scale

    current = system if system_at is None else system_at(float(times[0]))
    last_time = float(times[0])

    def at(t: float) -> PopulationSystem:
        nonlocal current, last_time
        if system_at is not None and t != last_time:
            current, last_time = system_at(float(t)), float(t)
        return current

    def rhs(t: float, n: np.ndarray) -> np.ndarray:
        return at(t).rhs(np.maximum(n, 0.0))

    def jacobian(t: float, n: np.ndarray) -> sps.csr_matrix:
        return at(t).jacobian(np.maximum(n, 0.0))

    sparsity = system.jacobian(np.ones(system.size))
    sparsity.data[:] = 1.0
    options: dict[str, Any] = {"jac": jacobian} if method != "LSODA" else {}
    if method == "BDF" or method == "Radau":
        options["jac_sparsity"] = sparsity
    solution = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        initial,
        method=method,
        t_eval=times,
        rtol=rtol,
        atol=atol,
        **options,
    )
    report = TransientReport(
        method,
        success=bool(solution.success),
        evaluations=int(solution.nfev),
        jacobians=int(getattr(solution, "njev", 0) or 0),
        factorisations=int(getattr(solution, "nlu", 0) or 0),
    )
    if not solution.success:
        report.messages.append(str(solution.message))
        raise RuntimeError(f"the transient solve failed\n{report.summary()}")
    densities = np.asarray(solution.y.T, dtype=float)
    negative = densities < 0.0
    if np.any(negative):
        worst = float(densities.min() / max(densities.max(), 1e-300))
        report.messages.append(
            f"negative densities down to {worst:.1e} of the largest set to zero "
            "(tighten atol if they matter)"
        )
        densities = np.where(negative, 0.0, densities)
    names, matrix = system.conservation()
    for name, row in zip(names, matrix, strict=True):
        totals = densities @ row
        start = float(totals[0])
        report.conservation[name] = float(
            np.max(np.abs(totals - start)) / max(abs(start), 1e-300)
        )
    largest = float(densities.max(initial=0.0))
    report.minimum = float(densities.min() / largest) if largest > 0 else 0.0
    return densities, report
