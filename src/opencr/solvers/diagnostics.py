"""What a solve reports: convergence, residuals, conservation and positivity.

Every solve returns a :class:`SolverReport`. A solve that does not converge
raises :class:`ConvergenceError` with the report attached; an underdetermined
system (a conserved element without a total) raises
:class:`UnderdeterminedError`; a singular one
:class:`~opencr.kinetics.effective.SingularSystemError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from opencr.kinetics.effective import SingularSystemError
from opencr.kinetics.system import PopulationSystem

__all__ = [
    "ConvergenceError",
    "SingularSystemError",
    "SolverReport",
    "UnderdeterminedError",
    "relative_residual",
]


class UnderdeterminedError(ValueError):
    """The steady state is not unique: a conserved quantity has no given total."""


@dataclass
class SolverReport:
    """The diagnostics of one solve.

    Attributes:
        mode: ``"quasi-static"`` or ``"full"``.
        method: The numerical method, e.g. ``"sparse LU"`` or ``"Newton"``.
        converged: Whether the tolerances were met.
        iterations: Newton iterations or integrator steps.
        residual: The largest ``|R_i|`` relative to the gross flow through state
            ``i`` (the sum of all its gains and losses), see :func:`relative_residual`.
        conservation: The relative error of each conserved total.
        minimum: The smallest density divided by the largest.
        clipped: States whose small negative densities (round-off) were set to zero.
        messages: Anything worth knowing, in sentences.
    """

    mode: str
    method: str
    converged: bool = True
    iterations: int = 0
    residual: float = 0.0
    conservation: dict[str, float] = field(default_factory=dict)
    minimum: float = 0.0
    clipped: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return the report as a few lines of text."""
        lines = [
            f"{self.mode} solve by {self.method}: "
            + ("converged" if self.converged else "NOT converged")
            + f" in {self.iterations} iterations",
            f"  relative residual {self.residual:.2e}, min/max density {self.minimum:.2e}",
        ]
        if self.conservation:
            worst = max(self.conservation.values())
            lines.append(
                f"  conservation error {worst:.2e} ({', '.join(self.conservation)})"
            )
        if self.clipped:
            lines.append(f"  clipped round-off negatives: {', '.join(self.clipped)}")
        lines += [f"  {m}" for m in self.messages]
        return "\n".join(lines)


class ConvergenceError(RuntimeError):
    """A solve did not converge; ``report`` says how far it got."""

    def __init__(self, message: str, report: SolverReport) -> None:
        """Keep the report and append its summary to the message."""
        self.report = report
        super().__init__(f"{message}\n{report.summary()}")


def relative_residual(system: PopulationSystem, n: np.ndarray) -> float:
    """Return ``max_i |R_i| / (sum_r |N_ir| w_r)``, the residual relative to the gross flow."""
    w = system.reaction_rates(n)
    gross = abs(system.stoichiometry) @ np.abs(w)
    residual = np.abs(system.stoichiometry @ w)
    scale = np.where(gross > 0.0, gross, 1.0)
    return float(np.max(residual / scale, initial=0.0))
