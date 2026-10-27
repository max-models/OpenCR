"""The result of a steady-state solve.

A :class:`Result` keeps the model, the conditions, the rate coefficients, the
population equations and the densities of every state, with the solver's
report.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from opencr.core.conditions import Conditions
from opencr.kinetics.system import PopulationSystem
from opencr.solvers.diagnostics import SolverReport

if TYPE_CHECKING:
    from opencr.kinetics.effective import QuasiStatic
    from opencr.model import Model

__all__ = ["Result"]


class Result:
    """A steady state.

    Attributes:
        model: The model solved.
        conditions: The conditions.
        rates: The rate coefficient of every reaction (SI), in network order.
        system: The population equations.
        evolving: The steady densities of the evolving states (m^-3).
        report: The solver's report.
    """

    def __init__(
        self,
        model: Model,
        conditions: Conditions,
        rates: np.ndarray,
        system: PopulationSystem,
        evolving: np.ndarray,
        report: SolverReport,
    ) -> None:
        """Store the solution."""
        self.model = model
        self.conditions = conditions
        self.rates = rates
        self.system = system
        self.evolving = np.asarray(evolving, dtype=float)
        self.report = report

    def __repr__(self) -> str:
        """The model, the mode and the conditions."""
        return (
            f"Result({self.model.name!r}, {self.report.mode}, "
            f"Te={self.conditions.electron_temperature:g} eV, ne={self.electron_density:.3g} m^-3)"
        )

    @property
    def mode(self) -> str:
        """The solve: ``"quasi-static"`` or ``"full"``."""
        return self.report.mode

    @cached_property
    def populations(self) -> dict[str, float]:
        """The density of every state in m^-3, by name, in network order."""
        densities = self.system.densities(self.evolving)
        return {name: densities[name] for name in self.model.state_names}

    def population(self, state: str) -> float:
        """Return the density of ``state`` in m^-3."""
        return self.populations[state]

    @property
    def electron_density(self) -> float:
        """``n_e`` in m^-3."""
        return self.system.electron_density(self.evolving)

    @cached_property
    def reduction(self) -> QuasiStatic:
        """The quasi-static reduction at these conditions (population and effective coefficients)."""
        from opencr.kinetics.effective import quasi_static

        conditions = self.conditions
        if conditions.quasi_neutral:
            conditions = conditions.replace(electron_density=self.electron_density)
        return quasi_static(self.model.network, self.rates, conditions)
