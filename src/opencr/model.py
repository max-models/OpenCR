"""A CR model: a reaction network with its drivers, and the solves on it.

:class:`Model` is what :func:`opencr.load_model` returns and what a custom
network is wrapped in. It evaluates the rate coefficients (once per set of
conditions), builds the population equations and solves them (see the guide
"The CR model"):

* :meth:`Model.solve_steady_state` with ``mode="quasi-static"`` (the default):
  the states with a density in the conditions are prescribed, the rest are in
  steady state;
* ``mode="full"``: a closed steady state with the element totals of the
  conditions;
* :meth:`Model.evolve`: the time-dependent populations;
* :meth:`Model.effective_rates`: the quasi-static reduction to effective rate
  and population coefficients.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from opencr.core.conditions import Conditions
from opencr.core.network import ReactionNetwork
from opencr.kinetics.effective import QuasiStatic, quasi_static
from opencr.kinetics.rates import RateEvaluator
from opencr.kinetics.system import PopulationSystem
from opencr.solvers.steady import solve_steady

__all__ = ["MODES", "Model"]

#: The steady-state solves.
MODES = ("quasi-static", "full")


class Model:
    """A reaction network with its default drivers.

    Args:
        network: The reaction network.
        drivers: The states a quasi-static solve needs densities for (for
            atomic hydrogen H(n=1) and H+); checked against the conditions.
        description: One sentence on what the model is.
    """

    def __init__(
        self,
        network: ReactionNetwork,
        drivers: Sequence[str] = (),
        description: str = "",
    ) -> None:
        """Prepare the rate evaluation."""
        for driver in drivers:
            network.state(driver)
        self.network = network
        self.drivers = tuple(drivers)
        self.description = description
        self.evaluator = RateEvaluator(network)

    @property
    def name(self) -> str:
        """The network's name."""
        return self.network.name

    @property
    def state_names(self) -> tuple[str, ...]:
        """The state names."""
        return self.network.state_names

    def manifest(self) -> dict[str, Any]:
        """Return the network's manifest with the drivers and the description."""
        return {
            **self.network.manifest(),
            "drivers": list(self.drivers),
            "description": self.description,
        }

    def rate_coefficients(self, conditions: Conditions) -> dict[str, float]:
        """Return every reaction's rate coefficient (SI) by name."""
        k = self.evaluator.coefficients(conditions)
        return dict(zip(self.evaluator.names, k.tolist(), strict=True))

    def system(
        self, conditions: Conditions, rates: np.ndarray | None = None
    ) -> PopulationSystem:
        """Return the population equations with the prescribed states of ``conditions``."""
        if rates is None:
            rates = self.evaluator.coefficients(conditions)
        evolving = [
            n for n in self.network.state_names if n not in conditions.densities
        ]
        return PopulationSystem(self.network, rates, conditions, evolving)

    def solve_steady_state(
        self,
        conditions: Conditions,
        mode: str = "quasi-static",
        initial: dict[str, float] | None = None,
        rtol: float = 1.0e-9,
    ):
        """Return the steady state as a :class:`~opencr.analysis.results.Result`.

        Args:
            conditions: The conditions. In ``"quasi-static"`` mode the drivers
                need densities; in ``"full"`` mode the conserved elements need totals.
            mode: One of :data:`MODES`.
            initial: Starting densities by state name for a nonlinear solve.
            rtol: The tolerance of the relative residual.
        """
        from opencr.analysis.results import Result

        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "quasi-static":
            missing = [d for d in self.drivers if d not in conditions.densities]
            if missing:
                raise ValueError(
                    f"a quasi-static solve of {self.name} needs the densities of the drivers "
                    f"{missing} in Conditions.densities (or use mode='full' with totals)"
                )
        rates = self.evaluator.coefficients(conditions)
        system = self.system(conditions, rates)
        start = None
        if initial is not None:
            start = np.array([initial.get(n, 0.0) for n in system.evolving])
        n, report = solve_steady(system, conditions.totals, start, mode, rtol)
        return Result(self, conditions, rates, system, n, report)

    def effective_rates(self, conditions: Conditions) -> QuasiStatic:
        """Return the quasi-static reduction with the prescribed states of ``conditions``."""
        rates = self.evaluator.coefficients(conditions)
        return quasi_static(self.network, rates, conditions)

    def evolve(self, conditions, times, initial, **options):
        """Return the time-dependent populations (see :func:`opencr.solvers.transient.evolve`)."""
        from opencr.solvers.transient import evolve

        return evolve(self, conditions, times, initial, **options)

    def __repr__(self) -> str:
        """The name and the size."""
        return (
            f"Model({self.name!r}, {len(self.network.states)} states, "
            f"{len(self.network.reactions)} reactions)"
        )
