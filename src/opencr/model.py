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
* :meth:`Model.effective_rates`: the quasi-static reduction to effective rate
  and population coefficients;
* :meth:`Model.solve_transient`: the populations against time from initial
  densities, with constant or time-dependent conditions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from opencr.core.conditions import Conditions
from opencr.core.network import ReactionNetwork
from opencr.core.rates import DetailedBalance
from opencr.kinetics.effective import QuasiStatic, quasi_static
from opencr.kinetics.rates import RateEvaluator
from opencr.kinetics.system import PopulationSystem
from opencr.solvers.steady import solve_steady
from opencr.solvers.transient import solve_transient

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

    def scaled_rates(
        self, conditions: Conditions, factors: Mapping[str, float] | None = None
    ) -> np.ndarray:
        """Return the rate coefficients with some multiplied by ``factors`` (by name).

        A detailed-balance inverse takes the factor of its forward reaction
        (unless it has its own), so the scaled network keeps detailed balance:
        this is what sensitivities and uncertainty propagation perturb.
        """
        rates = self.evaluator.coefficients(conditions)
        if not factors:
            return rates
        unknown = set(factors) - set(self.evaluator.names)
        if unknown:
            raise KeyError(f"unknown reactions in factors: {sorted(unknown)}")
        scale = np.ones(len(rates))
        for i, reaction in enumerate(self.network.reactions):
            name = reaction.name
            if name not in factors and isinstance(reaction.rate, DetailedBalance):
                name = reaction.rate.forward
            scale[i] = factors.get(name, 1.0)
        return rates * scale

    def _solve_once(self, conditions, mode, start, rtol, factors=None):
        from opencr.analysis.results import Result

        rates = self.scaled_rates(conditions, factors)
        system = self.system(conditions, rates)
        n, report = solve_steady(system, conditions.totals, start, mode, rtol)
        return Result(self, conditions, rates, system, n, report)

    def _solve_trapped(self, conditions, mode, start, rtol, factors=None):
        """Iterate the solve and the escape factors to self-consistency."""
        from opencr.kinetics.radiation import escape_factors
        from opencr.solvers.diagnostics import ConvergenceError

        trapping = conditions.trapping
        assert trapping is not None
        given = dict(conditions.escape)
        current: dict[str, float] = {}
        previous = float("inf")
        damping = 1.0
        for iteration in range(1, trapping.max_iterations + 1):
            point = conditions.replace(escape={**current, **given})
            result = self._solve_once(point, mode, start, rtol, factors)
            computed = escape_factors(
                self.network, result.populations, trapping, point.gas_temperature
            )
            change = max(
                (abs(g - current.get(name, 1.0)) for name, g in computed.items()),
                default=0.0,
            )
            result.report.messages.append(
                f"escape factors: iteration {iteration}, largest change {change:.1e}"
            )
            if change <= trapping.tolerance:
                return result
            if iteration == trapping.max_iterations:
                result.report.converged = False
                raise ConvergenceError(
                    "the escape factors did not converge; give a larger tolerance or "
                    "more iterations in RadiationTrapping",
                    result.report,
                )
            if change > previous:
                damping = 0.5  # the fixed point oscillates: relax it
            previous = change
            current = {
                name: current.get(name, 1.0) + damping * (g - current.get(name, 1.0))
                for name, g in computed.items()
            }
        raise AssertionError("unreachable")  # pragma: no cover

    def solve_steady_state(
        self,
        conditions: Conditions,
        mode: str = "quasi-static",
        initial: dict[str, float] | None = None,
        rtol: float = 1.0e-9,
        factors: Mapping[str, float] | None = None,
    ):
        """Return the steady state as a :class:`~opencr.analysis.results.Result`.

        Args:
            conditions: The conditions. In ``"quasi-static"`` mode the drivers
                need densities; in ``"full"`` mode the conserved elements need totals.
            mode: One of :data:`MODES`.
            initial: Starting densities by state name for a nonlinear solve.
            rtol: The tolerance of the relative residual.
            factors: Multiply the rate coefficients of these reactions (by
                name) by these factors (see :meth:`scaled_rates`).
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "quasi-static":
            missing = [d for d in self.drivers if d not in conditions.densities]
            if missing:
                raise ValueError(
                    f"a quasi-static solve of {self.name} needs the densities of the drivers "
                    f"{missing} in Conditions.densities (or use mode='full' with totals)"
                )
        start = None
        if initial is not None:
            start = np.array(
                [initial.get(n, 0.0) for n in self.network.state_names
                 if n not in conditions.densities]
            )  # fmt: skip
        if conditions.trapping is None:
            return self._solve_once(conditions, mode, start, rtol, factors)
        return self._solve_trapped(conditions, mode, start, rtol, factors)

    def solve_transient(
        self,
        conditions: Conditions | Callable[[float], Conditions],
        initial: dict[str, float],
        times: np.ndarray,
        method: str = "BDF",
        rtol: float = 1.0e-6,
        atol: float | None = None,
    ):
        """Return the populations against time as a :class:`~opencr.analysis.transient.TransientResult`.

        The states with a density in the conditions stay prescribed (constant,
        or following ``conditions(t)``); the others evolve from ``initial``.

        Args:
            conditions: The conditions, or a function of the time in s
                returning them (same prescribed states at every time).
            initial: The densities at ``times[0]`` by state name; an evolving
                state not listed starts empty.
            times: The output times in s, increasing.
            method: ``"BDF"`` (default), ``"Radau"`` or ``"LSODA"``.
            rtol: The relative tolerance.
            atol: The absolute tolerance in m^-3 (default ``1e-12`` of the
                largest initial density).
        """
        from opencr.analysis.transient import TransientResult

        times = np.asarray(times, dtype=float)
        conditions_at: Callable[[float], Conditions]
        system_at: Callable[[float], PopulationSystem]
        varying: Callable[[float], PopulationSystem] | None = None
        if callable(conditions):
            conditions_at = conditions
            first = conditions_at(float(times[0]))
            cache: dict[float, PopulationSystem] = {}

            def cached(t: float) -> PopulationSystem:
                if t not in cache:
                    cache.clear()
                    point = conditions_at(t)
                    if set(point.densities) != set(first.densities):
                        raise ValueError(
                            "the prescribed states must be the same at every time"
                        )
                    cache[t] = self.system(point)
                return cache[t]

            system_at = varying = cached
        else:
            fixed, fixed_system = conditions, self.system(conditions)
            conditions_at = lambda _: fixed  # noqa: E731
            system_at = lambda _: fixed_system  # noqa: E731
        system = system_at(float(times[0]))
        unknown = set(initial) - set(self.network.state_names)
        if unknown:
            raise KeyError(f"unknown states in initial: {sorted(unknown)}")
        start = np.array([initial.get(n, 0.0) for n in system.evolving])
        densities, report = solve_transient(
            system,
            start,
            times,
            method,
            rtol,
            atol,
            varying,
        )
        return TransientResult(self, conditions_at, system_at, times, densities, report)

    def effective_rates(self, conditions: Conditions) -> QuasiStatic:
        """Return the quasi-static reduction with the prescribed states of ``conditions``."""
        rates = self.evaluator.coefficients(conditions)
        return quasi_static(self.network, rates, conditions)

    def __repr__(self) -> str:
        """The name and the size."""
        return (
            f"Model({self.name!r}, {len(self.network.states)} states, "
            f"{len(self.network.reactions)} reactions)"
        )
