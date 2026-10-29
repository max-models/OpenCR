"""The result of a time-dependent solve.

A :class:`TransientResult` keeps the densities of every state at the output
times, the electron density, the conditions and the integrator's report. Any
output time can be turned into a steady-state-like :class:`~opencr.analysis.results.Result`
(:meth:`TransientResult.at`) for fluxes, dominant processes and emission, and
the whole history into an xarray dataset (:meth:`TransientResult.to_dataset`).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from opencr.analysis.results import Result
from opencr.core.conditions import Conditions
from opencr.kinetics.system import PopulationSystem
from opencr.solvers.diagnostics import SolverReport, relative_residual
from opencr.solvers.transient import TransientReport

if TYPE_CHECKING:
    from opencr.model import Model

__all__ = ["TransientResult"]


class TransientResult:
    """The populations against time.

    Attributes:
        model: The model solved.
        times: The output times in s.
        evolving: The densities of the evolving states, shape ``(times, states)``.
        report: The integrator's report.
    """

    def __init__(
        self,
        model: Model,
        conditions_at: Callable[[float], Conditions],
        system_at: Callable[[float], PopulationSystem],
        times: np.ndarray,
        evolving: np.ndarray,
        report: TransientReport,
    ) -> None:
        """Store the history."""
        self.model = model
        self._conditions_at = conditions_at
        self._system_at = system_at
        self.times = np.asarray(times, dtype=float)
        self.evolving = np.asarray(evolving, dtype=float)
        self.report = report

    @property
    def conditions(self) -> Conditions:
        """The conditions at the first time."""
        return self._conditions_at(float(self.times[0]))

    def conditions_at(self, time: float) -> Conditions:
        """Return the conditions at ``time``."""
        return self._conditions_at(float(time))

    @cached_property
    def populations(self) -> dict[str, np.ndarray]:
        """The density of every state against time (m^-3), by name, in network order."""
        names = self.model.state_names
        out: dict[str, np.ndarray] = {}
        for name in names:
            out[name] = np.empty(self.times.size)
        for k, (t, n) in enumerate(zip(self.times, self.evolving, strict=True)):
            densities = self._system_at(float(t)).densities(n)
            for name in names:
                out[name][k] = densities[name]
        return out

    def population(self, state: str) -> np.ndarray:
        """Return the density of ``state`` against time in m^-3."""
        return self.populations[state]

    @cached_property
    def electron_density(self) -> np.ndarray:
        """``n_e`` against time in m^-3."""
        return np.array(
            [
                self._system_at(float(t)).electron_density(n)
                for t, n in zip(self.times, self.evolving, strict=True)
            ]
        )

    def at(self, index: int = -1) -> Result:
        """Return the state at output time ``index`` as a :class:`Result`.

        Its report says how far the populations are from a steady state (the
        relative residual at that moment).
        """
        time = float(self.times[index])
        system = self._system_at(time)
        n = self.evolving[index]
        report = SolverReport(
            "transient",
            f"{self.report.method} at t = {time:.3e} s",
            converged=self.report.success,
            iterations=self.report.evaluations,
            residual=relative_residual(system, n),
            minimum=float(n.min() / n.max()) if n.size and n.max() > 0 else 0.0,
        )
        return Result(
            self.model, self._conditions_at(time), system.rates, system, n, report
        )

    def to_dataset(self) -> xr.Dataset:
        """Return the populations and ``n_e`` against time as an xarray dataset."""
        names = list(self.model.state_names)
        data = xr.Dataset(
            {
                "population": (
                    ("time", "state"),
                    np.column_stack([self.populations[n] for n in names]),
                    {"units": "m^-3"},
                ),
                "electron_density": ("time", self.electron_density, {"units": "m^-3"}),
            },
            coords={"time": ("time", self.times, {"units": "s"}), "state": names},
            attrs={
                "model": self.model.name,
                "solve": "transient",
                "conditions": json.dumps(self.conditions.to_dict()),
                "report": self.report.summary(),
                "manifest": json.dumps(self.model.manifest()),
            },
        )
        return data

    def __repr__(self) -> str:
        """The model and the time span."""
        return (
            f"TransientResult({self.model.name!r}, {self.times.size} times from "
            f"{self.times[0]:.3g} to {self.times[-1]:.3g} s)"
        )
