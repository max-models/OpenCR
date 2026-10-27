"""The result of a steady-state solve, and what can be read from it.

A :class:`Result` keeps the model, the conditions, the rate coefficients, the
population equations and the densities of every state, with the solver's
report. From it: the reaction fluxes and the dominant gains and losses of a
state, the lines and their photon emissivity coefficients, a synthetic
spectrum, and everything as an xarray dataset (:meth:`Result.to_dataset`) or a
NetCDF file with the model's manifest (:meth:`Result.export`).
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from opencr.analysis.emission import line_table, spectrum
from opencr.core.conditions import Conditions
from opencr.kinetics.system import PopulationSystem
from opencr.solvers.diagnostics import SolverReport

if TYPE_CHECKING:
    from opencr.kinetics.effective import QuasiStatic
    from opencr.model import Model

__all__ = ["Result"]


def _opencr_version() -> str:
    import opencr

    return opencr.__version__


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

    @cached_property
    def fluxes(self) -> dict[str, float]:
        """The rate of every reaction in m^-3 s^-1, by name."""
        w = self.system.reaction_rates(self.evolving)
        return dict(zip(self.model.evaluator.names, w.tolist(), strict=True))

    def dominant_processes(self, state: str, number: int = 5) -> list[dict[str, Any]]:
        """Return the largest gains and losses of ``state``.

        Each entry has the reaction ``name``, its ``equation``, the ``rate`` of
        change of the state (m^-3 s^-1; negative for a loss) and its ``fraction``
        of the state's total gain or total loss. Sorted by ``|rate|``.
        """
        entries = []
        for reaction in self.model.network.reactions_of(state):
            rate = reaction.net[state] * self.fluxes[reaction.name]
            entries.append(
                {"name": reaction.name, "equation": reaction.equation, "rate": rate}
            )
        gain = sum(e["rate"] for e in entries if e["rate"] > 0)
        loss = -sum(e["rate"] for e in entries if e["rate"] < 0)
        for entry in entries:
            total = gain if entry["rate"] > 0 else loss
            entry["fraction"] = abs(entry["rate"]) / total if total > 0 else 0.0
        entries.sort(key=lambda e: -abs(e["rate"]))
        return entries[:number]

    def emission(self, air: bool = False) -> xr.Dataset:
        """Return the lines: wavelength, emissivity and power (:func:`~opencr.analysis.emission.line_table`)."""
        return line_table(self.model.network, self.rates, self.populations, air)

    def photon_emissivity_coefficients(self) -> xr.DataArray:
        """Return ``PEC_d`` of every line per driver in m^3/s (quasi-static only).

        ``emissivity = n_e sum_d PEC_d n_d``; for atomic hydrogen the ``H(n=1)``
        part is the excitation component and the ``H+`` part the recombination one.
        """
        reduction = self.reduction
        lines = self.emission()
        values = np.zeros((lines.sizes["line"], len(reduction.drivers)))
        names = self.model.evaluator.names
        for i, (line, upper) in enumerate(
            zip(lines["line"].values, lines["upper"].values, strict=True)
        ):
            if upper in reduction.drivers:
                continue
            row = reduction.coefficients[reduction.excited.index(upper)]
            values[i] = self.rates[names.index(line)] * row / reduction.electron_density
        return xr.DataArray(
            values,
            coords={
                "line": lines["line"].values,
                "driver": list(reduction.drivers),
                "label": ("line", lines["label"].values),
            },
            dims=("line", "driver"),
            name="pec",
            attrs={"units": "m^3 s^-1"},
        )

    def emission_spectrum(
        self,
        wavelength: np.ndarray | None = None,
        fwhm: float = 0.0,
        air: bool = False,
        quantity: str = "power",
    ) -> xr.DataArray:
        """Return a synthetic spectrum: the lines Doppler broadened at the gas temperature.

        Args:
            wavelength: The grid in nm; by default 2000 points around the visible
                lines (350-700 nm).
            fwhm: An instrumental FWHM in nm, added in quadrature.
            air: Wavelengths in air.
            quantity: ``"power"`` or ``"emissivity"``.
        """
        if wavelength is None:
            wavelength = np.linspace(350.0, 700.0, 2000)
        lines = self.emission(air)
        mass = self.model.network.species[0].mass
        return spectrum(
            lines, wavelength, self.conditions.gas_temperature, mass, fwhm, quantity
        )

    def to_dataset(self) -> xr.Dataset:
        """Return the populations, rates, fluxes and lines as one xarray dataset.

        The conditions, the solver report and the model's manifest are in the
        attributes, as JSON.
        """
        names = list(self.model.state_names)
        reactions = list(self.model.evaluator.names)
        lines = self.emission()
        data = xr.Dataset(
            {
                "population": (
                    "state",
                    [self.populations[n] for n in names],
                    {"units": "m^-3"},
                ),
                "rate_coefficient": (
                    "reaction",
                    self.rates,
                    {"units": "SI (m^3(k-1)/s)"},
                ),
                "flux": (
                    "reaction",
                    [self.fluxes[r] for r in reactions],
                    {"units": "m^-3 s^-1"},
                ),
                "line_wavelength": (
                    "line",
                    lines["wavelength"].values,
                    {"units": "nm"},
                ),
                "line_emissivity": (
                    "line",
                    lines["emissivity"].values,
                    {"units": "photons m^-3 s^-1"},
                ),
            },
            coords={
                "state": names,
                "reaction": reactions,
                "line": lines["line"].values.astype(str),
            },
            attrs={
                "model": self.model.name,
                "solve": self.mode,
                "electron_density": self.electron_density,
                "conditions": json.dumps(self.conditions.to_dict()),
                "report": self.report.summary(),
                "manifest": json.dumps(self.model.manifest()),
                "opencr_version": _opencr_version(),
            },
        )
        return data

    def export(self, path: str | Path) -> Path:
        """Write :meth:`to_dataset` to a NetCDF file and return its path."""
        path = Path(path)
        self.to_dataset().to_netcdf(path, engine="scipy")
        return path

    def __repr__(self) -> str:
        """The model, the mode and the conditions."""
        return (
            f"Result({self.model.name!r}, {self.report.mode}, "
            f"Te={self.conditions.electron_temperature:g} eV, ne={self.electron_density:.3g} m^-3)"
        )
