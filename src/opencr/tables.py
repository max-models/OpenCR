"""Effective-rate tables for the codes that use OpenCR's results.

:func:`effective_rate_table` solves a model on a grid of electron
temperature and density and stores named quantities
(:mod:`opencr.analysis.quantities`: effective rates, population and electron
cooling coefficients, line emissivities) as an xarray dataset with the
model's provenance; :class:`EffectiveRateTable` interpolates such a table,
bilinearly in ``ln q`` against ``ln T_e`` and ``ln n_e`` (end values outside),
and writes and reads it as netCDF. wellbalanced, beamtracer and freeofcharge
take their effective rates from these tables in place of fits.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr

from opencr.analysis.quantities import check_quantity, quantities
from opencr.analysis.results import _opencr_version
from opencr.core.conditions import Conditions
from opencr.model import Model

__all__ = ["EffectiveRateTable", "effective_rate_table"]


class EffectiveRateTable:
    """A table of quantities against ``T_e`` and ``n_e``, interpolated in log space.

    Args:
        dataset: A dataset as :func:`effective_rate_table` makes it.
    """

    def __init__(self, dataset: xr.Dataset) -> None:
        """Check the dataset and prepare the interpolation."""
        missing = {"value", "electron_temperature", "electron_density", "quantity"} - (
            set(dataset.variables)
        )
        if missing:
            raise ValueError(f"not an effective-rate table: missing {sorted(missing)}")
        self.dataset = dataset
        self._log_t = np.log(dataset["electron_temperature"].values)
        self._log_n = np.log(dataset["electron_density"].values)

    @property
    def quantities(self) -> tuple[str, ...]:
        """The tabulated quantities."""
        return tuple(self.dataset["quantity"].values.tolist())

    def function(self, name: str, electron_density: float):
        """Return ``T_e -> quantity`` at a fixed ``n_e``, for codes whose rates depend on ``T_e`` only."""
        if name not in self.quantities:
            raise KeyError(f"no quantity {name!r}; the table has {self.quantities}")

        def rate(electron_temperature):
            return self(name, electron_temperature, electron_density)

        return rate

    def to_netcdf(self, path: str | Path) -> Path:
        """Write the table to ``path`` (netCDF) and return the path."""
        path = Path(path)
        self.dataset.to_netcdf(path, engine="scipy")
        return path

    @classmethod
    def from_netcdf(cls, path: str | Path) -> EffectiveRateTable:
        """Read a table written by :meth:`to_netcdf`."""
        with xr.open_dataset(path, engine="scipy") as dataset:
            return cls(dataset.load())

    def __call__(
        self,
        name: str,
        electron_temperature: np.ndarray | float,
        electron_density: np.ndarray | float,
    ) -> np.ndarray:
        """Return the quantity ``name`` at ``T_e`` (eV) and ``n_e`` (m^-3), broadcast.

        Positive quantities are interpolated in ``ln q``; a quantity with
        negative or zero values (an effective loss rate, a cooling with
        recombination) linearly.
        """
        if name not in self.quantities:
            raise KeyError(f"no quantity {name!r}; the table has {self.quantities}")
        table = self.dataset["value"].sel(quantity=name).values
        positive = bool(np.all(table > 0.0))
        grid = np.log(table) if positive else table
        log_t, log_n = np.broadcast_arrays(
            np.log(np.asarray(electron_temperature, dtype=float)),
            np.log(np.asarray(electron_density, dtype=float)),
        )
        x = np.clip(log_t, self._log_t[0], self._log_t[-1])
        y = np.clip(log_n, self._log_n[0], self._log_n[-1])
        i = np.clip(
            np.searchsorted(self._log_t, x) - 1, 0, max(len(self._log_t) - 2, 0)
        )
        j = np.clip(
            np.searchsorted(self._log_n, y) - 1, 0, max(len(self._log_n) - 2, 0)
        )
        if len(self._log_t) > 1:
            u = (x - self._log_t[i]) / (self._log_t[i + 1] - self._log_t[i])
        else:
            u = np.zeros_like(x)
        if len(self._log_n) > 1:
            w = (y - self._log_n[j]) / (self._log_n[j + 1] - self._log_n[j])
        else:
            w = np.zeros_like(y)
        i1 = np.minimum(i + 1, len(self._log_t) - 1)
        j1 = np.minimum(j + 1, len(self._log_n) - 1)
        value = (
            (1 - u) * (1 - w) * grid[i, j]
            + u * (1 - w) * grid[i1, j]
            + (1 - u) * w * grid[i, j1]
            + u * w * grid[i1, j1]
        )
        return np.exp(value) if positive else value


def effective_rate_table(
    model: Model,
    conditions: Conditions,
    outputs: Sequence[str],
    temperatures: Sequence[float] = tuple(np.geomspace(0.5, 100.0, 25)),
    densities: Sequence[float] = tuple(np.geomspace(1e16, 1e21, 11)),
    mode: str = "quasi-static",
) -> EffectiveRateTable:
    """Solve ``model`` on a grid of ``T_e`` (eV) and ``n_e`` (m^-3) and tabulate ``outputs``.

    Args:
        model: The model.
        conditions: The conditions with the driver densities (the quasi-static
            quantities per driver do not depend on them for a linear network).
        outputs: The quantities.
        temperatures: The electron temperatures in eV, increasing.
        densities: The electron densities in m^-3, increasing.
        mode: The steady-state mode.
    """
    for name in outputs:
        check_quantity(model, name)
    t = np.asarray(temperatures, dtype=float)
    n = np.asarray(densities, dtype=float)
    values = np.zeros((len(outputs), t.size, n.size))
    for i, temperature in enumerate(t):
        for j, density in enumerate(n):
            point = conditions.replace(
                electron_temperature=float(temperature), electron_density=float(density)
            )
            values[:, i, j] = quantities(model.solve_steady_state(point, mode), outputs)
    dataset = xr.Dataset(
        {"value": (("quantity", "electron_temperature", "electron_density"), values)},
        coords={
            "quantity": list(outputs),
            "electron_temperature": ("electron_temperature", t, {"units": "eV"}),
            "electron_density": ("electron_density", n, {"units": "m^-3"}),
        },
        attrs={
            "model": model.name,
            "opencr_version": _opencr_version(),
            "conditions": json.dumps(conditions.to_dict(), default=str),
            "solve": mode,
            "assumptions": json.dumps(list(model.network.assumptions)),
        },
    )
    return EffectiveRateTable(dataset)
