"""Parameter scans over the conditions.

:func:`scan` solves a model on the outer product of values of condition
fields (``electron_temperature``, ``electron_density``, a driver density as
``density:<state>`` or a total as ``total:<element>``) and returns an xarray
dataset with the populations and, for quasi-static solves, the effective rate
coefficients between the drivers and the photon emissivity coefficients.
Each point is an independent reference solve; the batched backends
(:mod:`opencr.backends`) do the same for Maxwellian scans in one pass.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from opencr.core.conditions import Conditions
from opencr.model import Model

__all__ = ["scan"]


def _apply(conditions: Conditions, field: str, value: float) -> Conditions:
    if field.startswith("density:"):
        return conditions.replace(densities={**conditions.densities, field[8:]: value})
    if field.startswith("total:"):
        return conditions.replace(totals={**conditions.totals, field[6:]: value})
    return conditions.replace(**{field: value})


def scan(
    model: Model,
    conditions: Conditions,
    mode: str = "quasi-static",
    effective: bool = True,
    lines: bool = True,
    **axes: Sequence[float],
) -> xr.Dataset:
    """Solve ``model`` on a grid of conditions.

    Args:
        model: The model.
        conditions: The conditions every point starts from.
        mode: The steady-state mode.
        effective: Add ``effective_rate`` (target driver, source driver) and, with
            ``lines``, ``pec`` (line, driver) for quasi-static solves.
        lines: Add the line emissivities.
        **axes: The condition fields to scan and their values, e.g.
            ``electron_temperature=np.geomspace(0.5, 50, 20)``. Use
            ``{"density:H(n=1)": [...]}`` or ``{"total:H": [...]}`` unpacked with ``**``.

    Returns:
        A dataset over the axes (and ``state``, ``line``, ``driver``).
    """
    if not axes:
        raise ValueError("give at least one axis to scan")
    names = list(axes)
    values = [np.asarray(v, dtype=float) for v in axes.values()]
    shape = tuple(v.size for v in values)
    states = list(model.state_names)
    populations = np.zeros((*shape, len(states)))
    data: dict[str, Any] = {}
    first = None
    for index in itertools.product(*(range(n) for n in shape)):
        point = conditions
        for field, axis, i in zip(names, values, index, strict=True):
            point = _apply(point, field, float(axis[i]))
        result = model.solve_steady_state(point, mode=mode)
        populations[index] = [result.populations[s] for s in states]
        if lines:
            emission = result.emission()
            data.setdefault("emissivity", np.zeros((*shape, emission.sizes["line"])))
            data["emissivity"][index] = emission["emissivity"].values
            first = first or emission
        if effective and mode == "quasi-static":
            reduction = result.reduction
            data.setdefault(
                "effective_rate", np.zeros((*shape, *reduction.effective.shape))
            )
            data["effective_rate"][index] = (
                reduction.effective / reduction.electron_density
            )
            data.setdefault("drivers", list(reduction.drivers))
            if lines:
                pec = result.photon_emissivity_coefficients()
                data.setdefault("pec", np.zeros((*shape, *pec.shape)))
                data["pec"][index] = pec.values
    coords: dict[str, Any] = {n: v for n, v in zip(names, values, strict=True)}
    coords["state"] = states
    variables: dict[str, Any] = {
        "population": ((*names, "state"), populations, {"units": "m^-3"})
    }
    if lines and first is not None:
        coords["line"] = first["line"].values.astype(str)
        coords["wavelength"] = ("line", first["wavelength"].values)
        coords["label"] = ("line", first["label"].values.astype(str))
        variables["emissivity"] = (
            (*names, "line"),
            data["emissivity"],
            {"units": "photons m^-3 s^-1"},
        )
    if "effective_rate" in data:
        coords["target"] = data["drivers"]
        coords["source"] = data["drivers"]
        variables["effective_rate"] = (
            (*names, "target", "source"),
            data["effective_rate"],
            {"units": "m^3 s^-1", "description": "M_eff[target, source] / n_e"},
        )
        if "pec" in data:
            coords["driver"] = data["drivers"]
            variables["pec"] = (
                (*names, "line", "driver"),
                data["pec"],
                {"units": "m^3 s^-1"},
            )
    return xr.Dataset(
        variables, coords=coords, attrs={"model": model.name, "solve": mode}
    )
