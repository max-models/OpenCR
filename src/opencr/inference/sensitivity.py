"""Sensitivities of CR results to the rate data, and the propagation of their uncertainties.

:func:`sensitivities` gives the logarithmic sensitivities

    S_qr = d ln q / d ln k_r

of named quantities ``q`` (:mod:`opencr.analysis.quantities`) to the rate
coefficients ``k_r``, by central finite differences in ``ln k_r`` (a
detailed-balance inverse moves with its forward reaction, see
:meth:`opencr.model.Model.scaled_rates`). :func:`propagate` turns the relative
uncertainties of the data (the ``uncertainty`` of each reaction's
provenance, or given ones) into the uncertainty of the quantities, linearly
(``var ln q = sum_r S_qr^2 sigma_r^2`` for independent data) or by Monte
Carlo over log-normal factors. This is the uncertainty of the data; the
error of a surrogate (:mod:`opencr.ml`) is a separate thing and is reported
separately.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from opencr.analysis.quantities import check_quantity, quantities
from opencr.core.conditions import Conditions
from opencr.core.rates import DetailedBalance
from opencr.model import Model

__all__ = ["Propagation", "data_uncertainties", "propagate", "sensitivities"]


def _independent(model: Model) -> list[str]:
    """The reactions whose rates are data (not detailed-balance inverses)."""
    return [
        r.name
        for r in model.network.reactions
        if not isinstance(r.rate, DetailedBalance)
    ]


def sensitivities(
    model: Model,
    conditions: Conditions,
    outputs: Sequence[str],
    reactions: Sequence[str] | None = None,
    mode: str = "quasi-static",
    step: float = 0.01,
) -> xr.DataArray:
    """Return ``d ln q / d ln k_r`` of the quantities ``outputs``.

    Args:
        model: The model.
        conditions: The conditions of the steady state.
        outputs: Names of quantities (:mod:`opencr.analysis.quantities`).
        reactions: The reactions to perturb; by default every reaction that is
            not a detailed-balance inverse.
        mode: The steady-state mode.
        step: The step in ``ln k``.

    Returns:
        A DataArray over (``output``, ``reaction``).
    """
    for name in outputs:
        check_quantity(model, name)
    reactions = list(reactions) if reactions is not None else _independent(model)
    values = np.zeros((len(outputs), len(reactions)))
    up, down = np.exp(step), np.exp(-step)
    for j, reaction in enumerate(reactions):
        plus = quantities(
            model.solve_steady_state(conditions, mode, factors={reaction: up}), outputs
        )
        minus = quantities(
            model.solve_steady_state(conditions, mode, factors={reaction: down}),
            outputs,
        )
        values[:, j] = (np.log(plus) - np.log(minus)) / (2.0 * step)
    return xr.DataArray(
        values,
        dims=("output", "reaction"),
        coords={"output": list(outputs), "reaction": reactions},
        attrs={"description": "d ln q / d ln k", "step": step, "mode": mode},
    )


def data_uncertainties(model: Model, default: float | None = None) -> dict[str, float]:
    """Return the relative uncertainty of each reaction's data from its provenance.

    Reactions without a stated uncertainty get ``default`` (or are left out
    when it is ``None``); detailed-balance inverses are never listed.
    """
    out = {}
    for reaction in model.network.reactions:
        if isinstance(reaction.rate, DetailedBalance):
            continue
        value = None if reaction.provenance is None else reaction.provenance.uncertainty
        value = default if value is None else value
        if value is not None and value > 0.0:
            out[reaction.name] = float(value)
    return out


@dataclass(frozen=True)
class Propagation:
    """The propagated uncertainty of some quantities.

    Attributes:
        outputs: The quantity names.
        nominal: The nominal values.
        relative: The relative (log) standard deviations, ``sigma(ln q)``.
        contributions: For the linear method, ``(S_qr sigma_r)^2`` over
            (``output``, ``reaction``); ``None`` for Monte Carlo.
        samples: For Monte Carlo, the sampled values (sample, output).
        method: ``"linear"`` or ``"monte-carlo"``.
    """

    outputs: tuple[str, ...]
    nominal: np.ndarray
    relative: np.ndarray
    contributions: xr.DataArray | None
    samples: np.ndarray | None
    method: str

    def interval(self, coverage: float = 0.68) -> tuple[np.ndarray, np.ndarray]:
        """Return the (lower, upper) bounds of a log-normal interval of ``coverage``."""
        from scipy.stats import norm

        z = norm.ppf(0.5 + coverage / 2.0)
        return self.nominal * np.exp(-z * self.relative), self.nominal * np.exp(
            z * self.relative
        )

    def dominant(self, output: str, number: int = 5) -> list[tuple[str, float]]:
        """Return the reactions contributing most to the variance of ``output`` (linear only)."""
        if self.contributions is None:
            raise ValueError("contributions are only available for the linear method")
        row = self.contributions.sel(output=output)
        total = float(row.sum())
        order = np.argsort(-row.values)[:number]
        return [
            (
                str(row.reaction.values[i]),
                float(row.values[i]) / total if total else 0.0,
            )
            for i in order
        ]

    def to_dataset(self) -> xr.Dataset:
        """Return the nominal values and the relative uncertainties as a dataset."""
        return xr.Dataset(
            {
                "nominal": ("output", self.nominal),
                "relative_uncertainty": ("output", self.relative),
            },
            coords={"output": list(self.outputs)},
            attrs={"method": self.method},
        )


def propagate(
    model: Model,
    conditions: Conditions,
    outputs: Sequence[str],
    uncertainties: Mapping[str, float] | None = None,
    method: str = "linear",
    samples: int = 200,
    seed: int = 0,
    mode: str = "quasi-static",
    **options: Any,
) -> Propagation:
    """Propagate the uncertainties of the rate data to the quantities ``outputs``.

    Args:
        model: The model.
        conditions: The conditions of the steady state.
        outputs: Names of quantities.
        uncertainties: Relative (1 sigma, log-normal) uncertainty by reaction
            name; by default :func:`data_uncertainties` of the model.
        method: ``"linear"`` (first order, through :func:`sensitivities`) or
            ``"monte-carlo"`` (independent log-normal factors).
        samples: The number of Monte Carlo samples.
        seed: The seed of the Monte Carlo samples.
        mode: The steady-state mode.
        **options: Passed on to :func:`sensitivities` (``step``).
    """
    if uncertainties is None:
        uncertainties = data_uncertainties(model)
    names = list(uncertainties)
    sigma = np.array([uncertainties[n] for n in names])
    nominal = quantities(model.solve_steady_state(conditions, mode), outputs)
    if method == "linear":
        s = sensitivities(model, conditions, outputs, names, mode, **options)
        contributions = (s * sigma) ** 2
        relative = np.sqrt(contributions.sum("reaction").values)
        return Propagation(
            tuple(outputs), nominal, relative, contributions, None, method
        )
    if method == "monte-carlo":
        rng = np.random.default_rng(seed)
        drawn = np.zeros((samples, len(outputs)))
        for i in range(samples):
            factors = dict(
                zip(names, np.exp(sigma * rng.standard_normal(len(names))), strict=True)
            )
            drawn[i] = quantities(
                model.solve_steady_state(conditions, mode, factors=factors), outputs
            )
        relative = np.std(np.log(drawn), axis=0, ddof=1)
        return Propagation(tuple(outputs), nominal, relative, None, drawn, method)
    raise ValueError("method must be 'linear' or 'monte-carlo'")
