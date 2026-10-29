"""Inverse modelling: plasma parameters from measured quantities such as line ratios.

Given measured quantities ``y_i`` (:mod:`opencr.analysis.quantities`, e.g.
Balmer line ratios) with relative uncertainties ``s_i``, and parameters
``theta`` of the conditions (``electron_temperature``, ``electron_density``,
``density:<state>``) with log-uniform priors on ``[low, high]``, the
posterior is

    p(theta | y) ~ exp(-chi^2 / 2),   chi^2 = sum_i (ln m_i(theta) - ln y_i)^2 / s_i^2,

with log-normal errors (``s_i`` is the standard deviation of ``ln y_i``,
about the relative uncertainty). :func:`posterior_grid` evaluates it on a grid
of the parameters (exact for one to three parameters) and :func:`fit` finds
the maximum by least squares in the log parameters. The forward model is the
CR model, or any callable with the same signature, e.g. a trained surrogate
(:meth:`opencr.ml.Surrogate.forward`), for speed.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.optimize import least_squares

from opencr.analysis.quantities import check_quantity, quantities
from opencr.analysis.scan import _apply
from opencr.core.conditions import Conditions
from opencr.model import Model

__all__ = ["Fit", "Observation", "fit", "forward_model", "posterior_grid"]

#: A forward model: parameter values (in the order of the parameters) -> quantities.
Forward = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Observation:
    """A measured quantity.

    Attributes:
        quantity: The name of the quantity.
        value: The measured value.
        uncertainty: The standard deviation of its logarithm (about the relative
            uncertainty).
    """

    quantity: str
    value: float
    uncertainty: float = 0.1

    def __post_init__(self) -> None:
        """Check the value and the uncertainty."""
        if not self.value > 0.0:
            raise ValueError(f"{self.quantity}: the measured value must be > 0")
        if not self.uncertainty > 0.0:
            raise ValueError(f"{self.quantity}: the uncertainty must be > 0")


def forward_model(
    model: Model,
    conditions: Conditions,
    parameters: Sequence[str],
    outputs: Sequence[str],
    mode: str = "quasi-static",
) -> Forward:
    """Return the forward model ``theta -> quantities`` of a CR model.

    Args:
        model: The model.
        conditions: The conditions the parameters are set in.
        parameters: The condition fields (as for :func:`opencr.scan`).
        outputs: The quantities.
        mode: The steady-state mode.
    """
    for name in outputs:
        check_quantity(model, name)

    def forward(theta: np.ndarray) -> np.ndarray:
        point = conditions
        for field, value in zip(parameters, theta, strict=True):
            point = _apply(point, field, float(value))
        return quantities(model.solve_steady_state(point, mode), outputs)

    return forward


def _chi2(forward: Forward, observations: Sequence[Observation]):
    log_y = np.log([o.value for o in observations])
    sigma = np.array([o.uncertainty for o in observations])

    def residual(theta: np.ndarray) -> np.ndarray:
        m = np.asarray(forward(theta), dtype=float)
        with np.errstate(divide="ignore"):
            return (np.log(np.maximum(m, 1e-300)) - log_y) / sigma

    return residual


def _check_bounds(bounds: Mapping[str, tuple[float, float]]) -> None:
    if not bounds:
        raise ValueError("give at least one parameter")
    for name, (low, high) in bounds.items():
        if not 0.0 < low < high:
            raise ValueError(f"{name}: need 0 < low < high (log-uniform prior)")


def posterior_grid(
    forward: Forward,
    observations: Sequence[Observation],
    bounds: Mapping[str, tuple[float, float]],
    points: int = 25,
) -> xr.Dataset:
    """Return the posterior on a log-spaced grid of the parameters.

    Args:
        forward: The forward model, taking the parameters in the order of ``bounds``.
        observations: The measurements.
        bounds: The prior range of each parameter.
        points: The grid points per parameter.

    Returns:
        A dataset with ``posterior`` (normalised over the grid, per log cell),
        ``chi2``, and the attributes ``best`` (the grid maximum) and the
        posterior ``mean`` and ``std`` of each parameter's logarithm.
    """
    _check_bounds(bounds)
    names = list(bounds)
    axes = [np.geomspace(low, high, points) for low, high in bounds.values()]
    residual = _chi2(forward, observations)
    chi2 = np.full(tuple(points for _ in names), np.inf)
    for index in itertools.product(range(points), repeat=len(names)):
        theta = np.array([axis[i] for axis, i in zip(axes, index, strict=True)])
        r = residual(theta)
        if np.all(np.isfinite(r)):
            chi2[index] = float(r @ r)
    log_p = -0.5 * (chi2 - chi2.min())
    posterior = np.exp(log_p)
    posterior /= posterior.sum()
    best = np.unravel_index(int(np.argmax(posterior)), posterior.shape)
    attrs: dict[str, float] = {}
    for k, (name, axis) in enumerate(zip(names, axes, strict=True)):
        marginal = posterior.sum(axis=tuple(j for j in range(len(names)) if j != k))
        log_axis = np.log(axis)
        mean = float(marginal @ log_axis)
        attrs[f"best:{name}"] = float(axis[best[k]])
        attrs[f"mean:{name}"] = float(np.exp(mean))
        attrs[f"log_std:{name}"] = float(np.sqrt(marginal @ (log_axis - mean) ** 2))
    return xr.Dataset(
        {"posterior": (names, posterior), "chi2": (names, chi2)},
        coords={name: axis for name, axis in zip(names, axes, strict=True)},
        attrs=attrs,
    )


@dataclass(frozen=True)
class Fit:
    """The maximum of the posterior.

    Attributes:
        parameters: The best values by name.
        log_std: The standard deviation of each parameter's logarithm from the
            curvature (the Gauss-Newton covariance), by name.
        chi2: The chi^2 at the best values.
        success: Whether the least-squares solver converged.
    """

    parameters: dict[str, float]
    log_std: dict[str, float]
    chi2: float
    success: bool


def fit(
    forward: Forward,
    observations: Sequence[Observation],
    bounds: Mapping[str, tuple[float, float]],
    start: Mapping[str, float] | None = None,
) -> Fit:
    """Return the maximum of the posterior by least squares in the log parameters.

    Args:
        forward: The forward model, taking the parameters in the order of ``bounds``.
        observations: The measurements.
        bounds: The prior range of each parameter.
        start: Starting values (default: the geometric centre of each range).
    """
    _check_bounds(bounds)
    names = list(bounds)
    low = np.log([b[0] for b in bounds.values()])
    high = np.log([b[1] for b in bounds.values()])
    x0 = np.log([start[n] for n in names]) if start is not None else 0.5 * (low + high)
    residual = _chi2(forward, observations)
    solution = least_squares(
        lambda x: residual(np.exp(x)),
        x0,
        bounds=(low, high),
        diff_step=1e-3,
        x_scale=1.0,
    )
    jacobian = solution.jac
    try:
        covariance = np.linalg.inv(jacobian.T @ jacobian)
        std = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    except np.linalg.LinAlgError:
        std = np.full(len(names), np.inf)
    return Fit(
        dict(zip(names, np.exp(solution.x).tolist(), strict=True)),
        dict(zip(names, std.tolist(), strict=True)),
        float(2.0 * solution.cost),
        bool(solution.success),
    )
