"""The quasi-static reduction of many conditions at once, on cunumpy.

For a network that is linear in its states (one heavy reactant per reaction)
the rate matrix of one set of conditions is

    M = N diag(k n_e^(a_e)) S,

with ``N`` the net stoichiometry (states x reactions), ``S`` the selector of
each reaction's heavy reactant (reactions x states) and ``a_e`` the number
of electrons among the reactants. For a batch of ``B`` electron temperatures
and densities the matrices are a dense ``(B, n, n)`` array assembled in one
``einsum``, and the quasi-static reduction of :mod:`opencr.kinetics.effective`
is a batched dense solve, ``C = -M_QQ^-1 M_QP``, ``M_eff = M_PP + M_PQ C``.
Every point is independent, so the batch is the natural unit of work for a
GPU (``cunumpy.set_backend("cupy")``); the result comes back to the host and
is checked against the per-point sparse reference solves in the tests.

The drivers are the states with a density in the conditions, or the model's
drivers when the conditions give none; the driver densities do not enter the
coefficients (the system is linear).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cunumpy as xp
import numpy as np
import scope_profiler as sp
import xarray as xr

from opencr._profiling import PROFILE_TAGS
from opencr.backends import device
from opencr.core.conditions import Conditions
from opencr.kinetics.effective import SingularSystemError
from opencr.kinetics.system import PopulationSystem
from opencr.model import Model

__all__ = ["BatchedQuasiStatic", "BatchedReduction", "batched_scan"]


@dataclass(frozen=True)
class BatchedReduction:
    """The quasi-static reduction at a batch of conditions (NumPy arrays).

    Attributes:
        electron_temperatures: ``T_e`` in eV, shape ``(B,)``.
        electron_densities: ``n_e`` in m^-3, shape ``(B,)``.
        drivers: The names of the prescribed states ``P``.
        excited: The names of the quasi-static states ``Q``.
        coefficients: ``C`` with shape ``(B, len(excited), len(drivers))``.
        effective: ``M_eff`` with shape ``(B, len(drivers), len(drivers))`` in s^-1.
        rates: ``k_r`` in SI with shape ``(B, n_reactions)``.
        lines: The names of the spontaneous emissions.
        pec: The photon emissivity coefficients ``(B, len(lines), len(drivers))``
            in m^3/s.
    """

    electron_temperatures: np.ndarray
    electron_densities: np.ndarray
    drivers: tuple[str, ...]
    excited: tuple[str, ...]
    coefficients: np.ndarray
    effective: np.ndarray
    rates: np.ndarray
    lines: tuple[str, ...]
    pec: np.ndarray

    def effective_rate(self, source: str, target: str) -> np.ndarray:
        """Return ``M_eff[target, source] / n_e`` in m^3/s for every point."""
        i, j = self.drivers.index(target), self.drivers.index(source)
        return self.effective[:, i, j] / self.electron_densities

    def population_coefficient(self, state: str, driver: str) -> np.ndarray:
        """Return ``n_state / (n_e n_driver)`` in m^3 for every point."""
        i, j = self.excited.index(state), self.drivers.index(driver)
        return self.coefficients[:, i, j] / self.electron_densities

    def to_dataset(self) -> xr.Dataset:
        """Return the effective rates, population coefficients and PECs over ``point``."""
        ne = self.electron_densities
        return xr.Dataset(
            {
                "effective_rate": (
                    ("point", "target", "source"),
                    self.effective / ne[:, None, None],
                    {"units": "m^3 s^-1", "description": "M_eff[target, source] / n_e"},
                ),
                "population_coefficient": (
                    ("point", "state", "driver"),
                    self.coefficients / ne[:, None, None],
                    {"units": "m^3"},
                ),
                "pec": (("point", "line", "driver"), self.pec, {"units": "m^3 s^-1"}),
            },
            coords={
                "electron_temperature": (
                    "point",
                    self.electron_temperatures,
                    {"units": "eV"},
                ),
                "electron_density": ("point", ne, {"units": "m^-3"}),
                "target": list(self.drivers),
                "source": list(self.drivers),
                "driver": list(self.drivers),
                "state": list(self.excited),
                "line": list(self.lines),
            },
        )


class BatchedQuasiStatic:
    """The batched quasi-static reduction of a model.

    Args:
        model: The model; its network must be linear in the states.

    Raises:
        ValueError: A reaction has two heavy reactants.
    """

    def __init__(self, model: Model) -> None:
        """Prepare the structure arrays on ``xp``."""
        self.model = model
        network = model.network
        template = PopulationSystem(
            network,
            np.ones(len(network.reactions)),
            Conditions(1.0, 1.0),
            network.state_names,
        )
        if template.max_order > 1:
            raise ValueError(
                "the batched reduction needs every reaction to have at most one heavy "
                "reactant"
            )
        n_r, n_s = len(network.reactions), len(network.state_names)
        selector = np.zeros((n_r, n_s))
        for r, i in enumerate(template._slots[:, 0]):
            if i >= 0:
                selector[r, i] = 1.0
        self._stoichiometry = xp.asarray(template.stoichiometry.toarray())
        self._selector = xp.asarray(selector)
        self._electron_order = xp.asarray(template._electron_order)
        self._lines = tuple(
            r.name for r in network.reactions if r.kind == "spontaneous_emission"
        )
        self._line_index = np.array(
            [
                i
                for i, r in enumerate(network.reactions)
                if r.kind == "spontaneous_emission"
            ],
            dtype=int,
        )
        self._line_upper = [
            r.reactants[0].name
            for r in network.reactions
            if r.kind == "spontaneous_emission"
        ]

    def drivers_of(self, conditions: Conditions) -> tuple[str, ...]:
        """Return the drivers: the prescribed states of ``conditions``, else the model's."""
        names = self.model.network.state_names
        drivers = tuple(n for n in names if n in conditions.densities)
        return drivers or tuple(d for d in names if d in self.model.drivers)

    @sp.profile("opencr.batched_matrices", tags=PROFILE_TAGS)
    def matrices(
        self,
        conditions: Conditions,
        electron_temperatures: Any,
        electron_densities: Any,
    ) -> tuple[Any, Any]:
        """Return ``k`` of shape ``(B, n_reactions)`` and ``M`` of shape ``(B, n, n)`` on ``xp``."""
        te = xp.asarray(np.ravel(electron_temperatures), dtype=float)
        ne = xp.asarray(np.ravel(electron_densities), dtype=float)
        k = self.model.evaluator.coefficients_batched(conditions, te)
        weighted = k * ne[:, None] ** self._electron_order
        matrix = xp.einsum(
            "ir,br,rj->bij", self._stoichiometry, weighted, self._selector
        )
        return k, matrix

    @sp.profile("opencr.batched_reduction", tags=PROFILE_TAGS)
    def run(
        self,
        conditions: Conditions,
        electron_temperatures: Any,
        electron_densities: Any,
    ) -> BatchedReduction:
        """Return the reduction at every pair of ``electron_temperatures`` and ``electron_densities``.

        The two are broadcast against each other and flattened; the result's
        arrays follow that flat order.
        """
        te, ne = np.broadcast_arrays(
            np.asarray(electron_temperatures, dtype=float),
            np.asarray(electron_densities, dtype=float),
        )
        te, ne = np.ravel(te), np.ravel(ne)
        if np.any(ne <= 0.0):
            raise ValueError("electron densities must be > 0")
        drivers = self.drivers_of(conditions)
        if not drivers:
            raise ValueError("no drivers: give Conditions.densities or Model.drivers")
        names = self.model.network.state_names
        index = {n: i for i, n in enumerate(names)}
        excited = tuple(n for n in names if n not in drivers)
        p = [index[d] for d in drivers]
        q = [index[s] for s in excited]
        k, matrix = self.matrices(conditions, te, ne)
        m_qq = matrix[:, q][:, :, q]
        m_qp = matrix[:, q][:, :, p]
        m_pq = matrix[:, p][:, :, q]
        m_pp = matrix[:, p][:, :, p]
        diagonal = xp.to_numpy(xp.diagonal(m_qq, axis1=1, axis2=2))
        lossless = [excited[i] for i in np.flatnonzero(np.any(diagonal == 0.0, axis=0))]
        if lossless:
            raise SingularSystemError(
                f"the states {lossless} have no loss process; they cannot be quasi-static. "
                "Prescribe their densities or add their losses."
            )
        if q:
            try:
                coefficients = -xp.linalg.solve(m_qq, m_qp)
            except Exception as error:  # LinAlgError of either namespace
                raise SingularSystemError(
                    f"the quasi-static matrix is singular ({error}): a group of states has "
                    "no path to the drivers"
                ) from error
            effective = m_pp + m_pq @ coefficients
        else:
            coefficients = xp.zeros((te.size, 0, len(p)))
            effective = m_pp
        c = np.asarray(xp.to_numpy(coefficients), dtype=float)
        k_host = np.asarray(xp.to_numpy(k), dtype=float)
        pec = np.zeros((te.size, len(self._lines), len(drivers)))
        for j, (upper, r) in enumerate(
            zip(self._line_upper, self._line_index, strict=True)
        ):
            if upper in excited:
                pec[:, j, :] = (
                    k_host[:, r, None] * c[:, excited.index(upper), :] / ne[:, None]
                )
        return BatchedReduction(
            te,
            ne,
            drivers,
            excited,
            c,
            np.asarray(xp.to_numpy(effective), dtype=float),
            k_host,
            self._lines,
            pec,
        )


def batched_scan(
    model: Model,
    conditions: Conditions,
    electron_temperature: Sequence[float],
    electron_density: Sequence[float],
) -> xr.Dataset:
    """Return the effective rates and PECs on a grid of ``T_e`` and ``n_e``.

    The batched counterpart of :func:`opencr.scan` for quasi-static solves of
    Maxwellian plasmas: the same variables (``effective_rate`` over
    ``(electron_temperature, electron_density, target, source)``,
    ``pec`` over ``(..., line, driver)`` and ``population_coefficient`` over
    ``(..., state, driver)``), computed in one batch on cunumpy's ``xp``.
    """
    te = np.asarray(electron_temperature, dtype=float)
    ne = np.asarray(electron_density, dtype=float)
    grid_t, grid_n = np.meshgrid(te, ne, indexing="ij")
    reduction = BatchedQuasiStatic(model).run(conditions, grid_t, grid_n)
    data = reduction.to_dataset()
    shape = (te.size, ne.size)
    out = xr.Dataset(
        {
            name: (
                ("electron_temperature", "electron_density", *var.dims[1:]),
                var.values.reshape(*shape, *var.shape[1:]),
                var.attrs,
            )
            for name, var in data.data_vars.items()
        },
        coords={
            "electron_temperature": ("electron_temperature", te, {"units": "eV"}),
            "electron_density": ("electron_density", ne, {"units": "m^-3"}),
            **{k: v for k, v in data.coords.items() if "point" not in v.dims},
        },
        attrs={
            "model": model.name,
            "solve": "quasi-static",
            "device": device(),
        },
    )
    return out
