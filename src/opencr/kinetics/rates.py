"""Rate coefficients of every reaction of a network for given conditions.

:class:`RateEvaluator` prepares each reaction once and then returns the vector
of rate coefficients ``k_r`` (SI) for any :class:`~opencr.core.Conditions`,
or for many electron temperatures at once.

**Cross sections, Maxwellian.** plasmacoll's cross sections are linear between
their table points and constant beyond the last one, so
``k(T) = (8 / (pi mu))^1/2 (kT)^-3/2 int sigma(E) E exp(-E / kT) dE`` is a sum of
integrals of quadratics times an exponential, done in closed form per
segment: exact for the table, with no integration grid. Each evaluator
returns ``k exp(E_th / T)`` with the threshold ``E_th`` taken out, so that
detailed balance never forms ``exp(+E/T)`` of a tiny number.

**Cross sections, any EEDF.** With a non-Maxwellian
:class:`~opencr.kinetics.eedf.EEDF`, the electron-impact rates are
``(2 e / m_e)^1/2 int eps sigma F0 deps`` (:func:`plasmacoll.eedf_rate_coefficient`)
on the distribution's grid. Rates that are not cross sections use the
conditions' electron temperature (``2/3 <eps>`` from
:meth:`~opencr.core.Conditions.from_eedf`).

**Detailed balance.** The inverse of an electron-impact excitation uses the
Klein-Rosseland inverse cross section
``sigma_inv(eps) = (g_l / g_u) (eps + dE) / eps sigma(eps + dE)`` for any
EEDF, which for a Maxwellian is the Boltzmann relation
``k_inv = k (g_l / g_u) exp(dE / T)`` (used in closed form). An inverse with one
electron more on the forward product side (three-body recombination from
ionization) follows the Saha relation
``k_inv = k (g_0 / (g_e g_+)) lambda^3 exp(dE / T)`` with the electron's thermal
de Broglie volume ``lambda^3``, at the electron temperature (the effective one
for a non-Maxwellian EEDF). ``dE`` and the weights come from the states.

**Escape factors** multiply the spontaneous emission rates (radiation
trapping): the conditions' value for the reaction if given, else the
reaction's own.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import cunumpy as xp
import numpy as np
import scope_profiler as sp
from plasmacoll import CrossSection, eedf_rate_coefficient

from opencr._profiling import PROFILE_TAGS
from opencr.constants import ELEMENTARY_CHARGE, thermal_de_broglie_cubed
from opencr.core.conditions import Conditions
from opencr.core.network import ReactionNetwork
from opencr.core.rates import (
    Arrhenius,
    Constant,
    CrossSectionRate,
    DetailedBalance,
    Function,
    Tabulated,
)
from opencr.core.reactions import Reaction
from opencr.core.species import ELECTRON

__all__ = ["MaxwellianIntegral", "RateEvaluator", "inverse_cross_section"]

#: log of the smallest positive rate kept; below it a rate is zero.
_LOG_TINY = math.log(np.finfo(float).tiny)


@dataclass(frozen=True)
class MaxwellianIntegral:
    """The exact Maxwellian average of a piecewise-linear cross section.

    Attributes:
        threshold: The threshold in the centre-of-mass frame (eV).
        mass: The reduced mass (kg).
        y: The segment boundaries above the threshold (eV), ``n + 1`` values;
            the last segment continues to infinity with a constant cross section.
        c0, c1, c2: The coefficients of ``sigma(E) E = c0 + c1 y + c2 y^2`` per
            segment (``n + 1`` values, the last for the tail with ``c2 = 0``).
    """

    threshold: float
    mass: float
    y: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    c2: np.ndarray

    @classmethod
    def from_rate(cls, rate: CrossSectionRate) -> MaxwellianIntegral:
        """Prepare the integral of a cross-section rate model."""
        cs = rate.cross_section
        scale = 1.0 if rate.incident_mass is None else rate.incident_mass / rate.mass
        energy = np.asarray(xp.to_numpy(cs.energy), dtype=float) / scale
        threshold = cs.threshold / scale
        x = np.concatenate([[threshold], energy[energy > threshold]])
        sigma = np.asarray(xp.to_numpy(cs(xp.asarray(x * scale))), dtype=float)
        y = x - threshold
        slope = np.zeros_like(y)
        slope[:-1] = np.diff(sigma) / np.diff(y)
        a = sigma - slope * y
        return cls(
            threshold=float(threshold),
            mass=rate.mass,
            y=y,
            c0=a * threshold,
            c1=a + slope * threshold,
            c2=slope,
        )

    def log_scaled(self, temperature: np.ndarray) -> np.ndarray:
        """Return ``log(k exp(E_th / T))`` with ``k`` in m^3/s at ``temperature`` (eV)."""
        t = np.asarray(temperature, dtype=float)[..., None]
        y = self.y

        def antiderivative(points: np.ndarray, index: slice) -> np.ndarray:
            g = self.c0[index] + self.c1[index] * points + self.c2[index] * points**2
            dg = self.c1[index] + 2.0 * self.c2[index] * points
            return -t * np.exp(-points / t) * (g + t * dg + t**2 * 2.0 * self.c2[index])

        inner = slice(0, y.size - 1)
        segments = antiderivative(y[1:], inner) - antiderivative(y[:-1], inner)
        tail = -antiderivative(y[-1:], slice(y.size - 1, y.size))
        integral = segments.sum(axis=-1) + tail[..., 0]
        prefactor = (
            math.sqrt(8.0 / (math.pi * self.mass))
            * (t[..., 0] * ELEMENTARY_CHARGE) ** -1.5
            * ELEMENTARY_CHARGE**2
        )
        with np.errstate(divide="ignore"):
            return np.log(prefactor * np.maximum(integral, 0.0))


def inverse_cross_section(
    cross_section: CrossSection, weight_ratio: float, energy_gap: float
) -> CrossSection:
    """Return the Klein-Rosseland inverse of an electron-impact cross section.

    ``sigma_inv(eps) = weight_ratio (eps + dE) / eps sigma(eps + dE)`` with
    ``dE = energy_gap``; the table starts just above zero energy.
    """
    energy = np.asarray(xp.to_numpy(cross_section.energy), dtype=float)
    above = energy[energy > energy_gap] - energy_gap
    eps = np.unique(np.concatenate([[1.0e-6 * max(energy_gap, 1e-3)], above]))
    sigma = np.asarray(
        xp.to_numpy(cross_section(xp.asarray(eps + energy_gap))), dtype=float
    )
    return CrossSection(
        energy=eps,
        sigma=weight_ratio * (eps + energy_gap) / eps * sigma,
        label=f"inverse of {cross_section.label}",
    )


@dataclass(frozen=True)
class _Prepared:
    """How one reaction's coefficient is computed."""

    depends_on: str
    log_scaled: Callable[[np.ndarray], np.ndarray]
    threshold: float
    electron_cross_section: CrossSection | None = None
    #: for inverses: (forward index, log weight ratio, energy gap, extra electrons)
    inverse: tuple[int, float, float, int] | None = None


class RateEvaluator:
    """Rate coefficients of a network's reactions.

    Args:
        network: The reaction network.
    """

    def _prepare(self, reaction: Reaction, index: dict[str, int]) -> _Prepared:
        rate = reaction.rate
        if isinstance(rate, CrossSectionRate):
            integral = MaxwellianIntegral.from_rate(rate)
            electron = rate.cross_section if rate.depends_on == "electron" else None
            return _Prepared(
                rate.depends_on, integral.log_scaled, integral.threshold, electron
            )
        if isinstance(rate, Arrhenius):
            a, b, c = rate.a, rate.b, rate.c

            def arrhenius(t: np.ndarray) -> np.ndarray:
                with np.errstate(divide="ignore"):
                    return np.log(a) + b * np.log(t) + 0.0 * t

            return _Prepared(rate.depends_on, arrhenius, c)
        if isinstance(rate, Tabulated):
            log_t = np.log(rate.temperatures)
            with np.errstate(divide="ignore"):
                log_k = np.log(np.asarray(rate.values)) + rate.threshold / np.asarray(
                    rate.temperatures
                )

            def tabulated(t: np.ndarray) -> np.ndarray:
                clipped = np.clip(np.log(t), log_t[0], log_t[-1])
                return np.interp(clipped, log_t, log_k) + 0.0 * t

            return _Prepared(rate.depends_on, tabulated, rate.threshold)
        if isinstance(rate, Constant):
            value = rate.value

            def constant(t: np.ndarray) -> np.ndarray:
                with np.errstate(divide="ignore"):
                    return np.full(np.shape(t), np.log(value))

            return _Prepared(rate.depends_on, constant, 0.0)
        if isinstance(rate, Function):
            function, threshold = rate.function, rate.threshold

            def evaluated(t: np.ndarray) -> np.ndarray:
                k = np.asarray(function(t), dtype=float)
                with np.errstate(divide="ignore"):
                    return np.log(np.maximum(k, 0.0)) + threshold / t

            return _Prepared(rate.depends_on, evaluated, threshold)
        if isinstance(rate, DetailedBalance):
            return self._prepare_inverse(reaction, index)
        raise TypeError(
            f"{reaction.name}: unsupported rate model {type(rate).__name__}"
        )

    def __init__(self, network: ReactionNetwork) -> None:
        """Prepare every reaction."""
        self.network = network
        self._prepared: list[_Prepared] = []
        index = {r.name: i for i, r in enumerate(network.reactions)}
        for reaction in network.reactions:
            self._prepared.append(self._prepare(reaction, index))
        self._radiative = np.array(
            [r.kind == "spontaneous_emission" for r in network.reactions], dtype=bool
        )

    def _prepare_inverse(self, reaction: Reaction, index: dict[str, int]) -> _Prepared:
        forward_index = index[reaction.rate.forward]
        forward = self.network.reactions[forward_index]
        prepared = self._prepare(forward, index)
        weights_in = math.fsum(math.log(s.weight) for s in forward.reactants)
        weights_out = math.fsum(math.log(s.weight) for s in forward.products)
        extra = forward.product_counts.get(
            ELECTRON.name, 0
        ) - forward.reactant_counts.get(ELECTRON.name, 0)
        heavy_in = sum(1 for s in forward.reactants if s.species != ELECTRON)
        heavy_out = sum(1 for s in forward.products if s.species != ELECTRON)
        if heavy_in != heavy_out or extra < 0:
            raise NotImplementedError(
                f"{reaction.name}: detailed balance is implemented for the same number of "
                "heavy particles on both sides and extra electrons on the forward side"
            )
        if prepared.depends_on == "none":
            raise NotImplementedError(
                f"{reaction.name}: the forward rate has no temperature"
            )
        if extra > 0 and prepared.depends_on != "electron":
            raise NotImplementedError(
                f"{reaction.name}: a Saha inverse needs electron rates"
            )
        gap = forward.energy_change
        electron_cs = None
        if prepared.electron_cross_section is not None and extra == 0:
            electron_cs = inverse_cross_section(
                prepared.electron_cross_section,
                math.exp(weights_in - weights_out),
                prepared.threshold,
            )
        return _Prepared(
            prepared.depends_on,
            prepared.log_scaled,
            prepared.threshold,
            electron_cs,
            (forward_index, weights_in - weights_out, gap, extra),
        )

    @property
    def names(self) -> tuple[str, ...]:
        """The reaction names, in the order of the coefficient vector."""
        return tuple(r.name for r in self.network.reactions)

    def _log_k(self, prepared: _Prepared, t: np.ndarray) -> np.ndarray:
        log_k = prepared.log_scaled(t)
        if prepared.inverse is None:
            return log_k - prepared.threshold / t
        _, log_weights, gap, extra = prepared.inverse
        log_k = log_k + log_weights + (gap - prepared.threshold) / t
        if extra:
            log_k = log_k + extra * np.log(thermal_de_broglie_cubed(t))
        return log_k

    def _escape(self, k: np.ndarray, conditions: Conditions) -> np.ndarray:
        for i in np.flatnonzero(self._radiative):
            reaction = self.network.reactions[i]
            k[..., i] *= conditions.escape.get(reaction.name, reaction.escape)
        return k

    @sp.profile("opencr.rate_coefficients", tags=PROFILE_TAGS)
    def coefficients(self, conditions: Conditions) -> np.ndarray:
        """Return ``k_r`` in SI for every reaction at ``conditions``."""
        temperatures = conditions.heavy_temperature
        eedf = conditions.eedf
        tabulated = None
        if eedf is not None and not eedf.is_maxwellian:
            tabulated = eedf.tabulate()
        k = np.empty(len(self._prepared))
        for i, prepared in enumerate(self._prepared):
            if tabulated is not None and prepared.electron_cross_section is not None:
                k[i] = eedf_rate_coefficient(
                    prepared.electron_cross_section, tabulated.energy, tabulated.values
                )
                continue
            t = np.asarray(temperatures[prepared.depends_on], dtype=float)
            log_k = self._log_k(prepared, t)
            k[i] = 0.0 if log_k < _LOG_TINY else math.exp(min(float(log_k), 700.0))
        return self._escape(k, conditions)

    def coefficients_scan(
        self, conditions: Conditions, electron_temperatures: np.ndarray
    ) -> np.ndarray:
        """Return ``k`` of shape ``(len(electron_temperatures), n_reactions)``.

        Maxwellian electrons at each temperature; everything else from
        ``conditions``. Vectorised over the temperatures.
        """
        te = np.asarray(electron_temperatures, dtype=float).ravel()
        if np.any(te <= 0.0):
            raise ValueError("temperatures must be > 0")
        temperatures = conditions.heavy_temperature
        k = np.empty((te.size, len(self._prepared)))
        for i, prepared in enumerate(self._prepared):
            if prepared.depends_on == "electron":
                t = te
            else:
                t = np.full(te.size, temperatures[prepared.depends_on])
            log_k = self._log_k(prepared, t)
            k[:, i] = np.where(log_k < _LOG_TINY, 0.0, np.exp(np.minimum(log_k, 700.0)))
        return self._escape(k, conditions)
