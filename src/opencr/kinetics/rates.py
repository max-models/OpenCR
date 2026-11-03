"""Rate coefficients of every reaction of a network for given conditions.

`RateEvaluator` prepares each reaction once and then returns the vector
of rate coefficients ``k_r`` (SI) for any `Conditions`,
or for many electron temperatures at once.

**Cross sections, Maxwellian.** plasmacoll's cross sections are linear between
their table points and constant beyond the last one, so
``k(T) = (8 / (pi mu))^1/2 (kT)^-3/2 int sigma(E) E exp(-E / kT) dE`` is a sum of
integrals of quadratics times an exponential, done in closed form per
segment: exact for the table, with no integration grid. Each evaluator
returns ``k exp(E_th / T)`` with the threshold ``E_th`` taken out, so that
detailed balance never forms ``exp(+E/T)`` of a tiny number.

**Cross sections, any EEDF.** With a non-Maxwellian
`EEDF`, the electron-impact rates are
``(2 e / m_e)^1/2 int eps sigma F0 deps`` (`plasmacoll.eedf_rate_coefficient`)
on the distribution's grid. Rates that are not cross sections use the
conditions' electron temperature (``2/3 <eps>`` from
`from_eedf`).

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

**Arrays.** Every evaluator is written against cunumpy's ``xp`` (NumPy or
CuPy for the whole process, see `opencr.backends`; the optional JAX
backend passes ``jax.numpy`` in its place):
`RateEvaluator.coefficients_batched` evaluates all reactions at many
electron temperatures at once there, `RateEvaluator.coefficients` one
set of conditions, as floats on the host. A `Function`
rate runs on the host (its temperatures are copied there and the result
back); a non-Maxwellian EEDF is integrated on the host by plasmacoll and is
not batched.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import cunumpy as xp
import numpy as np
import scope_profiler as sp
from methodcard import ConvergenceProblem, Rule, method
from plasmacoll import CrossSection, eedf_rate_coefficient

from opencr._profiling import PROFILE_TAGS
from opencr.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, PLANCK
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
#: temperatures per chunk of a batched evaluation (bounds the (chunk, U, S) intermediate).
_CHUNK = 32
#: ``h^2 / (2 pi m_e e)`` in m^2 eV: ``lambda^3 = (this / T)^(3/2)``.
_DE_BROGLIE = PLANCK**2 / (2.0 * math.pi * ELECTRON_MASS * ELEMENTARY_CHARGE)


def _maxwellian_rules() -> dict:
    """The SymPy rule of `MaxwellianIntegral` (for the tests): a step cross section."""
    import sympy

    sigma, threshold, t = sympy.symbols("sigma threshold temperature", positive=True)
    e, m = sympy.Float(ELEMENTARY_CHARGE), sympy.Float(ELECTRON_MASS)
    rule = (
        sigma
        * sympy.sqrt(8 * e * t / (sympy.pi * m))
        * (1 + threshold / t)
        * sympy.exp(-threshold / t)
    )

    def rate(sigma: float, threshold: float, temperature: float) -> float:
        table = CrossSection.constant(sigma, threshold=threshold)
        integral = MaxwellianIntegral.from_rate(CrossSectionRate(cross_section=table))
        return math.exp(
            float(integral.log_scaled(np.array(temperature)))
            - integral.threshold / temperature
        )

    ranges = {"sigma": (1e-21, 1e-19), "threshold": (1.0, 15.0)}
    ranges["temperature"] = (0.5, 50.0)
    return {"maxwellian": Rule(rule, implementation=rate, ranges=ranges, rtol=1e-9)}


def _maxwellian_problem() -> ConvergenceProblem:
    """``sigma = sigma_0 (1 - E_th / E)`` on a uniform table, against its exact average.

    ``int sigma E exp(-E/T) dE = sigma_0 T^2 exp(-E_th/T)``; the table reaches
    ``E_th + 40 T``, past which the held value adds ``e^-40``.
    """
    threshold, temperature, sigma0 = 1.0, 1.0, 1.0e-20
    exact = math.log(
        math.sqrt(8.0 / (math.pi * ELECTRON_MASS))
        * (temperature * ELEMENTARY_CHARGE) ** -1.5
        * ELEMENTARY_CHARGE**2
        * sigma0
        * temperature**2
    )

    def error(step: float) -> float:
        energy = threshold + np.linspace(0.0, 40.0, round(40.0 / step) + 1)
        table = CrossSection(energy, sigma0 * (1.0 - threshold / energy), threshold)
        integral = MaxwellianIntegral.from_rate(CrossSectionRate(cross_section=table))
        return abs(float(integral.log_scaled(np.array(temperature))) - exact)

    return ConvergenceProblem(error, steps=[1 / 4, 1 / 8, 1 / 16, 1 / 32])


def _klein_rosseland_rules() -> dict:
    """The SymPy rule of `inverse_cross_section` (for the tests), on its table points.

    The forward cross section is ``1e-20 E`` m^2 (``E`` in eV) on a table with a
    point at ``energy_gap + energy``, where the inverse is evaluated.
    """
    import sympy

    def inverse(energy: float, weight_ratio: float, energy_gap: float) -> float:
        points = energy_gap + np.array([0.0, energy, 3.0 * energy])
        forward = CrossSection(points, 1e-20 * points, threshold=energy_gap)
        return float(
            inverse_cross_section(forward, weight_ratio, energy_gap)(np.array(energy))
        )

    energy, ratio, gap = sympy.symbols("energy weight_ratio energy_gap", positive=True)
    rule = ratio * (energy + gap) / energy * sympy.Float(1e-20) * (energy + gap)
    ranges = {"energy": (0.1, 10.0), "weight_ratio": (0.05, 1.0)}
    ranges["energy_gap"] = (1.0, 12.0)
    return {
        "klein_rosseland": Rule(rule, implementation=inverse, ranges=ranges, rtol=1e-9)
    }


def _detailed_balance_rules() -> dict:
    """The SymPy rules of the detailed-balance inverses (for the tests).

    Arrhenius forward rates ``a T^b exp(-c/T)``: an excitation ``H(1) -> H(n=2)``
    with a threshold ``c`` other than the state energy gap, and an ionization
    ``H(1) -> H+``, inverted by Boltzmann and by Saha.
    """
    import sympy

    from opencr.core.species import ELECTRON_STATE as E
    from opencr.core.species import State
    from opencr.data.hydrogen import hydrogen_levels

    (ground, upper), ion = hydrogen_levels(2)
    a, b = 1e-14, 0.5

    def inverse(forward: Reaction, backward: Reaction, temperature: float) -> float:
        states = {s.name: s for s in (*forward.reactants, *forward.products)}
        states.pop(E.name)
        network = ReactionNetwork(list(states.values()), [forward, backward])
        k = RateEvaluator(network).coefficients(Conditions(temperature))
        return float(k[1])

    def deexcitation(c: float, gap: float, temperature: float) -> float:
        lifted = State(
            upper.species, upper.label, gap, upper.weight, upper.quantum_numbers
        )
        forward = Reaction(
            "exc", (E, ground), (E, lifted), "excitation", Arrhenius(a=a, b=b, c=c)
        )
        backward = Reaction(
            "dex",
            (E, lifted),
            (E, ground),
            "deexcitation",
            DetailedBalance(forward="exc"),
        )
        return inverse(forward, backward, temperature)

    def recombination(c: float, temperature: float) -> float:
        forward = Reaction(
            "ion", (E, ground), (E, E, ion), "ionization", Arrhenius(a=a, b=b, c=c)
        )
        backward = Reaction(
            "rec", (E, E, ion), (E, ground), "three_body_recombination",
            DetailedBalance(forward="ion"),
        )  # fmt: skip
        return inverse(forward, backward, temperature)

    c, gap, t = sympy.symbols("c gap temperature", positive=True)
    forward = a * t**b * sympy.exp(-c / t)
    boltzmann = forward * sympy.Rational(2, 8) * sympy.exp(gap / t)
    h, me, e = (sympy.Float(x) for x in (PLANCK, ELECTRON_MASS, ELEMENTARY_CHARGE))
    debroglie = (h**2 / (2 * sympy.pi * me * e * t)) ** sympy.Rational(3, 2)
    energy = sympy.Float(ion.energy)
    saha = forward * sympy.Rational(2, 2 * 1) * debroglie * sympy.exp(energy / t)
    ranges = {"c": (5.0, 15.0), "gap": (5.0, 15.0), "temperature": (0.5, 20.0)}
    return {
        "boltzmann": Rule(
            boltzmann, implementation=deexcitation, ranges=ranges, rtol=1e-9
        ),
        "saha": Rule(saha, implementation=recombination, ranges=ranges, rtol=1e-9),
    }


def _host(array: Any) -> np.ndarray:
    """Return ``array`` on the host as a NumPy array (the execution boundary)."""
    try:
        return np.asarray(array, dtype=float)
    except TypeError:  # pragma: no cover - a device array refuses the implicit copy
        return np.asarray(xp.to_numpy(array), dtype=float)


@method(
    equations={
        "maxwellian": r"k(T) = \sqrt{\frac{8}{\pi\mu}}\,(kT)^{-3/2} \int_{E_{\mathrm{th}}}^\infty \sigma(E)\,E\,e^{-E/kT}\,dE",
        "segment": r"y = E - E_{\mathrm{th}}, \qquad \sigma(E)\,E = c_0 + c_1 y + c_2 y^2, \qquad \sigma = a + s\,y:\ c_0 = a E_{\mathrm{th}},\ c_1 = a + s E_{\mathrm{th}},\ c_2 = s",
        "antiderivative": r"\int g(y)\, e^{-y/T}\, dy = -T e^{-y/T}\big(g + T g' + 2 c_2 T^2\big)",
        "scaled": r"\ln\!\big(k\, e^{E_{\mathrm{th}}/T}\big) = \ln\Big[\sqrt{\tfrac{8}{\pi\mu}}\,(kT)^{-3/2}\, e^2 \sum_{\mathrm{segments} + \mathrm{tail}} \big[\cdots\big]\Big]",
    },
    order=2,
    assumptions=(
        r"exact for the piecewise-linear table, plus the constant tail to infinity ($c_2 = 0$); no energy grid",
        r"the order is that of the table spacing, for a smooth cross section sampled on the table",
        r"the threshold is taken out ($e^{-E/T} = e^{-E_{\mathrm{th}}/T} e^{-y/T}$), so detailed balance never forms $e^{+E/T}$ of a tiny number; $e^2$ because the integral is in eV$^2$ m$^2$",
    ),
    symbolic=_maxwellian_rules,
    problem=_maxwellian_problem,
)
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
        cs = rate.table
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

    def log_scaled(self, temperature: Any, xp: ModuleType = xp) -> Any:
        """Return ``log(k exp(E_th / T))`` with ``k`` in m^3/s at ``temperature`` (eV), on ``xp``."""
        t = xp.asarray(temperature, dtype=float)[..., None]
        y = xp.asarray(self.y)
        c0, c1, c2 = (xp.asarray(c) for c in (self.c0, self.c1, self.c2))

        def antiderivative(points: Any, index: slice) -> Any:
            g = c0[index] + c1[index] * points + c2[index] * points**2
            dg = c1[index] + 2.0 * c2[index] * points
            return -t * xp.exp(-points / t) * (g + t * dg + t**2 * 2.0 * c2[index])

        inner = slice(0, self.y.size - 1)
        segments = antiderivative(y[1:], inner) - antiderivative(y[:-1], inner)
        tail = -antiderivative(y[-1:], slice(self.y.size - 1, self.y.size))
        integral = xp.sum(segments, axis=-1) + tail[..., 0]
        prefactor = (
            math.sqrt(8.0 / (math.pi * self.mass))
            * (t[..., 0] * ELEMENTARY_CHARGE) ** -1.5
            * ELEMENTARY_CHARGE**2
        )
        with np.errstate(divide="ignore"):
            return xp.log(prefactor * xp.maximum(integral, 0.0))


@method(
    equations={
        "reversibility": r"g_l\,\varepsilon_l\,\sigma_{lu}(\varepsilon_l) = g_u\,\varepsilon_u\,\sigma_{ul}(\varepsilon_u), \qquad \varepsilon_l = \varepsilon_u + \Delta E",
        "klein_rosseland": r"\sigma_{ul}(\varepsilon) = \frac{g_l}{g_u}\,\frac{\varepsilon + \Delta E}{\varepsilon}\,\sigma_{lu}(\varepsilon + \Delta E)",
    },
    assumptions=(
        r"microscopic reversibility, so it holds for any electron energy distribution",
        r"tabulated on the forward table points shifted down by $\Delta E$, from $10^{-6}\max(\Delta E, 10^{-3})$ eV",
        r"$\Delta E$ is the cross section's threshold (13.6-based for hydrogen); the Maxwellian path uses the state energy gap (13.598-based)",
    ),
    references=(
        "[Klein and Rosseland, Z. Phys. 4, 46 (1921)](https://doi.org/10.1007/BF01328041)",
    ),
    symbolic=_klein_rosseland_rules,
)
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
    log_scaled: Callable[..., Any]
    threshold: float
    electron_cross_section: CrossSection | None = None
    #: for inverses: (forward index, log weight ratio, energy gap, extra electrons)
    inverse: tuple[int, float, float, int] | None = None
    #: the Maxwellian integral when the rate is a cross section (vectorised in batches)
    integral: MaxwellianIntegral | None = None
    #: the log of a constant rate, else None
    constant: float | None = None


class _Stack:
    """The Maxwellian integrals of a prepared list, padded into one set of arrays.

    ``log_scaled(t, xp)`` returns ``(..., U)`` for the ``U`` distinct integrals
    at temperatures ``t`` of shape ``(...)``, the vectorised form of
    `MaxwellianIntegral.log_scaled`; ``index[r]`` is the integral of
    reaction ``r`` (-1 for none).
    """

    def __init__(self, prepared: Sequence[_Prepared]) -> None:
        """Collect and pad the integrals."""
        unique: dict[int, int] = {}
        integrals: list[MaxwellianIntegral] = []
        self.index = np.full(len(prepared), -1, dtype=int)
        for r, item in enumerate(prepared):
            if item.integral is None:
                continue
            key = id(item.integral)
            if key not in unique:
                unique[key] = len(integrals)
                integrals.append(item.integral)
            self.index[r] = unique[key]
        u = len(integrals)
        width = max((g.y.size for g in integrals), default=1)
        self.y = np.zeros((u, width))
        self.c0 = np.zeros((u, width - 1))
        self.c1 = np.zeros((u, width - 1))
        self.c2 = np.zeros((u, width - 1))
        self.tail_y = np.zeros(u)
        self.tail_c0 = np.zeros(u)
        self.tail_c1 = np.zeros(u)
        self.log_prefactor = np.zeros(u)
        for k, g in enumerate(integrals):
            n = g.y.size
            self.y[k, :n] = g.y
            self.y[k, n:] = g.y[-1]  # repeated points: zero-length segments
            self.c0[k, : n - 1] = g.c0[:-1]
            self.c1[k, : n - 1] = g.c1[:-1]
            self.c2[k, : n - 1] = g.c2[:-1]
            self.tail_y[k] = g.y[-1]
            self.tail_c0[k] = g.c0[-1]
            self.tail_c1[k] = g.c1[-1]
            self.log_prefactor[k] = math.log(
                math.sqrt(8.0 / (math.pi * g.mass)) * ELEMENTARY_CHARGE**0.5
            )

    def log_scaled(self, temperature: Any, xp: ModuleType = xp) -> Any:
        """Return ``log(k exp(E_th / T))`` of every integral, shape ``(..., U)``."""
        t = xp.asarray(temperature, dtype=float)[..., None, None]
        y, c0, c1, c2 = (xp.asarray(a) for a in (self.y, self.c0, self.c1, self.c2))
        tail_y, tail_c0, tail_c1 = (
            xp.asarray(a) for a in (self.tail_y, self.tail_c0, self.tail_c1)
        )

        def antiderivative(points: Any, a0: Any, a1: Any, a2: Any) -> Any:
            g = a0 + a1 * points + a2 * points**2
            dg = a1 + 2.0 * a2 * points
            return -t * xp.exp(-points / t) * (g + t * dg + t**2 * 2.0 * a2)

        segments = antiderivative(y[:, 1:], c0, c1, c2) - antiderivative(
            y[:, :-1], c0, c1, c2
        )
        tail = -antiderivative(tail_y[:, None], tail_c0[:, None], tail_c1[:, None], 0.0)
        integral = xp.sum(segments, axis=-1) + tail[..., 0]
        kt = t[..., 0, 0][..., None] * ELEMENTARY_CHARGE
        with np.errstate(divide="ignore"):
            return (
                xp.asarray(self.log_prefactor)
                - 1.5 * xp.log(kt)
                + 1.5 * math.log(ELEMENTARY_CHARGE)
                + xp.log(xp.maximum(integral, 0.0))
            )


class RateEvaluator:
    """Rate coefficients of a network's reactions.

    Args:
        network: The reaction network.
    """

    def _prepare(self, reaction: Reaction, index: dict[str, int]) -> _Prepared:
        rate = reaction.rate
        if isinstance(rate, CrossSectionRate):
            integral = MaxwellianIntegral.from_rate(rate)
            electron = rate.table if rate.depends_on == "electron" else None
            return _Prepared(
                rate.depends_on,
                integral.log_scaled,
                integral.threshold,
                electron,
                integral=integral,
            )
        if isinstance(rate, Arrhenius):
            a, b, c = rate.a, rate.b, rate.c

            log_a = math.log(a) if a > 0.0 else -math.inf

            def arrhenius(t: Any, xp: ModuleType = xp) -> Any:
                return log_a + b * xp.log(t)

            return _Prepared(rate.depends_on, arrhenius, c)
        if isinstance(rate, Tabulated):
            log_t = np.log(rate.temperatures)
            with np.errstate(divide="ignore"):
                log_k = np.log(np.asarray(rate.values)) + rate.threshold / np.asarray(
                    rate.temperatures
                )

            def tabulated(t: Any, xp: ModuleType = xp) -> Any:
                clipped = xp.clip(xp.log(t), float(log_t[0]), float(log_t[-1]))
                return xp.interp(clipped, xp.asarray(log_t), xp.asarray(log_k))

            return _Prepared(rate.depends_on, tabulated, rate.threshold)
        if isinstance(rate, Constant):
            value = rate.value

            log_value = math.log(value) if value > 0.0 else -math.inf

            def constant(t: Any, xp: ModuleType = xp) -> Any:
                return xp.full(t.shape, log_value)

            return _Prepared(rate.depends_on, constant, 0.0, constant=log_value)
        if isinstance(rate, Function):
            function, threshold = rate.function, rate.threshold
            assert function is not None

            def evaluated(t: Any, xp: ModuleType = xp) -> Any:
                # the function is host code: cross the boundary and back
                k = np.asarray(function(_host(t)), dtype=float)
                with np.errstate(divide="ignore"):
                    log_k = np.log(np.maximum(k, 0.0))
                return xp.asarray(log_k) + threshold / t

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
        self._stacks: dict[int, tuple[Sequence[_Prepared], _Stack]] = {}

    def _prepare_inverse(self, reaction: Reaction, index: dict[str, int]) -> _Prepared:
        assert isinstance(reaction.rate, DetailedBalance)
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
            integral=prepared.integral,
            constant=prepared.constant,
        )

    @property
    def names(self) -> tuple[str, ...]:
        """The reaction names, in the order of the coefficient vector."""
        return tuple(r.name for r in self.network.reactions)

    def _log_k(self, prepared: _Prepared, t: Any, xp: ModuleType = xp) -> Any:
        log_k = prepared.log_scaled(t, xp)
        if prepared.inverse is None:
            return log_k - prepared.threshold / t
        _, log_weights, gap, extra = prepared.inverse
        log_k = log_k + log_weights + (gap - prepared.threshold) / t
        if extra:
            log_k = log_k + extra * 1.5 * (math.log(_DE_BROGLIE) - xp.log(t))
        return log_k

    def escape_factors(self, conditions: Conditions) -> np.ndarray:
        """Return the factor of every reaction: the escape factor of a line, 1 otherwise."""
        factor = np.ones(len(self._prepared))
        for i in np.flatnonzero(self._radiative):
            reaction = self.network.reactions[i]
            factor[i] = conditions.escape.get(reaction.name, reaction.escape)
        return factor

    def _escape(self, k: np.ndarray, conditions: Conditions) -> np.ndarray:
        return k * self.escape_factors(conditions)

    @method(
        equations={
            "scaled": r"\ln k = \ln\!\big(k\, e^{E_{\mathrm{th}}/T}\big) - \frac{E_{\mathrm{th}}}{T}, \qquad k = 0 \ \text{if}\ \ln k < \ln(2.2\times10^{-308}), \qquad \ln k \le 700",
            "inverse": r"\ln k_{\mathrm{inv}} = \ln\!\big(k_{\mathrm{fwd}}\, e^{E_{\mathrm{th}}/T}\big) + \ln\frac{g_{\mathrm{in}}}{g_{\mathrm{out}}} + \frac{\Delta E - E_{\mathrm{th}}}{T} + \tfrac32\, \Delta e\, \Big[\ln\frac{h^2}{2\pi m_e e} - \ln T\Big]",
            "boltzmann": r"k_{ul} = k_{lu}\,\frac{g_l}{g_u}\, e^{\Delta E/T}",
            "saha": r"k_{3b} = k_{\mathrm{ion}}\,\frac{g_0}{g_e\, g_+}\,\Lambda^3\, e^{I/T}, \qquad \Lambda^3 = \left(\frac{h^2}{2\pi m_e kT}\right)^{3/2}",
            "escape": r"k_r \leftarrow g_r\, k_r \quad \text{for every spontaneous emission}",
        },
        assumptions=(
            r"`depends_on` picks the temperature: $T_e$ ($T_{\mathrm{eff}}$ for a non-Maxwellian EEDF), $T_i$ (the gas temperature when not given), $T_{\mathrm{gas}}$, or none",
            r"with a non-Maxwellian EEDF only the electron cross sections (and their Klein-Rosseland inverses) are integrated over it (`plasmacoll.eedf_rate_coefficient`)",
            r"the escape factor $g_r$ is `Conditions.escape` by name, else the reaction's own `escape`",
        ),
        references=(
            "[Klein and Rosseland, Z. Phys. 4, 46 (1921)](https://doi.org/10.1007/BF01328041)",
            "[Tiesinga et al., Rev. Mod. Phys. 93, 025010 (2021)](https://doi.org/10.1103/RevModPhys.93.025010) (CODATA 2018)",
        ),
        components=(MaxwellianIntegral, inverse_cross_section),
        symbolic=_detailed_balance_rules,
    )
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
            t = xp.asarray(temperatures[prepared.depends_on], dtype=float)
            log_k = float(self._log_k(prepared, t))
            k[i] = 0.0 if log_k < _LOG_TINY else math.exp(min(log_k, 700.0))
        return self._escape(k, conditions)

    @sp.profile("opencr.rate_coefficients_batched", tags=PROFILE_TAGS)
    def coefficients_batched(self, conditions: Conditions, electron_temperatures: Any):
        """Return ``k`` of shape ``(len(electron_temperatures), n_reactions)`` on ``xp``.

        Maxwellian electrons at each temperature; everything else (the heavy
        temperatures, the escape factors) from ``conditions``. The input is
        copied to ``xp`` and the result stays there (``xp.to_numpy`` brings it
        back).

        Raises:
            ValueError: A non-Maxwellian EEDF (integrated on the host only), or
                a temperature that is not positive.
        """
        if conditions.eedf is not None and not conditions.eedf.is_maxwellian:
            raise ValueError(
                "a batched evaluation is Maxwellian; a non-Maxwellian EEDF is "
                "integrated per point with RateEvaluator.coefficients"
            )
        te = xp.asarray(np.ravel(_host(electron_temperatures)), dtype=float)
        if te.size and bool(xp.min(te) <= 0.0):
            raise ValueError("temperatures must be > 0")
        chunks = [
            self.evaluate(self._prepared, conditions, te[i : i + _CHUNK])
            for i in range(0, max(te.size, 1), _CHUNK)
        ]
        return xp.concatenate(chunks, axis=0)

    def _stack(self, prepared: Sequence[_Prepared]) -> _Stack:
        key = id(prepared)
        if key not in self._stacks:
            self._stacks[key] = (prepared, _Stack(prepared))
        return self._stacks[key][1]

    def evaluate(
        self,
        prepared: Sequence[_Prepared],
        conditions: Conditions,
        electron_temperatures: Any,
        xp: ModuleType = xp,
    ) -> Any:
        """Return ``k`` of shape ``(..., n_reactions)`` on the namespace ``xp``.

        The core of `coefficients_batched`, for a given list of prepared
        reactions (the JAX backend passes its own, with ``Function`` rates
        tabulated) and any array namespace with the NumPy API; ``conditions``
        gives the heavy temperatures and the escape factors. The Maxwellian
        integrals of all cross sections are evaluated at once
        (`_Stack`), the thresholds and detailed-balance factors as
        vectors; only Arrhenius, tabulated and function rates go one by one.
        An intermediate of ``(..., U, S)`` numbers (integrals times table
        points) is formed, so a long batch is better evaluated in chunks.
        """
        te = xp.asarray(electron_temperatures, dtype=float)
        stack = self._stack(prepared)
        heavy = conditions.heavy_temperature
        n = len(prepared)
        electron = np.array([p.depends_on == "electron" for p in prepared])
        other = np.array([heavy[p.depends_on] for p in prepared])
        t = xp.where(
            xp.asarray(electron),
            te[..., None],
            xp.asarray(other) * xp.ones_like(te)[..., None],
        )
        logs = stack.log_scaled(te, xp) if stack.y.size else None
        constants = np.array(
            [math.nan if p.constant is None else p.constant for p in prepared]
        )
        columns = []
        for r, item in enumerate(prepared):
            if item.integral is not None:
                assert logs is not None
                columns.append(logs[..., stack.index[r]])
            elif item.constant is not None:
                columns.append(xp.full(te.shape, constants[r]))
            else:
                columns.append(item.log_scaled(t[..., r], xp))
        log_k = xp.stack(columns, axis=-1)
        threshold = np.array([p.threshold for p in prepared])
        inverse = np.array([p.inverse is not None for p in prepared])
        log_weights = np.array([p.inverse[1] if p.inverse else 0.0 for p in prepared])
        gap = np.array([p.inverse[2] if p.inverse else 0.0 for p in prepared])
        extra = np.array(
            [p.inverse[3] if p.inverse else 0 for p in prepared], dtype=float
        )
        # forward: log k = log(k e^{E/T}) - E/T; inverse: + log g + (dE - E)/T + extra log lambda^3
        shift = xp.where(
            xp.asarray(inverse), xp.asarray(gap - threshold), -xp.asarray(threshold)
        )
        log_k = log_k + xp.asarray(log_weights) + shift / t
        log_k = log_k + xp.asarray(extra) * 1.5 * (math.log(_DE_BROGLIE) - xp.log(t))
        k = xp.where(log_k < _LOG_TINY, 0.0, xp.exp(xp.minimum(log_k, 700.0)))
        assert k.shape[-1] == n
        return k * xp.asarray(self.escape_factors(conditions))

    def coefficients_scan(
        self, conditions: Conditions, electron_temperatures: np.ndarray
    ) -> np.ndarray:
        """Return ``k`` of shape ``(len(electron_temperatures), n_reactions)`` on the host."""
        return _host(self.coefficients_batched(conditions, electron_temperatures))
