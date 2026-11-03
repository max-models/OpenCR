"""How the rate coefficient of a reaction is given.

A rate model describes the data, not how they are evaluated: the evaluation
(Maxwellian or any electron energy distribution, on any backend) is in
`opencr.kinetics.rates`. The rate of a reaction with reactants ``r_i``
is ``k prod n_i`` (a reactant that appears twice counts twice), so ``k`` is in
s^-1 for one reactant, m^3/s for two and m^6/s for three.

``depends_on`` names the temperature a model depends on: ``"electron"``
(``T_e``, or for a non-Maxwellian distribution the effective temperature
``2/3 <eps>``), ``"ion"``, ``"gas"`` or ``"none"``.

The masses (kg), energies and temperatures (eV) of a rate model take an
astropy Quantity or a string such as ``"11604 K"`` as well as a number (see
`plasmacoll.units`). The rate coefficients themselves (``Arrhenius.a``,
``Tabulated.values``, ``Constant.value``) are numbers in the SI unit of the
reaction's order, which the rate model does not know.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from methodcard import ConvergenceProblem, Rule, method
from plasmacoll import CrossSection
from plasmacoll.units import convert_units, unit_field

from opencr.constants import ELECTRON_MASS

__all__ = [
    "TEMPERATURES",
    "Arrhenius",
    "Constant",
    "CrossSectionRate",
    "DetailedBalance",
    "Function",
    "RateModel",
    "Tabulated",
]

#: The temperatures a rate model can depend on.
TEMPERATURES = ("electron", "ion", "gas", "none")


@dataclass(frozen=True)
class RateModel:
    """Base class of the rate models; ``kind`` names the model in files."""

    kind: ClassVar[str] = ""
    depends_on: str = "electron"

    def __post_init__(self) -> None:
        """Convert the fields with a unit and check ``depends_on``."""
        convert_units(self)
        if self.depends_on not in TEMPERATURES:
            raise ValueError(f"depends_on must be one of {TEMPERATURES}")

    def to_dict(self) -> dict[str, Any]:
        """Return the model as plain data (see `opencr.data.native`)."""
        raise NotImplementedError(f"{type(self).__name__} cannot be written to a file")


def _evaluate(rate: RateModel, temperature: Any) -> np.ndarray:
    """``k`` of ``rate`` at the temperatures (eV) of an excitation ``H(1) -> H(2)``.

    For the tests of the cards: the rate evaluated as a network evaluates it.
    """
    from opencr.core.conditions import Conditions
    from opencr.core.network import ReactionNetwork
    from opencr.core.reactions import Reaction
    from opencr.core.species import ELECTRON_STATE as E
    from opencr.data.hydrogen import hydrogen_levels
    from opencr.kinetics.rates import RateEvaluator

    (ground, excited), _ = hydrogen_levels(2)
    network = ReactionNetwork(
        [ground, excited],
        [Reaction("r", (E, ground), (E, excited), "excitation", rate)],
    )
    k = RateEvaluator(network).coefficients_scan(
        Conditions(1.0), np.atleast_1d(temperature)
    )
    return k[:, 0]


def _arrhenius_rules() -> dict:
    """The SymPy rule of `Arrhenius` (for the tests)."""
    import sympy

    a, b, c, t = sympy.symbols("a b c temperature", real=True)

    def rate(a: float, b: float, c: float, temperature: float) -> float:
        return float(_evaluate(Arrhenius(a=a, b=b, c=c), temperature)[0])

    ranges = {"a": (1e-15, 1e-13), "b": (-1.0, 1.0), "c": (0.5, 5.0)}
    ranges["temperature"] = (0.3, 30.0)
    return {
        "arrhenius": Rule(
            a * t**b * sympy.exp(-c / t), implementation=rate, ranges=ranges
        )
    }


def _tabulated_problem() -> ConvergenceProblem:
    """``k = exp(-(ln T)^2 - E/T)`` tabulated with ``E`` removed, at the interval midpoints."""
    threshold, low, high = 2.0, 0.5, 20.0
    span = np.log(high / low)

    def error(step: float) -> float:
        log_t = np.linspace(np.log(low), np.log(high), round(span / step) + 1)
        t = np.exp(log_t)
        rate = Tabulated(
            temperatures=tuple(t),
            values=tuple(np.exp(-(log_t**2) - threshold / t)),
            threshold=threshold,
        )
        middle = np.exp(0.5 * (log_t[1:] + log_t[:-1]))
        exact = -(np.log(middle) ** 2) - threshold / middle
        return float(np.max(np.abs(np.log(_evaluate(rate, middle)) - exact)))

    return ConvergenceProblem(error, steps=[span / 8, span / 16, span / 32, span / 64])


@method(
    equations={
        "maxwellian": r"k(T) = \sqrt{\frac{8}{\pi\mu}}\,(kT)^{-3/2} \int_{E_{\mathrm{th}}}^\infty \sigma(E)\,E\,e^{-E/kT}\,dE",
        "lab_frame": r"E_{\mathrm{lab}} = E\,\frac{m_{\mathrm{inc}}}{\mu}",
        "eedf": r"k = \sqrt{\frac{2e}{m_e}} \int_0^\infty \varepsilon\,\sigma(\varepsilon)\,F_0(\varepsilon)\,d\varepsilon",
    },
    assumptions=(
        r"$\sigma$ a plasmacoll `CrossSection`: linear between the table points, held at the last value above, zero below the threshold",
        r"a Maxwellian of the reduced mass $\mu$ at $T_e$, $T_i$ or $T_{\mathrm{gas}}$ (`depends_on`); for electrons the conditions' EEDF when it is not Maxwellian",
        r"a lab-frame table (target at rest) is read at $E_{\mathrm{lab}}$; a centre-of-mass table at $E$",
    ),
    references=(
        "[Hagelaar and Pitchford, Plasma Sources Sci. Technol. 14, 722 (2005)](https://doi.org/10.1088/0963-0252/14/4/011)",
    ),
    components=(
        "opencr.kinetics.rates.MaxwellianIntegral",
        "opencr.kinetics.eedf.EEDF",
    ),
)
@dataclass(frozen=True)
class CrossSectionRate(RateModel):
    """A rate from a cross section averaged over a velocity distribution.

    For electrons (``depends_on="electron"``) the average is over the electron
    energy distribution of the conditions (Maxwellian by default); otherwise over
    a Maxwellian of relative velocities at the ion or gas temperature.

    Attributes:
        cross_section: The cross section (m^2) against energy (eV).
        mass: The reduced mass in kg of the colliding pair.
        incident_mass: For a table against the incident particle's energy with
            the target at rest (the lab frame), the incident mass in kg; the
            table is looked up at ``E incident_mass / mass``. ``None`` for a
            table in the centre-of-mass frame. For electrons both agree.
    """

    kind: ClassVar[str] = "cross_section"
    cross_section: CrossSection | None = None
    mass: float = unit_field("kg", ELECTRON_MASS)
    incident_mass: float | None = unit_field("kg", None)

    def __post_init__(self) -> None:
        """Check the masses and the cross section."""
        super().__post_init__()
        if not isinstance(self.cross_section, CrossSection):
            raise TypeError("cross_section must be a plasmacoll.CrossSection")
        if self.mass <= 0.0 or (
            self.incident_mass is not None and self.incident_mass <= 0.0
        ):
            raise ValueError("masses must be > 0")
        if self.depends_on == "none":
            raise ValueError("a cross-section rate depends on a temperature")

    @property
    def table(self) -> CrossSection:
        """The cross section (never ``None`` after construction)."""
        assert self.cross_section is not None
        return self.cross_section

    @property
    def threshold(self) -> float:
        """The threshold of the cross section in eV (in its own frame)."""
        return float(self.table.threshold)

    def to_dict(self) -> dict[str, Any]:
        """Return the table, threshold and masses."""
        import cunumpy as xp

        return {
            "kind": self.kind,
            "depends_on": self.depends_on,
            "energy": np.asarray(xp.to_numpy(self.table.energy)).tolist(),
            "sigma": np.asarray(xp.to_numpy(self.table.sigma)).tolist(),
            "threshold": self.threshold,
            "label": self.table.label,
            "mass": self.mass,
            "incident_mass": self.incident_mass,
        }


@method(
    equations={
        "arrhenius": r"k = a\,T^b\,e^{-c/T}",
    },
    assumptions=(
        r"$T$ in eV, $k$ in the SI unit of the reaction's order; $c$ is the threshold taken out before evaluation",
    ),
    symbolic=_arrhenius_rules,
)
@dataclass(frozen=True)
class Arrhenius(RateModel):
    """``k = a T^b exp(-c / T)`` with ``T`` in eV and ``k`` in SI.

    Attributes:
        a: The prefactor in the SI unit of the reaction's order.
        b: The temperature exponent.
        c: The activation energy in eV.
    """

    kind: ClassVar[str] = "arrhenius"
    a: float = 0.0
    b: float = 0.0
    c: float = unit_field("eV", 0.0)

    def __post_init__(self) -> None:
        """Check the prefactor."""
        super().__post_init__()
        if self.a < 0.0:
            raise ValueError("the Arrhenius prefactor must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        """Return the three parameters."""
        return {
            "kind": self.kind,
            "depends_on": self.depends_on,
            "a": self.a,
            "b": self.b,
            "c": self.c,
        }


@method(
    equations={
        "interpolation": r"\ln\!\big(k\, e^{E_{\mathrm{th}}/T}\big) \text{ linear in } \ln T \text{ between the table points}",
        "extrapolation": r"k(T) = k_{\mathrm{end}}\, e^{E_{\mathrm{th}}/T_{\mathrm{end}} - E_{\mathrm{th}}/T} \quad \text{outside the table } (\ln T \text{ clipped})",
    },
    order=2,
    assumptions=(
        r"the order is in the table spacing in $\ln T$, for a smooth $\ln(k\,e^{E_{\mathrm{th}}/T})$",
        r"outside the table the scaled quantity $k\,e^{E_{\mathrm{th}}/T}$ holds its end value, not $k$ itself",
    ),
    problem=_tabulated_problem,
)
@dataclass(frozen=True)
class Tabulated(RateModel):
    """``k(T)`` from a table, interpolated linearly in ``log k`` against ``log T``.

    A ``threshold`` (eV) removes the Boltzmann factor before interpolating,
    ``log(k exp(threshold / T))``, which keeps the interpolation accurate at
    temperatures far below a threshold. Outside the table this scaled
    quantity holds its end value, so ``k = k_end exp(threshold / T_end -
    threshold / T)`` there.

    Attributes:
        temperatures: The table temperatures in eV, increasing.
        values: ``k`` in SI at ``temperatures``, non-negative.
        threshold: The energy in eV of the Boltzmann factor to remove.
    """

    kind: ClassVar[str] = "tabulated"
    temperatures: tuple[float, ...] = unit_field("eV", ())
    values: tuple[float, ...] = ()
    threshold: float = unit_field("eV", 0.0)

    def __post_init__(self) -> None:
        """Check and freeze the table."""
        super().__post_init__()
        t = np.asarray(self.temperatures, dtype=float)
        k = np.asarray(self.values, dtype=float)
        if t.ndim != 1 or t.shape != k.shape or t.size < 2:
            raise ValueError("a rate table needs two 1D arrays of equal length >= 2")
        if np.any(np.diff(t) <= 0.0) or np.any(t <= 0.0):
            raise ValueError("table temperatures must be positive and increasing")
        if np.any(k < 0.0) or not np.all(np.isfinite(k)):
            raise ValueError("table rates must be finite and >= 0")
        object.__setattr__(self, "temperatures", tuple(t.tolist()))
        object.__setattr__(self, "values", tuple(k.tolist()))

    def to_dict(self) -> dict[str, Any]:
        """Return the table."""
        return {
            "kind": self.kind,
            "depends_on": self.depends_on,
            "temperatures": list(self.temperatures),
            "values": list(self.values),
            "threshold": self.threshold,
        }


@method(
    equations={
        "constant": r"k = \mathrm{const}, \qquad \text{e.g. } k = g\,A_{ul} \text{ for a line with escape factor } g",
    },
    assumptions=(
        r"no temperature (`depends_on` is `none`): Einstein coefficients, wall losses, pumping",
    ),
)
@dataclass(frozen=True)
class Constant(RateModel):
    """A rate coefficient independent of temperature, e.g. an Einstein A in s^-1.

    Attributes:
        value: ``k`` in the SI unit of the reaction's order.
    """

    kind: ClassVar[str] = "constant"
    depends_on: str = "none"
    value: float = 0.0

    def __post_init__(self) -> None:
        """Check the value."""
        super().__post_init__()
        if not self.value >= 0.0:
            raise ValueError("a constant rate must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        """Return the value."""
        return {"kind": self.kind, "depends_on": self.depends_on, "value": self.value}


@method(
    equations={
        "function": r"k = f(T)",
        "tabulation": r"T_i = 10^{-2 + 6 i / 240}\ \mathrm{eV}, \quad i = 0, \dots, 240, \qquad \text{then as } \texttt{Tabulated}",
    },
    assumptions=(
        r"$f$ is host Python code; files and the JAX backend use its table",
        r"$k$ itself is returned, so the threshold scaling does not protect it: $k$ underflows to zero once $E_{\mathrm{th}}/T \gtrsim 708$, and so does an inverse of it",
    ),
    components=(Tabulated,),
)
@dataclass(frozen=True)
class Function(RateModel):
    """``k(T)`` from a Python function of the temperature in eV (array in, array out).

    Written to files as a table on `file_temperatures`.

    Attributes:
        function: ``k`` in SI as a function of ``T`` in eV.
        label: What the function is, e.g. ``"Kramers radiative recombination"``.
        threshold: As for `Tabulated`, used when the function is tabulated.
        file_temperatures: The temperatures (eV) of the table written to files.
    """

    kind: ClassVar[str] = "function"
    function: Callable[[np.ndarray], np.ndarray] | None = None
    label: str = ""
    threshold: float = unit_field("eV", 0.0)
    file_temperatures: tuple[float, ...] = unit_field(
        "eV", default_factory=lambda: tuple(np.geomspace(0.01, 1.0e4, 241).tolist())
    )

    def __post_init__(self) -> None:
        """Check that a function is given."""
        super().__post_init__()
        if not callable(self.function):
            raise TypeError("function must be callable")

    def tabulate(self) -> Tabulated:
        """Return the function as a `Tabulated` model on `file_temperatures`."""
        assert self.function is not None
        t = np.asarray(self.file_temperatures, dtype=float)
        k = np.asarray(self.function(t), dtype=float)
        return Tabulated(
            depends_on=self.depends_on,
            temperatures=tuple(t),
            values=tuple(k),
            threshold=self.threshold,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the function tabulated (see `tabulate`)."""
        data = self.tabulate().to_dict()
        data["label"] = self.label
        return data


@method(
    equations={
        "inverse": r"k_{\mathrm{inv}}(T) = k_{\mathrm{fwd}}(T)\,\frac{g_{\mathrm{in}}}{g_{\mathrm{out}}}\, e^{\Delta E/T}\,\big(\Lambda^3\big)^{\Delta e}, \qquad \Lambda^3 = \left(\frac{h^2}{2\pi m_e kT}\right)^{3/2}",
        "klein_rosseland": r"\sigma_{ul}(\varepsilon) = \frac{g_l}{g_u}\,\frac{\varepsilon + \Delta E}{\varepsilon}\,\sigma_{lu}(\varepsilon + \Delta E)",
    },
    assumptions=(
        r"$g_{\mathrm{in}}, g_{\mathrm{out}}$ the products of the weights of the forward reactants and products ($g_e = 2$), $\Delta E$ the forward energy change from the state energies",
        r"the same number of heavy particles on both sides of the forward reaction and $\Delta e \ge 0$ extra electrons among its products: Boltzmann ($\Delta e = 0$) or Saha ($\Delta e = 1$)",
        r"with a non-Maxwellian EEDF an electron-impact excitation is inverted by its Klein-Rosseland cross section; a Saha inverse uses the Maxwellian forward rate at $T_{\mathrm{eff}}$",
    ),
    references=(
        "[Klein and Rosseland, Z. Phys. 4, 46 (1921)](https://doi.org/10.1007/BF01328041)",
    ),
    components=(
        "opencr.kinetics.rates.RateEvaluator.coefficients",
        "opencr.kinetics.rates.inverse_cross_section",
    ),
)
@dataclass(frozen=True)
class DetailedBalance(RateModel):
    """The inverse of another reaction of the network, by detailed balance.

    De-excitation from excitation, three-body recombination from ionization,
    and any other inverse whose forward reaction has the products as
    reactants and an electron on both sides or one more on the forward
    product side. The evaluation (`opencr.kinetics.rates`) uses the
    state energies and weights: the Klein-Rosseland inverse cross section for
    an electron-impact excitation (valid for any energy distribution), and
    the Maxwellian relation (Boltzmann, Saha) otherwise.

    Attributes:
        forward: The name of the forward reaction.
    """

    kind: ClassVar[str] = "detailed_balance"
    forward: str = ""

    def __post_init__(self) -> None:
        """Check that the forward reaction is named."""
        super().__post_init__()
        if not self.forward:
            raise ValueError("DetailedBalance needs the name of the forward reaction")

    def to_dict(self) -> dict[str, Any]:
        """Return the forward reaction's name."""
        return {
            "kind": self.kind,
            "depends_on": self.depends_on,
            "forward": self.forward,
        }
