"""Electron energy distribution functions.

An EEDF is given as the energy probability function ``F0(eps)`` in eV^-3/2,
normalised by ``int eps^1/2 F0 deps = 1`` (the convention of BOLSIG+ and of
`plasmacoll.solve_eedf`), so that ``eps^1/2 F0`` is the probability
density of the energy. Rate coefficients are
``k = (2 e / m_e)^1/2 int eps sigma F0 deps``.

* `Maxwellian`: ``F0 = 2 / sqrt(pi) T^-3/2 exp(-eps / T)``.
* `Generalized`: ``F0 ~ exp(-(eps / eps0)^x)``, Maxwellian for ``x = 1``
  and Druyvesteyn for ``x = 2``, at a given mean energy.
* `TabulatedEEDF`: any distribution on an energy grid, for example from
  a probe or from plasmacoll's two-term Boltzmann solver
  (`TabulatedEEDF.from_boltzmann`).

OpenCR does not compute EEDFs; they are inputs (see the guide "The CR model").
Their energies and temperatures take an astropy Quantity or a string such as
``"3 eV"`` or ``"30000 K"`` as well as a number in eV (see `plasmacoll.units`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from methodcard import Rule, method
from plasmacoll.units import convert_units, to_array, unit_field
from scipy.special import gamma

__all__ = ["EEDF", "Generalized", "Maxwellian", "TabulatedEEDF"]


def _maxwellian_rules() -> dict:
    """The SymPy rule of `Maxwellian` (for the tests)."""
    import sympy

    energy, t = sympy.symbols("energy temperature", positive=True)
    rule = (
        2 / sympy.sqrt(sympy.pi) * t ** sympy.Rational(-3, 2) * sympy.exp(-energy / t)
    )

    def eepf(energy: float, temperature: float) -> float:
        return float(Maxwellian(temperature).eepf(np.array(energy)))

    ranges = {"energy": (0.0, 50.0), "temperature": (0.5, 20.0)}
    return {"eepf": Rule(rule, implementation=eepf, ranges=ranges)}


def _generalized_rules() -> dict:
    """The SymPy rule of `Generalized` (for the tests)."""
    import sympy

    energy, mean, x = sympy.symbols("energy mean shape", positive=True)
    scale = mean * sympy.gamma(3 / (2 * x)) / sympy.gamma(5 / (2 * x))
    c = x / (scale ** sympy.Rational(3, 2) * sympy.gamma(3 / (2 * x)))
    rule = c * sympy.exp(-((energy / scale) ** x))

    def eepf(energy: float, mean: float, shape: float) -> float:
        return float(Generalized(mean, shape).eepf(np.array(energy)))

    ranges = {"energy": (0.0, 30.0), "mean": (1.0, 10.0), "shape": (1.0, 2.5)}
    return {"eepf": Rule(rule, implementation=eepf, ranges=ranges, rtol=1e-9)}


@method(
    equations={
        "normalisation": r"\int_0^\infty \varepsilon^{1/2} F_0(\varepsilon)\, d\varepsilon = 1, \qquad F_0 \ \text{in eV}^{-3/2}",
        "mean": r"\langle\varepsilon\rangle = \int_0^\infty \varepsilon^{3/2} F_0\, d\varepsilon, \qquad T_{\mathrm{eff}} = \tfrac23 \langle\varepsilon\rangle",
        "rate": r"k = \sqrt{\frac{2e}{m_e}} \int_0^\infty \varepsilon\,\sigma(\varepsilon)\,F_0(\varepsilon)\,d\varepsilon",
        "grid": r"\varepsilon \in \{0\} \cup \mathrm{geomspace}\big(10^{-4}\varepsilon_{\max},\ \varepsilon_{\max}\big), \qquad \varepsilon_{\max} = 60\, T_{\mathrm{eff}}",
    },
    assumptions=(
        r"the EEDF is an input: OpenCR does not compute it",
        r"the rate integral is `plasmacoll.eedf_rate_coefficient`: trapezoid on the union of the EEDF grid, the table energies and the threshold, each interval refined 4 times, $\ln F_0$ linear between points, $F_0 = 0$ beyond the grid",
        r"only the electron cross-section rates (and their Klein-Rosseland inverses) see a non-Maxwellian EEDF; every other rate uses $T_{\mathrm{eff}}$",
    ),
    references=(
        "[Hagelaar and Pitchford, Plasma Sources Sci. Technol. 14, 722 (2005)](https://doi.org/10.1088/0963-0252/14/4/011)",
    ),
)
class EEDF:
    """Base class: a normalised electron energy probability function."""

    def eepf(self, energy: np.ndarray) -> np.ndarray:
        """Return ``F0`` in eV^-3/2 at ``energy`` (eV)."""
        raise NotImplementedError

    @property
    def mean_energy(self) -> float:
        """``<eps> = int eps^3/2 F0 deps`` in eV."""
        raise NotImplementedError

    @property
    def effective_temperature(self) -> float:
        """``2/3 <eps>`` in eV, the temperature of a Maxwellian of the same mean energy."""
        return 2.0 * self.mean_energy / 3.0

    @property
    def is_maxwellian(self) -> bool:
        """Whether the distribution is exactly Maxwellian."""
        return False

    def grid(self, num: int = 2000) -> np.ndarray:
        """Return an energy grid (eV) that resolves the distribution and its tail."""
        top = 60.0 * self.effective_temperature
        return np.concatenate([[0.0], np.geomspace(1.0e-4 * top, top, num - 1)])

    def tabulate(self, num: int = 2000) -> TabulatedEEDF:
        """Return the distribution on `grid`."""
        energy = self.grid(num)
        return TabulatedEEDF(energy, self.eepf(energy))


@method(
    equations={
        "eepf": r"F_0(\varepsilon) = \frac{2}{\sqrt\pi}\, T^{-3/2}\, e^{-\varepsilon/T}",
        "mean": r"\langle\varepsilon\rangle = \tfrac32 T",
    },
    assumptions=(
        r"detected by `is_maxwellian`: the rates then take the closed-form Maxwellian path",
    ),
    symbolic=_maxwellian_rules,
)
@dataclass(frozen=True, eq=False)
class Maxwellian(EEDF):
    """A Maxwellian at ``temperature`` (``kT`` in eV, or a quantity)."""

    temperature: float = unit_field("eV")

    def __post_init__(self) -> None:
        """Convert the temperature to eV and check it."""
        convert_units(self)
        if not self.temperature > 0.0:
            raise ValueError("temperature must be > 0")

    def eepf(self, energy: np.ndarray) -> np.ndarray:
        """Return ``2 / sqrt(pi) T^-3/2 exp(-eps / T)``."""
        t = self.temperature
        return (
            2.0
            / math.sqrt(math.pi)
            * t**-1.5
            * np.exp(-np.asarray(energy, dtype=float) / t)
        )

    @property
    def mean_energy(self) -> float:
        """``3/2 T``."""
        return 1.5 * self.temperature

    @property
    def is_maxwellian(self) -> bool:
        """True."""
        return True

    def __repr__(self) -> str:
        """The temperature."""
        return f"Maxwellian(temperature={self.temperature:g})"


@method(
    equations={
        "eepf": r"F_0(\varepsilon) = c\, e^{-(\varepsilon/\varepsilon_0)^x}, \qquad \varepsilon_0 = \langle\varepsilon\rangle\,\frac{\Gamma(3/2x)}{\Gamma(5/2x)}, \qquad c = \frac{x}{\varepsilon_0^{3/2}\,\Gamma(3/2x)}",
        "grid": r"\varepsilon_{\max} = \varepsilon_0\, 60^{1/x}",
    },
    assumptions=(
        r"$x = 1$ Maxwellian, $x = 2$ Druyvesteyn (elastic collisions with a constant cross section in a field)",
        r"normalised with the mean $\langle\varepsilon\rangle$ given",
    ),
    symbolic=_generalized_rules,
)
@dataclass(frozen=True, eq=False)
class Generalized(EEDF):
    """``F0 = c exp(-(eps / eps0)^shape)`` with the given mean energy.

    ``shape = 1`` is a Maxwellian, ``shape = 2`` a Druyvesteyn distribution
    (elastic collisions with a constant cross section in a field). With
    ``eps0 = <eps> Gamma(3/(2x)) / Gamma(5/(2x))`` and
    ``c = x / (eps0^3/2 Gamma(3/(2x)))``. The mean energy is in eV (or a
    quantity).
    """

    mean: float = unit_field("eV")
    shape: float = 2.0

    def __post_init__(self) -> None:
        """Convert the mean energy to eV and check the parameters."""
        convert_units(self)
        if not self.mean > 0.0 or not self.shape > 0.0:
            raise ValueError("mean energy and shape must be > 0")

    @property
    def _scale(self) -> float:
        x = self.shape
        return self.mean * gamma(1.5 / x) / gamma(2.5 / x)

    def eepf(self, energy: np.ndarray) -> np.ndarray:
        """Return ``F0`` at ``energy``."""
        x = self.shape
        e0 = self._scale
        c = x / (e0**1.5 * gamma(1.5 / x))
        return c * np.exp(-((np.asarray(energy, dtype=float) / e0) ** x))

    @property
    def mean_energy(self) -> float:
        """The mean energy."""
        return self.mean

    @property
    def is_maxwellian(self) -> bool:
        """Whether ``shape`` is 1."""
        return self.shape == 1.0

    def grid(self, num: int = 2000) -> np.ndarray:
        """A grid to where ``(eps/eps0)^x`` reaches 60."""
        top = self._scale * 60.0 ** (1.0 / self.shape)
        return np.concatenate([[0.0], np.geomspace(1.0e-4 * top, top, num - 1)])

    def __repr__(self) -> str:
        """The mean energy and the shape."""
        return f"Generalized(mean={self.mean:g}, shape={self.shape:g})"


@method(
    equations={
        "normalisation": r"F_0 \leftarrow \frac{F_0}{\sum_i \tfrac12 \big(\varepsilon_i^{1/2} F_{0,i} + \varepsilon_{i+1}^{1/2} F_{0,i+1}\big)(\varepsilon_{i+1} - \varepsilon_i)}",
        "interpolation": r"\ln F_0 \ \text{linear between the points} \ (F_0 \text{ itself next to a zero}), \qquad F_0 = 0 \ \text{beyond the last}",
        "mean": r"\langle\varepsilon\rangle = \sum_i \tfrac12 \big(\varepsilon_i^{3/2} F_{0,i} + \varepsilon_{i+1}^{3/2} F_{0,i+1}\big)(\varepsilon_{i+1} - \varepsilon_i)",
    },
    assumptions=(
        r"any distribution on an energy grid: a probe, or plasmacoll's two-term Boltzmann solver",
        r"normalised by the trapezoidal rule on construction",
    ),
)
class TabulatedEEDF(EEDF):
    """A distribution tabulated on an energy grid, normalised on construction.

    Between the points ``log F0`` is interpolated linearly (``F0`` where it is
    zero); beyond the last point ``F0`` is zero.

    Args:
        energy: The grid in eV (or a quantity), increasing, ``>= 0``.
        eepf: ``F0`` at ``energy``, non-negative, in any normalisation.
    """

    def __init__(self, energy: np.ndarray, eepf: np.ndarray) -> None:
        """Check, normalise and store the table."""
        energy = np.asarray(to_array(energy, "eV", "energy"), dtype=float)
        eepf = np.asarray(eepf, dtype=float)
        if energy.ndim != 1 or energy.shape != eepf.shape or energy.size < 2:
            raise ValueError("energy and eepf must be 1D arrays of equal length >= 2")
        if np.any(np.diff(energy) <= 0.0) or energy[0] < 0.0:
            raise ValueError("energy must be increasing and >= 0")
        if np.any(eepf < 0.0) or not np.all(np.isfinite(eepf)):
            raise ValueError("eepf must be finite and >= 0")
        norm = np.trapezoid(np.sqrt(energy) * eepf, energy)
        if not norm > 0.0:
            raise ValueError("eepf is zero everywhere")
        self.energy = energy
        self.values = eepf / norm

    @classmethod
    def from_boltzmann(cls, solution: Any) -> TabulatedEEDF:
        """Return the distribution of a `plasmacoll.BoltzmannSolution`."""
        return cls(np.asarray(solution.energy), np.asarray(solution.eepf))

    def eepf(self, energy: np.ndarray) -> np.ndarray:
        """Return ``F0`` at ``energy``, interpolated."""
        energy = np.asarray(energy, dtype=float)
        tiny = np.finfo(float).tiny
        logged = np.exp(
            np.interp(
                energy,
                self.energy,
                np.log(np.maximum(self.values, tiny)),
                right=-np.inf,
            )
        )
        positive = np.interp(
            energy, self.energy, (self.values > 0.0).astype(float), right=0.0
        )
        linear = np.interp(energy, self.energy, self.values, right=0.0)
        return np.where(positive >= 1.0, logged, linear)

    @property
    def mean_energy(self) -> float:
        """``int eps^3/2 F0 deps`` on the table."""
        return float(np.trapezoid(self.energy**1.5 * self.values, self.energy))

    def grid(self, num: int = 2000) -> np.ndarray:
        """The table's own grid."""
        return self.energy

    def tabulate(self, num: int = 2000) -> TabulatedEEDF:
        """Itself."""
        return self

    def __repr__(self) -> str:
        """The size and the mean energy."""
        return (
            f"TabulatedEEDF({self.energy.size} points, mean={self.mean_energy:.4g} eV)"
        )
