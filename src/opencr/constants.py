"""Physical constants (CODATA 2018, through `scipy.constants`) and unit helpers.

OpenCR's units follow the max-models convention: SI everywhere, except
temperatures, given as ``kT`` in eV, and particle and level energies in eV.
Densities are in m^-3, rate coefficients of an order-``k`` reaction in
m^(3(k-1))/s (s^-1 for spontaneous emission, m^3/s for electron impact, m^6/s
for three-body recombination), wavelengths in nm (in vacuum unless a function
says otherwise).
"""

from __future__ import annotations

import math

import numpy as np
from methodcard import Rule, method
from scipy import constants as _c

__all__ = [
    "ATOMIC_MASS",
    "BOHR_RADIUS",
    "ELECTRON_MASS",
    "ELEMENTARY_CHARGE",
    "HYDROGEN_IONIZATION_ENERGY",
    "PLANCK",
    "PROTON_MASS",
    "SPEED_OF_LIGHT",
    "VACUUM_PERMITTIVITY",
    "air_wavelength",
    "photon_wavelength",
    "thermal_de_broglie_cubed",
]

ELEMENTARY_CHARGE = _c.e
ELECTRON_MASS = _c.m_e
PROTON_MASS = _c.m_p
ATOMIC_MASS = _c.atomic_mass
PLANCK = _c.h
SPEED_OF_LIGHT = _c.c
VACUUM_PERMITTIVITY = _c.epsilon_0
BOHR_RADIUS = _c.physical_constants["Bohr radius"][0]
#: The ionization energy of H(1s) in eV (NIST ASD).
HYDROGEN_IONIZATION_ENERGY = 13.598434599702

#: h c in eV nm.
_HC_EV_NM = PLANCK * SPEED_OF_LIGHT / ELEMENTARY_CHARGE * 1.0e9


def _wavelength_rules() -> dict:
    """The SymPy rule of `photon_wavelength` (for the tests)."""
    import sympy

    energy = sympy.Symbol("energy", positive=True)
    h, c, e = (sympy.Float(x) for x in (PLANCK, SPEED_OF_LIGHT, ELEMENTARY_CHARGE))
    rule = h * c / (energy * e) * sympy.Float(1e9)
    return {"wavelength": Rule(rule, ranges={"energy": (0.5, 20.0)})}


def _air_rules() -> dict:
    """The SymPy rule of `air_wavelength` (for the tests), above 200 nm."""
    import sympy

    lam = sympy.Symbol("vacuum_wavelength", positive=True)
    s2 = (1000 / lam) ** 2
    index = (
        1
        + sympy.Float("8.34254e-5")
        + sympy.Float("2.406147e-2") / (130 - s2)
        + sympy.Float("1.5998e-4") / (sympy.Float("38.9") - s2)
    )
    return {"edlen": Rule(lam / index, ranges={"vacuum_wavelength": (200.5, 2000.0)})}


def _de_broglie_rules() -> dict:
    """The SymPy rule of `thermal_de_broglie_cubed` (for the tests)."""
    import sympy

    t = sympy.Symbol("temperature", positive=True)
    h, m, e = (sympy.Float(x) for x in (PLANCK, ELECTRON_MASS, ELEMENTARY_CHARGE))
    rule = (h**2 / (2 * sympy.pi * m * t * e)) ** sympy.Rational(3, 2)
    return {"volume": Rule(rule, ranges={"temperature": (0.1, 100.0)})}


@method(
    equations={
        "wavelength": r"\lambda\,[\mathrm{nm}] = \frac{hc}{E} = \frac{1239.84\ldots}{E\,[\mathrm{eV}]}",
    },
    assumptions=(r"in vacuum",),
    references=(
        "[Tiesinga et al., Rev. Mod. Phys. 93, 025010 (2021)](https://doi.org/10.1103/RevModPhys.93.025010) (CODATA 2018)",
    ),
    symbolic=_wavelength_rules,
)
def photon_wavelength(energy: float | np.ndarray) -> np.ndarray:
    """Return the vacuum wavelength in nm of a photon of ``energy`` eV."""
    return _HC_EV_NM / np.asarray(energy, dtype=float)


@method(
    equations={
        "edlen": r"\lambda_{\mathrm{air}} = \frac{\lambda}{n}, \qquad n - 1 = 8.34254\times10^{-5} + \frac{2.406147\times10^{-2}}{130 - \sigma^2} + \frac{1.5998\times10^{-4}}{38.9 - \sigma^2}, \qquad \sigma = \frac{10^3}{\lambda\,[\mathrm{nm}]}\ \mu\mathrm{m}^{-1}",
    },
    assumptions=(
        r"standard air, the IAU standard used by NIST; above 200 nm only (shorter wavelengths are returned unchanged)",
    ),
    references=(
        "[Edlén, Metrologia 2, 71 (1966)](https://doi.org/10.1088/0026-1394/2/2/002)",
        "[Birch and Downs, Metrologia 31, 315 (1994)](https://doi.org/10.1088/0026-1394/31/4/006)",
    ),
    symbolic=_air_rules,
)
def air_wavelength(vacuum_wavelength: float | np.ndarray) -> np.ndarray:
    """Return the wavelength in standard air of a vacuum wavelength (both in nm).

    The refractive index of Edlén (1966) as revised by Birch and Downs (1994),
    the IAU standard used by NIST, applied above 200 nm; shorter wavelengths
    are returned unchanged (there is no air conversion in the vacuum UV).
    """
    vacuum = np.asarray(vacuum_wavelength, dtype=float)
    sigma2 = (1.0e3 / vacuum) ** 2  # (1/um)^2
    index = (
        1.0 + 8.34254e-5 + 2.406147e-2 / (130.0 - sigma2) + 1.5998e-4 / (38.9 - sigma2)
    )
    return np.where(vacuum > 200.0, vacuum / index, vacuum)


@method(
    equations={
        "volume": r"\Lambda^3 = \left(\frac{h^2}{2\pi m_e kT}\right)^{3/2}",
        "saha": r"\frac{n_e n_+}{n_0} = \frac{g_e g_+}{g_0}\,\frac{e^{-I/kT}}{\Lambda^3}",
    },
    assumptions=(r"$kT$ in eV, $\Lambda^3$ in m$^3$",),
    symbolic=_de_broglie_rules,
)
def thermal_de_broglie_cubed(temperature: float | np.ndarray) -> np.ndarray:
    """Return ``(h^2 / (2 pi m_e kT))^(3/2)`` in m^3, ``kT`` in eV.

    The electron's thermal de Broglie volume, the factor of the Saha equation
    ``n_e n_+ / n_0 = (g_e g_+ / g_0) exp(-I / kT) / lambda^3``.
    """
    kt = np.asarray(temperature, dtype=float) * ELEMENTARY_CHARGE
    return (PLANCK**2 / (2.0 * math.pi * ELECTRON_MASS * kt)) ** 1.5
