"""Physical constants (CODATA 2018, through :mod:`scipy.constants`) and unit helpers.

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


def photon_wavelength(energy: float | np.ndarray) -> np.ndarray:
    """Return the vacuum wavelength in nm of a photon of ``energy`` eV."""
    return _HC_EV_NM / np.asarray(energy, dtype=float)


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


def thermal_de_broglie_cubed(temperature: float | np.ndarray) -> np.ndarray:
    """Return ``(h^2 / (2 pi m_e kT))^(3/2)`` in m^3, ``kT`` in eV.

    The electron's thermal de Broglie volume, the factor of the Saha equation
    ``n_e n_+ / n_0 = (g_e g_+ / g_0) exp(-I / kT) / lambda^3``.
    """
    kt = np.asarray(temperature, dtype=float) * ELEMENTARY_CHARGE
    return (PLANCK**2 / (2.0 * math.pi * ELECTRON_MASS * kt)) ** 1.5
