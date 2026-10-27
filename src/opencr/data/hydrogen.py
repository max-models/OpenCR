"""Atomic hydrogen data resolved in the principal quantum number ``n``.

* Levels: ``E_n = I_H (1 - 1/n^2)`` with the NIST ionization energy
  ``I_H = 13.598 eV`` (no fine structure, no Lamb shift), weights ``2 n^2``;
  the proton at ``I_H`` with weight 1.
* Spontaneous emission ``A(u -> l)`` from Johnson's (1972) absorption
  oscillator strengths (:func:`plasmacoll.atomic_hydrogen.oscillator_strength`,
  within 1% of the exact ``n``-bundled values), with
  ``A = 2 pi e^2 nu^2 / (eps0 m_e c^3) (g_l / g_u) f_lu``.
* Radiative recombination ``e + H+ -> H(n) + hv`` from Janev, Reiter and Samm
  (2003), eq. (21) for ``n = 1`` (1s) and ``n = 2`` (2s plus 2p), and the
  Kramers formula eq. (22) for ``n >= 3``.
* Electron-impact excitation and ionization cross sections:
  :mod:`plasmacoll.atomic_hydrogen` (the same report, section 2.1).

The ``l`` sub-states are assumed statistically populated (``n``-bundled), which
holds when ``l``-mixing collisions are fast; see the model's assumptions.
"""

from __future__ import annotations

import math

import numpy as np
from plasmacoll.atomic_hydrogen import oscillator_strength
from scipy.special import exp1

from opencr.constants import (
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    HYDROGEN_IONIZATION_ENERGY,
    PLANCK,
    SPEED_OF_LIGHT,
    VACUUM_PERMITTIVITY,
)
from opencr.core.species import State, species_from_formula
from opencr.data.provenance import Provenance, Reference, Validity

__all__ = [
    "EINSTEIN_A_PROVENANCE",
    "IMPACT_PROVENANCE",
    "JANEV_2003",
    "JOHNSON_1972",
    "LEVELS_PROVENANCE",
    "RADIATIVE_RECOMBINATION_PROVENANCE",
    "einstein_a",
    "hydrogen_levels",
    "level_energy",
    "radiative_recombination",
    "scaled_exp1",
]

JANEV_2003 = Reference(
    "R. K. Janev, D. Reiter and U. Samm, Collision Processes in Low-Temperature "
    "Hydrogen Plasmas, Forschungszentrum Juelich report Juel-4105 (2003)",
    url="https://eirene.de/Documentation/report_4105.pdf",
)
JOHNSON_1972 = Reference(
    "L. C. Johnson, Approximations for collisional and radiative transition rates "
    "in atomic hydrogen, Astrophys. J. 174, 227 (1972)",
    doi="10.1086/151486",
)
LEVELS_PROVENANCE = Provenance(
    Reference(
        "NIST Atomic Spectra Database, ionization energy of H I",
        url="https://physics.nist.gov/asd",
    ),
    retrieved="2026-10-27",
    units="eV",
    notes="E_n = I_H (1 - 1/n^2): no fine structure, no Lamb shift",
    licence="public-domain",
)
EINSTEIN_A_PROVENANCE = Provenance(
    JOHNSON_1972,
    units="s^-1",
    conversions=("A from f with A = 2 pi e^2 nu^2 / (eps0 m_e c^3) g_l / g_u f",),
    uncertainty=0.01,
    uncertainty_note="Johnson's f within about 1% of the exact n-bundled values",
    licence="published-fit",
)
RADIATIVE_RECOMBINATION_PROVENANCE = Provenance(
    Reference(JANEV_2003.citation, url=JANEV_2003.url, detail="eqs. (21) and (22)"),
    units="m^3/s against eV",
    conversions=("1e-14 cm^3/s -> 1e-20 m^3/s",),
    validity=(Validity("electron_temperature", 0.01, 1.0e4),),
    uncertainty=0.1,
    uncertainty_note="Kramers' formula without Gaunt factor for n >= 3",
    licence="published-fit",
)
IMPACT_PROVENANCE = Provenance(
    Reference(
        JANEV_2003.citation, url=JANEV_2003.url, detail="section 2.1, eqs. (4)-(16)"
    ),
    units="m^2 against eV (incident electron energy)",
    conversions=(
        "1e-16 cm^2 -> 1e-20 m^2",
        "tabulated by plasmacoll, linear interpolation",
    ),
    validity=(Validity("energy", 0.0, 1.0e4),),
    interpolation="linear between 241 points spaced logarithmically above threshold",
    extrapolation="the last value above 1e4 eV",
    uncertainty=0.2,
    uncertainty_note="10-20% from the ground state, 20-50% near threshold between excited states",
    licence="published-fit",
    notes="eq. (13a) misprints g2(1) = 0.0714; plasmacoll uses Johnson's 0.07014",
)

#: Radiative recombination to 1s, 2s and 2p, eq. (21): (A_nl, chi_nl).
_RR_EQ21 = {"1s": (3.92, 0.35), "2s": (2.47, 0.12), "2p": (6.22, 0.61)}
#: The Rydberg energy of eqs. (21) and (22) [eV].
_RY = 13.6


def level_energy(n: int) -> float:
    """Return the energy of H(n) above H(1s) in eV."""
    if int(n) != n or n < 1:
        raise ValueError(f"n must be an integer >= 1, not {n}")
    return HYDROGEN_IONIZATION_ENERGY * (1.0 - 1.0 / n**2)


def hydrogen_levels(n_max: int, isotope: str = "H") -> tuple[list[State], State]:
    """Return the states H(n=1..n_max) and the bare nucleus.

    Args:
        n_max: The highest principal quantum number.
        isotope: ``"H"`` or ``"D"``; the levels are the same (the reduced-mass
            shift of 0.03% is neglected).

    Returns:
        The atom's levels, labelled ``"n=<n>"``, and the ion's state.
    """
    if n_max < 1:
        raise ValueError("n_max must be >= 1")
    atom = species_from_formula(isotope)
    ion = species_from_formula(f"{isotope}+")
    levels = [
        State(atom, f"n={n}", level_energy(n), 2.0 * n**2, {"n": n})
        for n in range(1, n_max + 1)
    ]
    return levels, State(ion, "", HYDROGEN_IONIZATION_ENERGY, 1.0)


def einstein_a(upper: int, lower: int) -> float:
    """Return the spontaneous emission rate ``A(upper -> lower)`` of H in s^-1."""
    if not upper > lower >= 1:
        raise ValueError("need upper > lower >= 1")
    energy = level_energy(upper) - level_energy(lower)
    nu = energy * ELEMENTARY_CHARGE / PLANCK
    weights = (2.0 * lower**2) / (2.0 * upper**2)
    prefactor = (
        2.0
        * math.pi
        * ELEMENTARY_CHARGE**2
        * nu**2
        / (VACUUM_PERMITTIVITY * ELECTRON_MASS * SPEED_OF_LIGHT**3)
    )
    return prefactor * weights * oscillator_strength(lower, upper)


def scaled_exp1(x: np.ndarray) -> np.ndarray:
    """Return ``exp(x) E_1(x)`` without overflow (an asymptotic series above x = 50)."""
    x = np.asarray(x, dtype=float)
    small = np.minimum(x, 50.0)
    direct = np.exp(small) * exp1(small)
    big = np.maximum(x, 50.0)
    series = (1.0 - 1.0 / big + 2.0 / big**2 - 6.0 / big**3 + 24.0 / big**4) / big
    return np.where(x <= 50.0, direct, series)


def radiative_recombination(n: int, temperature: float | np.ndarray) -> np.ndarray:
    """Return the rate coefficient of ``e + H+ -> H(n) + hv`` in m^3/s.

    Eq. (21) of Janev, Reiter and Samm (2003) for ``n = 1`` and ``n = 2``
    (2s plus 2p), the Kramers formula eq. (22) for ``n >= 3``.

    Args:
        n: The principal quantum number of the captured level.
        temperature: ``kT_e`` in eV.
    """
    if int(n) != n or n < 1:
        raise ValueError(f"n must be an integer >= 1, not {n}")
    t = np.asarray(temperature, dtype=float)
    ionization = _RY / n**2
    beta = ionization / t
    if n <= 2:
        subshells = ("1s",) if n == 1 else ("2s", "2p")
        total = 0.0
        for key in subshells:
            a, chi = _RR_EQ21[key]
            total = total + a * math.sqrt(ionization / _RY) * beta**1.5 / (beta + chi)
        return 1.0e-20 * np.asarray(total)
    return 1.0e-20 * 5.201 * beta**1.5 * scaled_exp1(beta)
