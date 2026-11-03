"""Atomic hydrogen data resolved in the principal quantum number ``n``.

* Levels: ``E_n = I_H (1 - 1/n^2)`` with the NIST ionization energy
  ``I_H = 13.598 eV`` (no fine structure, no Lamb shift), weights ``2 n^2``;
  the proton at ``I_H`` with weight 1.
* Spontaneous emission ``A(u -> l)`` from Johnson's (1972) absorption
  oscillator strengths (`plasmacoll.atomic_hydrogen.oscillator_strength`,
  within 1% of the exact ``n``-bundled values), with
  ``A = 2 pi e^2 nu^2 / (eps0 m_e c^3) (g_l / g_u) f_lu``.
* Radiative recombination ``e + H+ -> H(n) + hv`` from Janev, Reiter and Samm
  (2003), eq. (21) for ``n = 1`` (1s) and ``n = 2`` (2s plus 2p), and the
  Kramers formula eq. (22) for ``n >= 3``.
* Electron-impact excitation and ionization cross sections:
  `plasmacoll.atomic_hydrogen` (the same report, section 2.1).

The ``l`` sub-states are assumed statistically populated (``n``-bundled), which
holds when ``l``-mixing collisions are fast; see the model's assumptions.
"""

from __future__ import annotations

import math

import numpy as np
from methodcard import Rule, method
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
    "VRIENS_SMEETS_PROVENANCE",
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
VRIENS_SMEETS_PROVENANCE = Provenance(
    Reference(
        "L. Vriens and A. H. M. Smeets, Cross-section and rate formulas for "
        "electron-impact ionization, excitation, deexcitation, and total depopulation "
        "of excited atoms, Phys. Rev. A 22, 940 (1980)",
        doi="10.1103/PhysRevA.22.940",
        detail="eq. (17)",
    ),
    units="m^3/s against eV",
    conversions=("1 cm^3/s -> 1e-6 m^3/s",),
    validity=(Validity("electron_temperature", 0.01, 1.0e4),),
    uncertainty=0.2,
    uncertainty_note="about 20% for n >= 3; Maxwellian electrons only",
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


def _level_rules() -> dict:
    """The SymPy rule of `level_energy` (for the tests); ``n`` is rounded down."""
    import sympy

    n = sympy.Symbol("n", positive=True)
    rule = sympy.Float(HYDROGEN_IONIZATION_ENERGY) * (1 - 1 / sympy.floor(n) ** 2)
    return {
        "level": Rule(
            rule,
            implementation=lambda n: level_energy(math.floor(n)),
            ranges={"n": (1.0, 60.0)},
        )
    }


def _einstein_rules() -> dict:
    """The SymPy rule of `einstein_a` (for the tests): the Balmer lines, ``m -> 2``.

    With Johnson's oscillator strengths and Gaunt coefficients of ``n = 2``.
    """
    import sympy

    upper = sympy.Symbol("upper", positive=True)
    m, n = sympy.floor(upper), 2
    g0, g1, g2 = (sympy.Float(x) for x in ("1.0785", "-0.2319", "0.02947"))
    y = 1 - sympy.Rational(n) ** 2 / m**2
    f = 32 / (3 * sympy.sqrt(3) * sympy.pi) * n / (m**3 * y**3)
    f *= g0 + g1 / y + g2 / y**2
    e, h, c, eps0, me = (
        sympy.Float(x)
        for x in (
            ELEMENTARY_CHARGE,
            PLANCK,
            SPEED_OF_LIGHT,
            VACUUM_PERMITTIVITY,
            ELECTRON_MASS,
        )
    )
    nu = sympy.Float(HYDROGEN_IONIZATION_ENERGY) * (sympy.Rational(1, n**2) - 1 / m**2)
    nu *= e / h
    rule = 2 * sympy.pi * e**2 * nu**2 / (eps0 * me * c**3) * n**2 / m**2 * f
    return {
        "einstein": Rule(
            rule,
            implementation=lambda upper: einstein_a(math.floor(upper), 2),
            ranges={"upper": (3.0, 30.0)},
        )
    }


def _scaled_exp1_rules() -> dict:
    """The SymPy rule of `scaled_exp1` (for the tests), through the asymptotic branch."""
    import sympy

    x = sympy.Symbol("x", positive=True)
    return {
        "scaled": Rule(
            sympy.exp(x) * sympy.expint(1, x), ranges={"x": (0.01, 200.0)}, rtol=1e-6
        )
    }


def _recombination_rules() -> dict:
    """The SymPy rules of `radiative_recombination` (for the tests): n = 1 and n = 5."""
    import sympy

    t = sympy.Symbol("temperature", positive=True)
    beta1 = sympy.Float(_RY) / t
    a, chi = (sympy.Float(x) for x in _RR_EQ21["1s"])
    ground = sympy.Float(1e-20) * a * beta1 ** sympy.Rational(3, 2) / (beta1 + chi)
    beta5 = sympy.Float(_RY) / 25 / t
    kramers = sympy.Float("5.201e-20") * beta5 ** sympy.Rational(3, 2)
    kramers *= sympy.exp(beta5) * sympy.expint(1, beta5)
    ranges = {"temperature": (0.05, 100.0)}
    return {
        "ground": Rule(
            ground,
            implementation=lambda temperature: radiative_recombination(1, temperature),
            ranges=ranges,
        ),
        "kramers": Rule(
            kramers,
            implementation=lambda temperature: radiative_recombination(5, temperature),
            ranges=ranges,
            rtol=1e-9,
        ),
    }


@method(
    equations={
        "level": r"E_n = I_H\left(1 - \frac{1}{n^2}\right), \qquad I_H = 13.598434599702\ \mathrm{eV}",
    },
    assumptions=(r"no fine structure, no Lamb shift; energies above H(1s)",),
    references=("[NIST Atomic Spectra Database](https://physics.nist.gov/asd)",),
    symbolic=_level_rules,
)
def level_energy(n: int) -> float:
    """Return the energy of H(n) above H(1s) in eV."""
    if int(n) != n or n < 1:
        raise ValueError(f"n must be an integer >= 1, not {n}")
    return HYDROGEN_IONIZATION_ENERGY * (1.0 - 1.0 / n**2)


@method(
    equations={
        "states": r"\mathrm{H}(n):\ E_n,\ g_n = 2n^2, \quad n = 1, \dots, n_{\max}; \qquad \mathrm{H}^+:\ E = I_H,\ g = 1",
    },
    assumptions=(
        r"$n$-bundled: the $l$ sub-states statistically populated",
        r"`isotope` `D` renames the species with the same data (the 0.03% reduced-mass shift is neglected)",
    ),
    components=(level_energy,),
)
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


@method(
    equations={
        "einstein": r"A_{mn} = \frac{2\pi e^2\nu^2}{\varepsilon_0 m_e c^3}\,\frac{g_n}{g_m}\,f_{nm}, \qquad \nu = \frac{(E_m - E_n)\,e}{h}",
        "oscillator": r"f_{nm} = \frac{32}{3\sqrt3\,\pi}\,\frac{n}{m^3 y^3}\,\big(g_0 + g_1/y + g_2/y^2\big), \qquad y = 1 - (n/m)^2",
    },
    assumptions=(
        r"Johnson's oscillator strengths and Gaunt coefficients (`plasmacoll.atomic_hydrogen`), within about 1% of the exact $n$-bundled values",
        r"$\nu$ from the NIST-based state energies",
    ),
    references=(
        "[Johnson, Astrophys. J. 174, 227 (1972)](https://doi.org/10.1086/151486), eqs. (11)-(13)",
        "[Wiese and Fuhr, J. Phys. Chem. Ref. Data 38, 565 (2009)](https://doi.org/10.1063/1.3077727) (the tests)",
    ),
    symbolic=_einstein_rules,
)
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


@method(
    equations={
        "scaled": r"e^{x}E_1(x)",
        "asymptotic": r"e^{x}E_1(x) \approx \frac{1}{x}\left(1 - \frac{1}{x} + \frac{2}{x^2} - \frac{6}{x^3} + \frac{24}{x^4}\right), \quad x > 50",
    },
    assumptions=(
        r"SciPy's `exp1` up to $x = 50$; continuous to $10^{-6}$ at the switch",
    ),
    references=(
        "[Virtanen et al., Nat. Methods 17, 261 (2020)](https://doi.org/10.1038/s41592-019-0686-2) (`exp1`)",
    ),
    symbolic=_scaled_exp1_rules,
)
def scaled_exp1(x: np.ndarray) -> np.ndarray:
    """Return ``exp(x) E_1(x)`` without overflow (an asymptotic series above x = 50)."""
    x = np.asarray(x, dtype=float)
    small = np.minimum(x, 50.0)
    direct = np.exp(small) * exp1(small)
    big = np.maximum(x, 50.0)
    series = (1.0 - 1.0 / big + 2.0 / big**2 - 6.0 / big**3 + 24.0 / big**4) / big
    return np.where(x <= 50.0, direct, series)


@method(
    equations={
        "ground": r"\alpha_{nl} = 10^{-20} A_{nl}\sqrt{\frac{I_n}{\mathrm{Ry}}}\;\frac{\beta^{3/2}}{\beta + \chi_{nl}}\ \mathrm{m^3/s}, \qquad \beta = \frac{I_n}{T}, \quad (A, \chi)_{1s} = (3.92, 0.35),\ (A, \chi)_{2s} = (2.47, 0.12),\ (A, \chi)_{2p} = (6.22, 0.61)",
        "kramers": r"\alpha_n = 5.201\times10^{-20}\,\beta^{3/2}\,e^{\beta}E_1(\beta)\ \mathrm{m^3/s}, \qquad n \ge 3",
    },
    assumptions=(
        r"$I_n = \mathrm{Ry}/n^2$ with $\mathrm{Ry} = 13.6$ eV; `rr:2` is the sum of 2s and 2p",
        r"Kramers without Gaunt factor for $n \ge 3$; recorded range 0.01-$10^4$ eV, evaluated at any $T$",
    ),
    references=(
        "[Janev, Reiter and Samm, Jül-4105 (2003)](https://eirene.de/Documentation/report_4105.pdf), eqs. (21) and (22)",
    ),
    components=(scaled_exp1,),
    symbolic=_recombination_rules,
)
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
