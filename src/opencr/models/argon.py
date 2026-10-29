"""Argon with lumped 4s levels, and a sputtered metal, for magnetron discharges.

``argon/metastable``: the global-model reaction set of Ashida, Lee and
Lieberman (1995) as collected by Lieberman and Lichtenberg (2005): the ground
state Ar, the metastables 4s[3/2]_2 and 4s'[1/2]_0 lumped into Ar(m), the
resonant 4s[3/2]_1 and 4s'[1/2]_1 lumped into Ar(r), and Ar+, with

* excitation, de-excitation, ionization and stepwise ionization by
  electrons (Arrhenius fits over 1-7 eV, the range of the source),
* metastable-resonant mixing by electrons and metastable pooling,
* the resonance lines of Ar(r) to the ground state, trappable through
  :class:`~opencr.core.conditions.RadiationTrapping`.

With ``metal="Cu"`` or ``"Ti"`` the sputtered atom M, M+ and M2+ are added:
electron-impact ionization of M and M+ from the Lotz formula
(:func:`plasmacoll.metal_ionization_cross_section`, a lower estimate) and
Penning ionization of M by Ar(m) at ``penning_rate``. The drivers are Ar and
(with a metal) M. The ions need losses to have a steady state: give them as
``wall_loss`` (s^-1 by state), or prescribe the ion densities. Metastable
pooling is quadratic in Ar(m); with ``pooling=False`` the network is linear in
the quasi-static states and :meth:`~opencr.model.Model.effective_rates` gives
the stepwise ionization through the metastables as effective rates.

This is the first rung of the magnetron programme's CR needs; the resolved
4s levels, Ar2+, the metal metastables and the hot-electron rates of the
ionization region model are not in it (see the guide "Argon and metals").
"""

from __future__ import annotations

from collections.abc import Mapping

from plasmacoll import metal_ionization_cross_section
from plasmacoll.metals import METAL_OUTER_SUBSHELLS

from opencr.core.network import ReactionNetwork
from opencr.core.rates import Arrhenius, Constant, CrossSectionRate
from opencr.core.reactions import Reaction
from opencr.core.species import ELECTRON_STATE, State, species_from_formula
from opencr.data.provenance import Provenance, Reference

__all__ = [
    "ARGON_ASSUMPTIONS",
    "ARGON_IONIZATION_ENERGY",
    "ARGON_PROVENANCE",
    "argon_network",
]

#: The ionization energy of Ar [eV] (NIST).
ARGON_IONIZATION_ENERGY = 15.7596
#: The lumped levels: label -> (energy in eV, weight).
_LEVELS = {"m": (11.6, 6.0), "r": (11.7, 6.0)}
#: The resonance lines 104.8 nm (A = 1.19e8 s^-1) and 106.7 nm (5.1e8 s^-1),
#: weighted equally in the lumped Ar(r) [s^-1].
_RESONANT_A = 0.5 * (1.19e8 + 5.1e8)

ARGON_ASSUMPTIONS = (
    "Lumped 4s levels: Ar(m) for 1s5 and 1s3, Ar(r) for 1s4 and 1s2; the 4p levels and "
    "higher are not resolved (their excitation is in the effective rates of the source).",
    "Arrhenius fits of a Maxwellian EEDF over about 1-7 eV; evaluated at the effective "
    "temperature for a non-Maxwellian EEDF.",
    "Ar2+, hot and warm argon groups and heavy-particle excitation are not included.",
    "Metal ionization from the simplified Lotz formula of the outer subshell (a lower "
    "estimate); the metal metastables are not resolved.",
    "Penning ionization of the metal by Ar(m) at a given rate coefficient; no charge "
    "exchange between Ar+ and the metal.",
)

ARGON_PROVENANCE = Provenance(
    Reference(
        "S. Ashida, C. Lee and M. A. Lieberman, Spatially averaged (global) model of "
        "time modulated high density argon plasmas, J. Vac. Sci. Technol. A 13, 2498 "
        "(1995); M. A. Lieberman and A. J. Lichtenberg, Principles of Plasma Discharges "
        "and Materials Processing, 2nd ed. (Wiley, 2005)",
        doi="10.1116/1.579494",
    ),
    units="m^3/s against eV",
    validity=(),
    uncertainty=0.3,
    uncertainty_note="fits over 1-7 eV of a Maxwellian; larger outside",
    licence="published-fit",
)
_NIST = Provenance(
    Reference(
        "NIST Atomic Spectra Database, Ar I resonance lines 104.8 and 106.7 nm",
        url="https://physics.nist.gov/asd",
    ),
    units="s^-1",
    uncertainty=0.1,
    licence="public-domain",
    notes="the two lines' A averaged for the lumped Ar(r)",
)
_LOTZ = Provenance(
    Reference(
        "W. Lotz, Z. Phys. 206, 205 (1967), simplified formula, through plasmacoll",
        doi="10.1007/BF01325928",
    ),
    units="m^2 against eV",
    uncertainty=0.5,
    uncertainty_note="outer subshell only: a lower estimate",
    licence="published-fit",
)
_PENNING = Provenance(
    Reference("given by the caller (penning_rate)"),
    units="m^3/s",
    uncertainty=1.0,
    uncertainty_note="the default is an order of magnitude; give a measured value",
    licence="public-domain",
)
_WALL = Provenance(
    Reference("given by the caller (wall_loss)"), units="s^-1", licence="public-domain"
)

#: (name, reactants, products, kind, a [m^3/s], b, c [eV]) of the argon set.
_ARGON_SET = (
    ("ion:0", ("e", "Ar"), ("e", "e", "Ar+"), "ionization", 2.34e-14, 0.59, 17.44),
    ("exc:0-m", ("e", "Ar"), ("e", "Ar(m)"), "excitation", 5.0e-15, 0.74, 11.56),
    ("exc:0-r", ("e", "Ar"), ("e", "Ar(r)"), "excitation", 1.9e-14, 0.33, 12.78),
    ("ion:m", ("e", "Ar(m)"), ("e", "e", "Ar+"), "ionization", 6.8e-15, 0.67, 4.20),
    ("ion:r", ("e", "Ar(r)"), ("e", "e", "Ar+"), "ionization", 6.8e-15, 0.67, 4.20),
    ("dex:m-0", ("e", "Ar(m)"), ("e", "Ar"), "deexcitation", 4.3e-16, 0.74, 0.0),
    ("mix:m-r", ("e", "Ar(m)"), ("e", "Ar(r)"), "excitation", 2.0e-13, 0.0, 0.0),
    ("mix:r-m", ("e", "Ar(r)"), ("e", "Ar(m)"), "deexcitation", 2.0e-13, 0.0, 0.0),
)


def argon_network(
    metal: str | None = None,
    penning_rate: float = 5.0e-16,
    pooling: bool = True,
    wall_loss: Mapping[str, float] | None = None,
    max_energy: float = 1.0e4,
) -> ReactionNetwork:
    """Return the network of ``argon/metastable``, with a sputtered metal if given.

    Args:
        metal: ``None``, ``"Cu"`` or ``"Ti"``.
        penning_rate: The rate coefficient of ``Ar(m) + M -> Ar + M+ + e`` in m^3/s.
        pooling: Include metastable pooling ``Ar(m) + Ar(m) -> Ar+ + Ar + e``.
        wall_loss: Loss frequencies in s^-1 by state name.
        max_energy: The last tabulated cross-section energy in eV.
    """
    argon = species_from_formula("Ar")
    states = {
        "Ar": State(argon, "", 0.0, 1.0),
        **{
            f"Ar({k})": State(argon, k, energy, weight, {"lumped": k})
            for k, (energy, weight) in _LEVELS.items()
        },
        "Ar+": State(species_from_formula("Ar+"), "", ARGON_IONIZATION_ENERGY, 6.0),
        "e": ELECTRON_STATE,
    }
    reactions = [
        Reaction(
            name,
            tuple(states[s] for s in reactants),
            tuple(states[s] for s in products),
            kind,
            Arrhenius(a=a, b=b, c=c),
            ARGON_PROVENANCE,
        )
        for name, reactants, products, kind, a, b, c in _ARGON_SET
    ]
    if pooling:
        reactions.append(
            Reaction(
                "pool:m-m",
                (states["Ar(m)"], states["Ar(m)"]),
                (states["Ar+"], states["Ar"], states["e"]),
                "penning_ionization",
                Constant(value=6.4e-16),
                ARGON_PROVENANCE,
            )
        )
    reactions += [
        Reaction(
            "A:r-0",
            (states["Ar(r)"],),
            (states["Ar"],),
            "spontaneous_emission",
            Constant(value=_RESONANT_A),
            _NIST,
        ),
    ]
    if metal is not None:
        if (metal, 0) not in METAL_OUTER_SUBSHELLS:
            raise ValueError(f"metal must be one of Cu, Ti, not {metal!r}")
        first = METAL_OUTER_SUBSHELLS[(metal, 0)][2]
        second = METAL_OUTER_SUBSHELLS[(metal, 1)][2]
        states[metal] = State(species_from_formula(metal), "", 0.0, 1.0)
        states[f"{metal}+"] = State(species_from_formula(f"{metal}+"), "", first, 1.0)
        states[f"{metal}++"] = State(
            species_from_formula(f"{metal}++"), "", first + second, 1.0
        )
        for charge, (low, high) in enumerate(
            ((metal, f"{metal}+"), (f"{metal}+", f"{metal}++"))
        ):
            reactions.append(
                Reaction(
                    f"ion:{low}",
                    (states["e"], states[low]),
                    (states["e"], states["e"], states[high]),
                    "ionization",
                    CrossSectionRate(
                        cross_section=metal_ionization_cross_section(
                            metal, charge, max_energy
                        )
                    ),
                    _LOTZ,
                )
            )
        reactions.append(
            Reaction(
                f"penning:{metal}",
                (states["Ar(m)"], states[metal]),
                (states["Ar"], states[f"{metal}+"], states["e"]),
                "penning_ionization",
                Constant(value=penning_rate),
                _PENNING,
            )
        )
    for name, frequency in (wall_loss or {}).items():
        if name not in states or name == "e":
            raise KeyError(f"wall_loss: unknown state {name!r}")
        reactions.append(
            Reaction(
                f"wall:{name}",
                (states[name],),
                (),
                "wall_loss",
                Constant(value=frequency),
                _WALL,
            )
        )
    heavy = [s for name, s in states.items() if name != "e"]
    return ReactionNetwork(
        heavy,
        reactions,
        name="argon/metastable" if metal is None else f"argon/metastable+{metal}",
        assumptions=ARGON_ASSUMPTIONS,
    )
