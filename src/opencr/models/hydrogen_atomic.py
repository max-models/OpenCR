"""Atomic hydrogen resolved in the principal quantum number.

``hydrogen/atomic/basic``: the levels H(n=1..n_max) and the proton, with

* electron-impact excitation ``n -> m`` and ionization of every level (Janev,
  Reiter and Samm 2003, section 2.1, and Johnson 1972, through
  :mod:`plasmacoll.atomic_hydrogen`);
* spontaneous emission ``m -> n`` (Johnson's oscillator strengths);
* electron-impact de-excitation and three-body recombination by detailed
  balance, and radiative recombination (Janev, Reiter and Samm, eqs. 21-22),
  each of which can be switched off.

Above n = 24, where Johnson's excitation formula fails, the excitations
involving a level ``n > 24`` use the Maxwellian rate coefficients of Vriens
and Smeets (1980) (:func:`plasmacoll.atomic_hydrogen.vriens_smeets_excitation_rate`),
so ``n_max`` can reach :data:`MAX_LEVEL`.

The drivers of the quasi-static solve are H(n=1) and H+.
"""

from __future__ import annotations

from dataclasses import replace

from plasmacoll.atomic_hydrogen import (
    JOHNSON_MAX_LEVEL,
    excitation_fit,
    ionization_fit,
    transition_energy,
    vriens_smeets_excitation_rate,
)

from opencr.core.network import ReactionNetwork
from opencr.core.rates import Constant, CrossSectionRate, DetailedBalance, Function
from opencr.core.reactions import Reaction
from opencr.core.species import ELECTRON_STATE
from opencr.data.hydrogen import (
    EINSTEIN_A_PROVENANCE,
    IMPACT_PROVENANCE,
    RADIATIVE_RECOMBINATION_PROVENANCE,
    VRIENS_SMEETS_PROVENANCE,
    einstein_a,
    hydrogen_levels,
    radiative_recombination,
)
from opencr.data.provenance import Reference

__all__ = ["ASSUMPTIONS", "MAX_LEVEL", "atomic_hydrogen_network"]

#: The highest ``n_max`` (the Vriens-Smeets rates hold for any n; the limit
#: keeps the number of reactions, ~3 n_max^2 / 2, manageable).
MAX_LEVEL = 60

#: The assumptions of the model, listed in its manifest.
ASSUMPTIONS = (
    "States resolved in the principal quantum number n only; the l sub-states are "
    "statistically populated (fast l-mixing), which overestimates the 2s population at "
    "low density.",
    "Levels above n_max are omitted (no continuum lowering, no extrapolation to "
    "n -> infinity); effective recombination at low density depends on n_max.",
    "Electron collisions only: heavy-particle (proton, atom) excitation, ionization and "
    "l-mixing of the excited states are omitted.",
    "Two-photon decay of 2s and fine structure are omitted.",
    "Optically thin unless escape factors are given in the conditions.",
    "De-excitation and three-body recombination follow from excitation and ionization "
    "by detailed balance (Klein-Rosseland for de-excitation with any EEDF; Saha at the "
    "electron temperature, the effective one for a non-Maxwellian EEDF, for three-body "
    "recombination).",
    "Cross sections from Janev, Reiter and Samm (2003) section 2.1 and Johnson (1972): "
    "10-20% from the ground state, 20-50% near threshold between excited states.",
    "Excitations involving a level above n = 24 use the Maxwellian rate coefficients of "
    "Vriens and Smeets (1980) (about 20%; up to a factor 2 below Johnson's near "
    "threshold), evaluated at the effective temperature for a non-Maxwellian EEDF.",
)


class _VriensSmeets:
    """``K_VS(n -> m, T)``: a picklable callable for :class:`~opencr.core.rates.Function`."""

    def __init__(self, n: int, m: int) -> None:
        self.n, self.m = n, m

    def __call__(self, temperature):
        return vriens_smeets_excitation_rate(self.n, self.m, temperature)


class _RadiativeRecombination:
    """``alpha_rr(n, T)``: a picklable callable for :class:`~opencr.core.rates.Function`."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __call__(self, temperature):
        return radiative_recombination(self.n, temperature)


def atomic_hydrogen_network(
    n_max: int = 20,
    isotope: str = "H",
    deexcitation: bool = True,
    three_body_recombination: bool = True,
    radiative_recombination_: bool = True,
    max_energy: float = 1.0e4,
) -> ReactionNetwork:
    """Return the network of ``hydrogen/atomic/basic``.

    Args:
        n_max: The highest principal quantum number, at most :data:`MAX_LEVEL`;
            above 24 the Vriens-Smeets rates take over from Johnson's formula.
        isotope: ``"H"`` or ``"D"`` (the same data; the species are renamed).
        deexcitation: Include electron-impact de-excitation.
        three_body_recombination: Include three-body recombination.
        radiative_recombination_: Include radiative recombination.
        max_energy: The last tabulated cross-section energy in eV.
    """
    if not 2 <= n_max <= MAX_LEVEL:
        raise ValueError(f"n_max must be in [2, {MAX_LEVEL}]")
    levels, ion = hydrogen_levels(n_max, isotope)
    e = ELECTRON_STATE
    reactions: list[Reaction] = []
    for lower in range(1, n_max + 1):
        for upper in range(lower + 1, n_max + 1):
            name = f"exc:{lower}-{upper}"
            low, up = levels[lower - 1], levels[upper - 1]
            if upper > JOHNSON_MAX_LEVEL:
                rate = Function(
                    function=_VriensSmeets(lower, upper),
                    label=f"Vriens-Smeets excitation {lower} -> {upper}",
                    threshold=transition_energy(lower, upper),
                )
                provenance = VRIENS_SMEETS_PROVENANCE
            else:
                fit = excitation_fit(lower, upper)
                rate = CrossSectionRate(cross_section=fit.cross_section(max_energy))
                provenance = replace(
                    IMPACT_PROVENANCE,
                    reference=Reference(
                        IMPACT_PROVENANCE.reference.citation,
                        url=IMPACT_PROVENANCE.reference.url,
                        detail=fit.source,
                    ),
                )
            reactions.append(
                Reaction(name, (e, low), (e, up), "excitation", rate, provenance)
            )
            if deexcitation:
                reactions.append(
                    Reaction(
                        f"dex:{upper}-{lower}",
                        (e, up),
                        (e, low),
                        "deexcitation",
                        DetailedBalance(forward=name),
                    )
                )
            reactions.append(
                Reaction(
                    f"A:{upper}-{lower}",
                    (up,),
                    (low,),
                    "spontaneous_emission",
                    Constant(value=einstein_a(upper, lower)),
                    EINSTEIN_A_PROVENANCE,
                )
            )
    for n, level in enumerate(levels, start=1):
        fit = ionization_fit(n)
        provenance = replace(
            IMPACT_PROVENANCE,
            reference=Reference(
                IMPACT_PROVENANCE.reference.citation,
                url=IMPACT_PROVENANCE.reference.url,
                detail=fit.source,
            ),
        )
        reactions.append(
            Reaction(
                f"ion:{n}",
                (e, level),
                (e, e, ion),
                "ionization",
                CrossSectionRate(cross_section=fit.cross_section(max_energy)),
                provenance,
            )
        )
        if three_body_recombination:
            reactions.append(
                Reaction(
                    f"rec3:{n}",
                    (e, e, ion),
                    (e, level),
                    "three_body_recombination",
                    DetailedBalance(forward=f"ion:{n}"),
                )
            )
        if radiative_recombination_:
            reactions.append(
                Reaction(
                    f"rr:{n}",
                    (e, ion),
                    (level,),
                    "radiative_recombination",
                    Function(
                        function=_RadiativeRecombination(n),
                        label=f"radiative recombination to n={n}",
                    ),
                    RADIATIVE_RECOMBINATION_PROVENANCE,
                )
            )
    return ReactionNetwork(
        [*levels, ion],
        reactions,
        name="hydrogen/atomic/basic"
        if isotope == "H"
        else f"{isotope.lower()}/atomic/basic",
        assumptions=ASSUMPTIONS,
    )
