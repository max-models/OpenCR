"""Molecular hydrogen data: the states of H2, H2+, H3+ and H-, and their radiative data.

* H2(X, v=0..14): the vibrational term energies of Fantz and Wuenderlich
  (2006) (:data:`plasmacoll.vibrational.H2_GROUND_STATE_VIBRATIONAL_ENERGY_EV`),
  rotations summed (weight 1 for every ``v``).
* The excited electronic states B, C, EF, a and d at the vertical excitation
  energies of the cross-section fits of Janev, Reiter and Samm (2003)
  (:data:`plasmacoll.janev.JANEV_FITS`), with their electronic weights.
* Radiative lifetimes of the ``v' = 0`` levels (Fantz and Wuenderlich 2006;
  Astashkevich and Lavrov 2015), rounded: B and C decay to X (Lyman and
  Werner bands), EF to B, a to the repulsive b (the continuum), d to a (the
  Fulcher band).

Energies share the scale of :mod:`opencr.core.species` with atomic hydrogen:
zero at two H(1s) atoms per H2 (so H2(X, v=0) is at minus its dissociation
energy ``D_0 = 4.478 eV``), H+ at 13.598 eV, H2+ at ``I(H2) - D_0`` and H3+
below H2 + H+ by the proton affinity of H2.
"""

from __future__ import annotations

from plasmacoll.janev import H_MINUS_BINDING_ENERGY, JANEV_FITS
from plasmacoll.vibrational import H2_GROUND_STATE_VIBRATIONAL_ENERGY_EV

from opencr.constants import HYDROGEN_IONIZATION_ENERGY
from opencr.core.species import State, species_from_formula
from opencr.data.provenance import Provenance, Reference

__all__ = [
    "ELECTRONIC_STATES",
    "H2_DISSOCIATION_ENERGY",
    "H2_IONIZATION_ENERGY",
    "H2_PROTON_AFFINITY",
    "LIFETIME_PROVENANCE",
    "NUM_VIBRATIONAL_LEVELS",
    "RADIATIVE_DECAYS",
    "ion_states",
    "molecular_states",
    "vibrational_energy",
]

#: The dissociation energy ``D_0`` of H2(X, v=0) [eV].
H2_DISSOCIATION_ENERGY = 4.478
#: The adiabatic ionization energy of H2 [eV] (Janev, Reiter and Samm, eq. (106)).
H2_IONIZATION_ENERGY = 15.42
#: The proton affinity of H2, H3+ -> H2 + H+ [eV].
H2_PROTON_AFFINITY = 4.38
#: The bound vibrational levels of H2(X), v = 0..14.
NUM_VIBRATIONAL_LEVELS = len(H2_GROUND_STATE_VIBRATIONAL_ENERGY_EV)

#: The resolved electronic states: label -> (key of the excitation fit, weight).
ELECTRONIC_STATES: dict[str, tuple[str, float]] = {
    "B": ("e_h2_singlet_B", 1.0),
    "C": ("e_h2_singlet_C", 2.0),
    "EF": ("e_h2_singlet_EF", 1.0),
    "a": ("e_h2_triplet_a", 3.0),
    "d": ("e_h2_triplet_d", 6.0),
}
#: Radiative decays (upper, lower, lifetime in s) of the electronic states;
#: ``lower=None`` is the dissociative continuum a -> b -> H + H.
RADIATIVE_DECAYS: tuple[tuple[str, str | None, float], ...] = (
    ("B", "X", 0.8e-9),
    ("C", "X", 0.6e-9),
    ("EF", "B", 100.0e-9),
    ("a", None, 11.0e-9),
    ("d", "a", 40.0e-9),
)

LIFETIME_PROVENANCE = Provenance(
    Reference(
        "U. Fantz and D. Wuenderlich, Franck-Condon factors, transition probabilities "
        "and radiative lifetimes for hydrogen molecules and their isotopomeres, "
        "At. Data Nucl. Data Tables 92, 853 (2006)",
        doi="10.1016/j.adt.2006.05.001",
        detail="radiative lifetimes of v' = 0, rounded",
    ),
    units="s^-1 (1 / lifetime)",
    uncertainty=0.3,
    uncertainty_note="v'-averaged to one value per state; the lifetimes vary with v' "
    "by tens of percent",
    licence="published-fit",
    notes="every decay of B and C goes to X(v=0): the vibrational redistribution of the "
    "radiative cascade (EV excitation) is not resolved",
)


def vibrational_energy(v: int) -> float:
    """Return the energy of H2(X, v) on the common scale (eV)."""
    if not 0 <= v < NUM_VIBRATIONAL_LEVELS:
        raise ValueError(f"v must be in [0, {NUM_VIBRATIONAL_LEVELS - 1}]")
    table = H2_GROUND_STATE_VIBRATIONAL_ENERGY_EV
    return -H2_DISSOCIATION_ENERGY + table[v] - table[0]


def molecular_states(
    v_max: int = NUM_VIBRATIONAL_LEVELS - 1, electronic: bool = True
) -> list[State]:
    """Return H2(X, v=0..v_max) and, with ``electronic``, the excited states.

    The states are named ``H2(X,v=<v>)`` and ``H2(<label>)``.
    """
    if not 0 <= v_max < NUM_VIBRATIONAL_LEVELS:
        raise ValueError(f"v_max must be in [0, {NUM_VIBRATIONAL_LEVELS - 1}]")
    h2 = species_from_formula("H2")
    states = [
        State(h2, f"X,v={v}", vibrational_energy(v), 1.0, {"electronic": "X", "v": v})
        for v in range(v_max + 1)
    ]
    if electronic:
        base = vibrational_energy(0)
        for label, (key, weight) in ELECTRONIC_STATES.items():
            states.append(
                State(
                    h2,
                    label,
                    base + JANEV_FITS[key].threshold,
                    weight,
                    {"electronic": label},
                )
            )
    return states


def ion_states() -> dict[str, State]:
    """Return the unresolved H2+, H3+ and H- (and H+) by name."""
    h2_ion = H2_IONIZATION_ENERGY - H2_DISSOCIATION_ENERGY
    return {
        "H+": State(species_from_formula("H+"), "", HYDROGEN_IONIZATION_ENERGY, 1.0),
        "H2+": State(species_from_formula("H2+"), "", h2_ion, 2.0),
        "H3+": State(
            species_from_formula("H3+"),
            "",
            -H2_DISSOCIATION_ENERGY + HYDROGEN_IONIZATION_ENERGY - H2_PROTON_AFFINITY,
            1.0,
        ),
        "H-": State(species_from_formula("H-"), "", -H_MINUS_BINDING_ENERGY, 1.0),
    }
