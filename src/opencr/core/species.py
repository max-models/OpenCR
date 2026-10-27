"""Species and their quantum states.

A :class:`Species` is a particle kind named by its formula, as in plasmacoll
and wellbalanced (``"e"``, ``"H"``, ``"H+"``, ``"H2"``, ``"H-"``), with its
charge, mass and elemental composition. A :class:`State` is one state of a
species that the model resolves: an atomic level ``H(n=2)``, a fine-structure
level, a molecular electronic-vibrational state ``H2(X,v=3)``, or the species
as a whole (an empty label) when it is not resolved. The solver sees only
states, so the resolution of a model changes without changing the solver.

Energies of the states of one element share one scale: zero at the ground
state of the neutral, the ion's ground state at the ionization energy. Detailed
balance and the photon energies of radiative transitions are computed from
these energies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from opencr.constants import ATOMIC_MASS, ELECTRON_MASS

__all__ = ["ELECTRON", "ELECTRON_STATE", "Species", "State", "species_from_formula"]

#: Atomic masses of the elements OpenCR knows by name, in u.
_ELEMENT_MASSES = {
    "H": 1.00794,
    "D": 2.014102,
    "He": 4.002602,
    "Ar": 39.948,
    "Cs": 132.905,
}


@dataclass(frozen=True)
class Species:
    """A particle kind.

    Attributes:
        name: The formula, e.g. ``"H2+"``.
        charge: The charge in units of e.
        mass: The mass in kg.
        composition: The number of atoms of each element, e.g. ``{"H": 2}``;
            empty for the electron.
    """

    name: str
    charge: int
    mass: float
    composition: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the definition and freeze the composition."""
        if not self.name:
            raise ValueError("a species needs a name")
        if int(self.charge) != self.charge:
            raise ValueError(f"{self.name}: the charge must be an integer")
        if self.mass <= 0.0:
            raise ValueError(f"{self.name}: the mass must be > 0")
        if any(int(v) != v or v <= 0 for v in self.composition.values()):
            raise ValueError(
                f"{self.name}: composition counts must be positive integers"
            )
        object.__setattr__(self, "charge", int(self.charge))
        object.__setattr__(
            self,
            "composition",
            MappingProxyType(dict(sorted(self.composition.items()))),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the species as plain data."""
        return {
            "name": self.name,
            "charge": self.charge,
            "mass": self.mass,
            "composition": dict(self.composition),
        }

    def __eq__(self, other: object) -> bool:
        """Compare every field."""
        if not isinstance(other, Species):
            return NotImplemented
        return (self.name, self.charge, self.mass, dict(self.composition)) == (
            other.name,
            other.charge,
            other.mass,
            dict(other.composition),
        )

    def __hash__(self) -> int:
        """Hash by name, charge and mass (the composition is a mapping)."""
        return hash((self.name, self.charge, self.mass))


#: The electron.
ELECTRON = Species("e", -1, ELECTRON_MASS, {})


def species_from_formula(formula: str) -> Species:
    """Return the species of a simple formula such as ``"H"``, ``"H2+"``, ``"Ar"`` or ``"H-"``.

    The formula is one element symbol, an optional atom count and the charge
    as repeated signs (``"H2+"``, ``"H-"``, ``"Ar++"``). The mass is the sum of
    the atomic masses minus the missing electrons (plus the extra ones of a
    negative ion).
    """
    if formula == "e":
        return ELECTRON
    symbol = formula[0]
    rest = formula[1:]
    while rest and rest[0].islower():
        symbol += rest[0]
        rest = rest[1:]
    if symbol not in _ELEMENT_MASSES:
        raise ValueError(f"unknown element in {formula!r}; build the Species yourself")
    digits = ""
    while rest and rest[0].isdigit():
        digits += rest[0]
        rest = rest[1:]
    count = int(digits) if digits else 1
    if rest == "":
        charge = 0
    elif set(rest) == {"+"} or set(rest) == {"-"}:
        charge = len(rest) * (1 if rest[0] == "+" else -1)
    else:
        raise ValueError(f"cannot read the charge of {formula!r}")
    mass = count * _ELEMENT_MASSES[symbol] * ATOMIC_MASS - charge * ELECTRON_MASS
    return Species(formula, charge, mass, {symbol: count})


@dataclass(frozen=True)
class State:
    """One resolved state of a species.

    Attributes:
        species: The species.
        label: The quantum label, e.g. ``"n=2"`` or ``"X,v=3"``; empty when the
            species is not resolved further.
        energy: The energy in eV on the common scale of the element (see the
            module docstring).
        weight: The statistical weight ``g``.
        quantum_numbers: The quantum numbers by name, e.g. ``{"n": 2}``,
            ``{"n": 2, "l": 1}`` or ``{"electronic": "X", "v": 3}``.
    """

    species: Species
    label: str = ""
    energy: float = 0.0
    weight: float = 1.0
    quantum_numbers: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the weight and freeze the quantum numbers."""
        if self.weight <= 0.0:
            raise ValueError(f"{self.name}: the statistical weight must be > 0")
        object.__setattr__(
            self, "quantum_numbers", MappingProxyType(dict(self.quantum_numbers))
        )

    @property
    def name(self) -> str:
        """The state's name, ``"H(n=2)"``, or the species' name when unresolved."""
        return f"{self.species.name}({self.label})" if self.label else self.species.name

    @property
    def charge(self) -> int:
        """The charge in units of e."""
        return self.species.charge

    def to_dict(self) -> dict[str, Any]:
        """Return the state as plain data (the species by name)."""
        return {
            "species": self.species.name,
            "label": self.label,
            "energy": self.energy,
            "weight": self.weight,
            "quantum_numbers": dict(self.quantum_numbers),
        }

    def __eq__(self, other: object) -> bool:
        """Compare every field."""
        if not isinstance(other, State):
            return NotImplemented
        return (
            self.species == other.species
            and self.label == other.label
            and self.energy == other.energy
            and self.weight == other.weight
            and dict(self.quantum_numbers) == dict(other.quantum_numbers)
        )

    def __hash__(self) -> int:
        """Hash by name (names are unique in a network)."""
        return hash(self.name)


#: The free electron, with its spin weight 2.
ELECTRON_STATE = State(ELECTRON, weight=2.0)
