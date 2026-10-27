"""Reactions between states.

A :class:`Reaction` lists its reactant and product states (a state that
appears twice counts twice), its kind, its rate model and the provenance of
its data. Creating one checks the charge and element balance; photons are
implicit products of the radiative kinds (:data:`RADIATIVE_KINDS`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from opencr.core.rates import (
    Constant,
    CrossSectionRate,
    DetailedBalance,
    RateModel,
)
from opencr.core.species import ELECTRON, State
from opencr.data.provenance import Provenance

__all__ = ["RADIATIVE_KINDS", "REACTION_KINDS", "Reaction"]

#: The kinds of reaction OpenCR knows. ``other`` is for anything else.
REACTION_KINDS = (
    "excitation",
    "deexcitation",
    "ionization",
    "three_body_recombination",
    "radiative_recombination",
    "dielectronic_recombination",
    "spontaneous_emission",
    "dissociation",
    "dissociative_excitation",
    "dissociative_ionization",
    "dissociative_recombination",
    "attachment",
    "dissociative_attachment",
    "detachment",
    "charge_exchange",
    "mutual_neutralization",
    "associative_detachment",
    "penning_ionization",
    "quenching",
    "vibrational_transfer",
    "proton_transfer",
    "wall_loss",
    "pumping",
    "other",
)
#: The kinds that emit a photon.
RADIATIVE_KINDS = ("spontaneous_emission", "radiative_recombination")


@dataclass(frozen=True)
class Reaction:
    """One reaction.

    Attributes:
        name: A unique name in its network, e.g. ``"exc:H(n=1)->H(n=2)"``.
        reactants: The reactant states; the electron is
            :data:`~opencr.core.species.ELECTRON_STATE`.
        products: The product states.
        kind: One of :data:`REACTION_KINDS`.
        rate: The rate model.
        provenance: The origin of the rate data.
        escape: For a spontaneous emission, the default escape factor of the
            line (1 for an optically thin plasma); conditions may override it.
    """

    name: str
    reactants: tuple[State, ...]
    products: tuple[State, ...]
    kind: str
    rate: RateModel
    provenance: Provenance | None = None
    escape: float = 1.0

    def __post_init__(self) -> None:
        """Check the kind, the balance and the rate model."""
        object.__setattr__(self, "reactants", tuple(self.reactants))
        object.__setattr__(self, "products", tuple(self.products))
        if not self.name:
            raise ValueError("a reaction needs a name")
        if self.kind not in REACTION_KINDS:
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")
        if not self.reactants:
            raise ValueError(f"{self.name}: a reaction needs reactants")
        if not isinstance(self.rate, RateModel):
            raise TypeError(f"{self.name}: rate must be a RateModel")
        if not 0.0 <= self.escape <= 1.0:
            raise ValueError(f"{self.name}: the escape factor must be in [0, 1]")
        charge_in = sum(s.charge for s in self.reactants)
        charge_out = sum(s.charge for s in self.products)
        if charge_in != charge_out and self.kind not in ("wall_loss", "pumping"):
            raise ValueError(
                f"{self.name}: charge {charge_in} -> {charge_out} is not conserved"
            )
        if _elements(self.reactants) != _elements(self.products) and self.kind not in (
            "wall_loss",
            "pumping",
        ):
            raise ValueError(f"{self.name}: the elements are not conserved")
        if isinstance(self.rate, Constant) and self.kind in RADIATIVE_KINDS[:1]:
            if self.order != 1:
                raise ValueError(f"{self.name}: spontaneous emission has one reactant")
        if (
            isinstance(self.rate, CrossSectionRate)
            and self.rate.depends_on == "electron"
        ):
            if self.reactant_counts.get(ELECTRON.name, 0) != 1 or self.order != 2:
                raise ValueError(
                    f"{self.name}: an electron cross section needs an electron and one target"
                )

    @property
    def order(self) -> int:
        """The number of reactant particles."""
        return len(self.reactants)

    @property
    def reactant_counts(self) -> dict[str, int]:
        """How often each state is a reactant, by name."""
        return dict(Counter(s.name for s in self.reactants))

    @property
    def product_counts(self) -> dict[str, int]:
        """How often each state is a product, by name."""
        return dict(Counter(s.name for s in self.products))

    @property
    def net(self) -> dict[str, int]:
        """The net change of each state per reaction (products minus reactants)."""
        change = Counter(self.product_counts)
        change.subtract(self.reactant_counts)
        return {k: v for k, v in change.items() if v != 0}

    @property
    def equation(self) -> str:
        """The reaction as text, e.g. ``"e + H(n=1) -> e + H(n=2)"``."""
        left = " + ".join(s.name for s in self.reactants)
        right = " + ".join(s.name for s in self.products) or "(lost)"
        if self.kind in RADIATIVE_KINDS:
            right += " + hv"
        return f"{left} -> {right}"

    @property
    def energy_change(self) -> float:
        """The internal energy of the products minus the reactants' in eV."""
        return sum(s.energy for s in self.products) - sum(
            s.energy for s in self.reactants
        )

    @property
    def photon_energy(self) -> float | None:
        """For a spontaneous emission, the photon energy in eV; otherwise ``None``."""
        if self.kind != "spontaneous_emission":
            return None
        return -self.energy_change

    @property
    def is_inverse(self) -> bool:
        """Whether the rate is the detailed balance of another reaction."""
        return isinstance(self.rate, DetailedBalance)

    def inverse_of(self, forward: Reaction) -> bool:
        """Return whether this reaction is ``forward`` run backwards."""
        return Counter(self.reactant_counts) == Counter(
            forward.product_counts
        ) and Counter(self.product_counts) == Counter(forward.reactant_counts)


def _elements(states: Sequence[State]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for state in states:
        total.update(state.species.composition)
    return {k: v for k, v in total.items() if v}
