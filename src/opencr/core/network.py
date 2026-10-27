"""A reaction network: the states, the reactions between them and what is assumed.

:class:`ReactionNetwork` holds plain Python objects and no arrays; the
numerical layer compiles it (:mod:`opencr.kinetics.system`). Building one
checks it (:meth:`ReactionNetwork.problems`): unique names, every state of
every reaction in the network, the forward reaction of every detailed
balance, the energy ordering of excitations and emissions. The
:meth:`~ReactionNetwork.manifest` lists the states, the reactions, their
sources and the assumptions in machine-readable form.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from opencr.core.rates import DetailedBalance
from opencr.core.reactions import Reaction
from opencr.core.species import ELECTRON, Species, State

__all__ = ["NetworkError", "ReactionNetwork"]


class NetworkError(ValueError):
    """A reaction network is inconsistent; ``problems`` lists every problem found."""

    def __init__(self, problems: Sequence[str]) -> None:
        """Keep the list and show it in the message."""
        self.problems = list(problems)
        super().__init__("invalid reaction network:\n  " + "\n  ".join(self.problems))


@dataclass(frozen=True)
class _Conservation:
    names: tuple[str, ...]
    matrix: np.ndarray


class ReactionNetwork:
    """States and reactions, with the model's assumptions.

    Args:
        states: The states (the electron is implicit and must not be listed).
        reactions: The reactions; every state they use must be in ``states``.
        name: The name of the network, e.g. ``"hydrogen/atomic/basic"``.
        assumptions: The model's assumptions and omissions, one sentence each.
        check: Raise :class:`NetworkError` if :meth:`problems` finds any.
    """

    def __init__(
        self,
        states: Iterable[State],
        reactions: Iterable[Reaction],
        name: str = "",
        assumptions: Iterable[str] = (),
        check: bool = True,
    ) -> None:
        """Store and check the network."""
        self.states: tuple[State, ...] = tuple(states)
        self.reactions: tuple[Reaction, ...] = tuple(reactions)
        self.name = name
        self.assumptions: tuple[str, ...] = tuple(assumptions)
        self._state_index = {s.name: i for i, s in enumerate(self.states)}
        self._reaction_index = {r.name: i for i, r in enumerate(self.reactions)}
        if check:
            problems = self.problems()
            if problems:
                raise NetworkError(problems)

    @property
    def state_names(self) -> tuple[str, ...]:
        """The names of the states, in order."""
        return tuple(s.name for s in self.states)

    @property
    def species(self) -> tuple[Species, ...]:
        """The species of the states, each once, in order of appearance."""
        seen: dict[str, Species] = {}
        for state in self.states:
            seen.setdefault(state.species.name, state.species)
        return tuple(seen.values())

    def state(self, name: str) -> State:
        """Return the state called ``name``."""
        try:
            return self.states[self._state_index[name]]
        except KeyError:
            raise KeyError(
                f"no state {name!r} in {self.name or 'the network'}"
            ) from None

    def state_index(self, name: str) -> int:
        """Return the position of the state called ``name``."""
        if name not in self._state_index:
            raise KeyError(f"no state {name!r} in {self.name or 'the network'}")
        return self._state_index[name]

    def reaction(self, name: str) -> Reaction:
        """Return the reaction called ``name``."""
        try:
            return self.reactions[self._reaction_index[name]]
        except KeyError:
            raise KeyError(
                f"no reaction {name!r} in {self.name or 'the network'}"
            ) from None

    def reactions_of(self, state: str) -> tuple[Reaction, ...]:
        """Return the reactions that change the population of ``state``."""
        return tuple(r for r in self.reactions if state in r.net)

    def _kind_problems(self, reaction: Reaction) -> list[str]:
        problems = []
        if isinstance(reaction.rate, DetailedBalance):
            forward = reaction.rate.forward
            if forward not in self._reaction_index:
                problems.append(
                    f"{reaction.name}: forward reaction {forward!r} is missing"
                )
            else:
                other = self.reaction(forward)
                if not reaction.inverse_of(other):
                    problems.append(
                        f"{reaction.name}: is not the inverse of {forward!r}"
                    )
                if isinstance(other.rate, DetailedBalance):
                    problems.append(
                        f"{reaction.name}: {forward!r} is itself an inverse"
                    )
        if (
            reaction.kind in ("excitation", "ionization")
            and reaction.energy_change <= 0.0
        ):
            problems.append(
                f"{reaction.name}: an {reaction.kind} must raise the energy"
            )
        if reaction.kind in ("deexcitation", "spontaneous_emission") and (
            reaction.energy_change >= 0.0
        ):
            problems.append(f"{reaction.name}: a {reaction.kind} must lower the energy")
        return problems

    def problems(self) -> list[str]:
        """Return every inconsistency found, as sentences; empty if none."""
        problems: list[str] = []
        names = Counter(s.name for s in self.states)
        problems += [
            f"state {n!r} is defined {c} times" for n, c in names.items() if c > 1
        ]
        if ELECTRON.name in names:
            problems.append("the electron is implicit; do not list it as a state")
        rnames = Counter(r.name for r in self.reactions)
        problems += [
            f"reaction {n!r} is defined {c} times" for n, c in rnames.items() if c > 1
        ]
        for reaction in self.reactions:
            for state in (*reaction.reactants, *reaction.products):
                if state.species == ELECTRON:
                    continue
                if state.name not in self._state_index:
                    problems.append(
                        f"{reaction.name}: state {state.name!r} is not in the network"
                    )
                elif self.state(state.name) != state:
                    problems.append(
                        f"{reaction.name}: state {state.name!r} differs from the network's"
                    )
            problems += self._kind_problems(reaction)
        used = {
            n for r in self.reactions for n in (*r.reactant_counts, *r.product_counts)
        }
        problems += [
            f"state {s.name!r} takes part in no reaction"
            for s in self.states
            if s.name not in used and len(self.states) > 1
        ]
        return problems

    def conservation(self) -> _Conservation:
        """Return the conserved quantities over the states: the elements and the charge.

        The matrix has one row per quantity and one column per state: the
        number of atoms of the element in the state, or its charge. Electrons
        are not states, so charge is conserved only together with them.
        """
        elements = sorted({e for s in self.states for e in s.species.composition})
        rows = [
            [s.species.composition.get(e, 0) for s in self.states] for e in elements
        ]
        return _Conservation(
            tuple(elements), np.asarray(rows, dtype=float).reshape(len(rows), -1)
        )

    def subset(self, keep: Iterable[str], name: str | None = None) -> ReactionNetwork:
        """Return the network of the states ``keep`` and the reactions among them only."""
        names = set(keep)
        unknown = names - set(self._state_index)
        if unknown:
            raise KeyError(f"unknown states {sorted(unknown)}")
        states = [s for s in self.states if s.name in names]
        allowed = names | {ELECTRON.name}
        reactions = [
            r
            for r in self.reactions
            if set(r.reactant_counts) | set(r.product_counts) <= allowed
        ]
        kept = {r.name for r in reactions}
        reactions = [
            r
            for r in reactions
            if not isinstance(r.rate, DetailedBalance) or r.rate.forward in kept
        ]
        return ReactionNetwork(
            states, reactions, name if name is not None else self.name, self.assumptions
        )

    def merge(self, other: ReactionNetwork, name: str = "") -> ReactionNetwork:
        """Return the union of two networks; shared states must be identical."""
        states = list(self.states)
        for state in other.states:
            if state.name in self._state_index:
                if self.state(state.name) != state:
                    raise NetworkError(
                        [f"state {state.name!r} differs between the networks"]
                    )
            else:
                states.append(state)
        reactions = list(self.reactions) + [
            r for r in other.reactions if r.name not in self._reaction_index
        ]
        assumptions = list(dict.fromkeys((*self.assumptions, *other.assumptions)))
        return ReactionNetwork(states, reactions, name or self.name, assumptions)

    def manifest(self) -> dict[str, Any]:
        """Return the states, reactions, sources and assumptions as plain data."""
        sources: dict[str, dict[str, Any]] = {}
        reactions = []
        for reaction in self.reactions:
            entry: dict[str, Any] = {
                "name": reaction.name,
                "equation": reaction.equation,
                "kind": reaction.kind,
                "rate": reaction.rate.kind,
                "depends_on": reaction.rate.depends_on,
            }
            if isinstance(reaction.rate, DetailedBalance):
                entry["forward"] = reaction.rate.forward
            if reaction.provenance is not None:
                key = str(reaction.provenance.reference)
                entry["source"] = key
                sources.setdefault(key, reaction.provenance.to_dict())
            reactions.append(entry)
        return {
            "name": self.name,
            "states": [s.to_dict() | {"name": s.name} for s in self.states],
            "species": [s.to_dict() for s in self.species],
            "reactions": reactions,
            "sources": sources,
            "assumptions": list(self.assumptions),
            "counts": {
                "states": len(self.states),
                "reactions": len(self.reactions),
                "by_kind": dict(Counter(r.kind for r in self.reactions)),
            },
        }

    def __repr__(self) -> str:
        """The name and the sizes."""
        return (
            f"ReactionNetwork({self.name!r}, {len(self.states)} states, "
            f"{len(self.reactions)} reactions)"
        )
