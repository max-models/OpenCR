"""OpenCR's native, versioned file format for reaction networks.

A network is written as one JSON document::

    {"format": "opencr-network", "version": 1, "name": ..., "assumptions": [...],
     "species": [...], "states": [...], "reactions": [...]}

Each reaction lists its reactants and products by state name (``"e"`` for the
electron), its kind, its rate model (:meth:`opencr.core.rates.RateModel.to_dict`;
a :class:`~opencr.core.rates.Function` is written as its table) and its
provenance. Reading checks the format name and version, then builds and
checks the network. Files of a newer major version are refused.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from plasmacoll import CrossSection

from opencr.core.network import ReactionNetwork
from opencr.core.rates import (
    Arrhenius,
    Constant,
    CrossSectionRate,
    DetailedBalance,
    RateModel,
    Tabulated,
)
from opencr.core.reactions import Reaction
from opencr.core.species import ELECTRON, ELECTRON_STATE, Species, State
from opencr.data.checks import DataError
from opencr.data.provenance import Provenance

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "load_network",
    "network_from_dict",
    "network_to_dict",
    "rate_from_dict",
    "save_network",
]

FORMAT_NAME = "opencr-network"
FORMAT_VERSION = 1


def rate_from_dict(data: Mapping[str, Any]) -> RateModel:
    """Return the rate model of :meth:`~opencr.core.rates.RateModel.to_dict`'s output."""
    data = dict(data)
    kind = data.pop("kind")
    depends_on = data.pop("depends_on", "electron")
    if kind == "cross_section":
        cross_section = CrossSection(
            energy=data["energy"],
            sigma=data["sigma"],
            threshold=data.get("threshold", 0.0),
            label=data.get("label", ""),
        )
        return CrossSectionRate(
            depends_on=depends_on,
            cross_section=cross_section,
            mass=data["mass"],
            incident_mass=data.get("incident_mass"),
        )
    if kind == "arrhenius":
        return Arrhenius(depends_on=depends_on, a=data["a"], b=data["b"], c=data["c"])
    if kind in ("tabulated", "function"):
        return Tabulated(
            depends_on=depends_on,
            temperatures=tuple(data["temperatures"]),
            values=tuple(data["values"]),
            threshold=data.get("threshold", 0.0),
        )
    if kind == "constant":
        return Constant(depends_on=depends_on, value=data["value"])
    if kind == "detailed_balance":
        return DetailedBalance(depends_on=depends_on, forward=data["forward"])
    raise DataError([f"unknown rate model {kind!r}"])


def network_to_dict(network: ReactionNetwork) -> dict[str, Any]:
    """Return the network in the native format as plain data."""
    reactions = []
    for reaction in network.reactions:
        reactions.append(
            {
                "name": reaction.name,
                "kind": reaction.kind,
                "reactants": [s.name for s in reaction.reactants],
                "products": [s.name for s in reaction.products],
                "rate": reaction.rate.to_dict(),
                "provenance": None
                if reaction.provenance is None
                else reaction.provenance.to_dict(),
                "escape": reaction.escape,
            }
        )
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "name": network.name,
        "assumptions": list(network.assumptions),
        "species": [s.to_dict() for s in network.species],
        "states": [s.to_dict() for s in network.states],
        "reactions": reactions,
    }


def network_from_dict(data: Mapping[str, Any]) -> ReactionNetwork:
    """Return the network of :func:`network_to_dict`'s output, checked."""
    problems = []
    if data.get("format") != FORMAT_NAME:
        problems.append(
            f"not an {FORMAT_NAME} document (format {data.get('format')!r})"
        )
    version = data.get("version")
    if not isinstance(version, int) or version > FORMAT_VERSION or version < 1:
        problems.append(
            f"unsupported version {version!r}; this OpenCR reads up to {FORMAT_VERSION}"
        )
    for key in ("species", "states", "reactions"):
        if key not in data:
            problems.append(f"missing {key!r}")
    if problems:
        raise DataError(problems)
    species = {
        s["name"]: Species(s["name"], s["charge"], s["mass"], s["composition"])
        for s in data["species"]
    }
    states: dict[str, State] = {}
    for item in data["states"]:
        if item["species"] not in species:
            raise DataError([f"state of unknown species {item['species']!r}"])
        state = State(
            species[item["species"]],
            item.get("label", ""),
            item.get("energy", 0.0),
            item.get("weight", 1.0),
            item.get("quantum_numbers", {}),
        )
        states[state.name] = state
    states_and_electron = {**states, ELECTRON.name: ELECTRON_STATE}
    reactions = []
    for item in data["reactions"]:
        missing = [
            n
            for n in (*item["reactants"], *item["products"])
            if n not in states_and_electron
        ]
        if missing:
            raise DataError([f"{item['name']}: unknown states {missing}"])
        provenance = item.get("provenance")
        reactions.append(
            Reaction(
                item["name"],
                tuple(states_and_electron[n] for n in item["reactants"]),
                tuple(states_and_electron[n] for n in item["products"]),
                item["kind"],
                rate_from_dict(item["rate"]),
                None if provenance is None else Provenance.from_dict(provenance),
                item.get("escape", 1.0),
            )
        )
    return ReactionNetwork(
        states.values(), reactions, data.get("name", ""), data.get("assumptions", ())
    )


def save_network(network: ReactionNetwork, path: str | Path) -> Path:
    """Write the network to ``path`` as native JSON and return the path."""
    path = Path(path)
    path.write_text(json.dumps(network_to_dict(network), indent=1))
    return path


def load_network(path: str | Path) -> ReactionNetwork:
    """Read a network written by :func:`save_network`."""
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError as error:
        raise DataError([f"{path}: not JSON ({error})"]) from error
    return network_from_dict(data)
