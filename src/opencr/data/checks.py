"""Checks of data before it enters a model.

Every importer runs these checks, and :func:`check_network_data` runs them on a
whole network. A check returns a list of problems as sentences (empty when the
data are fine); :func:`raise_problems` turns a non-empty list into a
:class:`DataError`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import cunumpy as xp
import numpy as np

from opencr.core.network import ReactionNetwork
from opencr.core.rates import CrossSectionRate, DetailedBalance, Tabulated
from opencr.core.species import State
from opencr.data.provenance import Provenance

__all__ = [
    "REQUIRED_PROVENANCE",
    "DataError",
    "check_cross_section",
    "check_levels",
    "check_network_data",
    "check_provenance",
    "check_rate_table",
    "raise_problems",
]

#: The provenance fields every imported dataset must fill.
REQUIRED_PROVENANCE = ("reference", "units", "licence")


class DataError(ValueError):
    """Data failed a check; ``problems`` lists every problem found."""

    def __init__(self, problems: Sequence[str]) -> None:
        """Keep the list and show it in the message."""
        self.problems = list(problems)
        super().__init__("invalid data:\n  " + "\n  ".join(self.problems))


def raise_problems(problems: Sequence[str]) -> None:
    """Raise :class:`DataError` if ``problems`` is not empty."""
    if problems:
        raise DataError(problems)


def check_cross_section(
    energy: Iterable[float],
    sigma: Iterable[float],
    threshold: float = 0.0,
    name: str = "cross section",
    max_sigma: float = 1.0e-14,
    strict_threshold: bool = True,
) -> list[str]:
    """Check a cross-section table (eV, m^2).

    Problems: arrays of different shapes or fewer than two points, missing
    or infinite values, energies not increasing, negative energies or cross
    sections, a negative threshold, a non-zero cross section below the
    threshold (unless ``strict_threshold`` is false: plasmacoll sets a cross
    section to zero below its threshold whatever the table says), and values
    above ``max_sigma`` (1e-14 m^2 by default, a hint that the table is in cm^2).
    """
    e = np.asarray(xp.to_numpy(xp.asarray(energy, dtype=float)))
    s = np.asarray(xp.to_numpy(xp.asarray(sigma, dtype=float)))
    problems: list[str] = []
    if e.ndim != 1 or e.shape != s.shape:
        return [f"{name}: energy and sigma must be 1D arrays of equal length"]
    if e.size < 2:
        problems.append(f"{name}: fewer than two points")
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(s))):
        problems.append(f"{name}: missing or infinite values")
        return problems
    if np.any(np.diff(e) < 0.0):
        problems.append(f"{name}: energies are not increasing")
    if np.any(e < 0.0):
        problems.append(f"{name}: negative energies")
    if np.any(s < 0.0):
        problems.append(f"{name}: negative cross sections")
    if threshold < 0.0:
        problems.append(f"{name}: negative threshold")
    if strict_threshold and np.any(s[e < threshold] > 0.0):
        problems.append(f"{name}: non-zero below the threshold {threshold} eV")
    if np.any(s > max_sigma):
        problems.append(
            f"{name}: values above {max_sigma:g} m^2 (cm^2 instead of m^2?)"
        )
    return problems


def check_rate_table(
    temperatures: Iterable[float],
    values: Iterable[float],
    name: str = "rate table",
) -> list[str]:
    """Check a rate-coefficient table (eV, SI).

    Problems: arrays of different shapes or fewer than two points, missing or
    infinite values, temperatures not positive and increasing, negative rates.
    """
    t = np.asarray(temperatures, dtype=float)
    k = np.asarray(values, dtype=float)
    if t.ndim != 1 or t.shape != k.shape:
        return [f"{name}: temperature and rate must be 1D arrays of equal length"]
    problems: list[str] = []
    if t.size < 2:
        problems.append(f"{name}: fewer than two points")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(k))):
        problems.append(f"{name}: missing or infinite values")
        return problems
    if np.any(t <= 0.0) or np.any(np.diff(t) <= 0.0):
        problems.append(f"{name}: temperatures must be positive and increasing")
    if np.any(k < 0.0):
        problems.append(f"{name}: negative rates")
    return problems


def check_levels(states: Sequence[State]) -> list[str]:
    """Check the levels of one or more species.

    Problems: repeated names, missing or infinite energies, and, per species,
    a ground state not at the lowest energy among states ordered by their
    principal quantum number ``n`` (energies must increase with ``n``).
    """
    problems: list[str] = []
    seen: set[str] = set()
    for state in states:
        if state.name in seen:
            problems.append(f"level {state.name!r} is listed twice")
        seen.add(state.name)
        if not np.isfinite(state.energy):
            problems.append(f"level {state.name!r} has no finite energy")
    by_species: dict[str, list[State]] = {}
    for state in states:
        if "n" in state.quantum_numbers:
            by_species.setdefault(state.species.name, []).append(state)
    for species, levels in by_species.items():
        ordered = sorted(levels, key=lambda s: s.quantum_numbers["n"])
        energies = [s.energy for s in ordered]
        if any(b < a for a, b in zip(energies[:-1], energies[1:], strict=True)):
            problems.append(f"{species}: level energies do not increase with n")
    return problems


def check_provenance(
    provenance: Provenance | None,
    name: str = "data",
    required: Sequence[str] = REQUIRED_PROVENANCE,
) -> list[str]:
    """Check that the fields ``required`` of a provenance are filled.

    ``reference`` needs a citation, ``licence`` a value other than
    ``"unknown"``; any other field must be non-empty.
    """
    if provenance is None:
        return [f"{name}: no provenance"]
    problems: list[str] = []
    for key in required:
        if key == "reference":
            if not provenance.reference.citation:
                problems.append(f"{name}: the reference has no citation")
        elif key == "licence":
            if provenance.licence in ("", "unknown"):
                problems.append(f"{name}: the licence is unknown")
        elif not getattr(provenance, key):
            problems.append(f"{name}: provenance field {key!r} is empty")
    return problems


def check_network_data(
    network: ReactionNetwork, required: Sequence[str] = REQUIRED_PROVENANCE
) -> list[str]:
    """Check the data of every reaction and level of a network.

    The levels (:func:`check_levels`), every cross section and rate table, and
    the provenance of every reaction that is not a detailed balance.
    """
    problems = check_levels(network.states)
    for reaction in network.reactions:
        rate = reaction.rate
        if isinstance(rate, CrossSectionRate):
            cs = rate.table
            problems += check_cross_section(
                cs.energy, cs.sigma, cs.threshold, reaction.name, strict_threshold=False
            )
        elif isinstance(rate, Tabulated):
            problems += check_rate_table(rate.temperatures, rate.values, reaction.name)
        if not isinstance(rate, DetailedBalance):
            problems += check_provenance(reaction.provenance, reaction.name, required)
    return problems
