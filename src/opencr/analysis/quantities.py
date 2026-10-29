"""Scalar quantities of a steady state, named by strings.

The surrogates (:mod:`opencr.ml`), the sensitivities and the inverse
modelling (:mod:`opencr.inference`) all ask a :class:`~opencr.analysis.results.Result`
for named scalars:

==========================================  ===============================================
name                                        quantity
==========================================  ===============================================
``population:<state>``                      the density of a state (m^-3)
``effective_rate:<source>><target>``        an effective rate coefficient (m^3/s), quasi-static
``emissivity:<reaction>``                   the photon emissivity of a line (photons m^-3 s^-1)
``electron_cooling:<driver>``               the electron energy loss per electron and per
                                            particle of a driver (eV m^3/s), quasi-static
``ratio:<quantity>/<quantity>``             the ratio of two of the above
==========================================  ===============================================

For example ``"effective_rate:H(n=1)>H+"`` is the CR ionization coefficient
of atomic hydrogen and ``"ratio:emissivity:A:3-2/emissivity:A:4-2"`` the
Balmer alpha to beta ratio.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["KINDS", "check_quantity", "electron_cooling", "quantities", "quantity"]

#: The kinds of quantity.
KINDS = ("population", "effective_rate", "emissivity", "electron_cooling", "ratio")


def check_quantity(model: Any, name: str) -> None:
    """Raise ``ValueError`` (or ``KeyError``) if ``name`` is not a quantity of ``model``."""
    kind, _, rest = name.partition(":")
    if kind not in KINDS or not rest:
        raise ValueError(f"{name!r}: a quantity is <kind>:<what> with kind in {KINDS}")
    if kind == "ratio":
        top, slash, bottom = rest.partition("/")
        if not slash:
            raise ValueError(f"{name!r}: a ratio is ratio:<quantity>/<quantity>")
        check_quantity(model, top)
        check_quantity(model, bottom)
    elif kind in ("population", "electron_cooling"):
        model.network.state(rest)
    elif kind == "effective_rate":
        source, arrow, target = rest.partition(">")
        if not arrow:
            raise ValueError(
                f"{name!r}: an effective rate is effective_rate:<source>><target>"
            )
        model.network.state(source)
        model.network.state(target)
    else:
        if model.network.reaction(rest).kind != "spontaneous_emission":
            raise ValueError(f"{name!r}: {rest} is not a spontaneous emission")


def electron_cooling(result: Any, driver: str) -> float:
    """Return the electron energy loss due to ``driver`` per electron and particle (eV m^3/s).

    The sum over the electron-impact reactions (an electron among the
    reactants, not radiative) of ``k n_e^(a-1) c_X Delta E``, where ``c_X`` is
    the quasi-static population of the reaction's heavy reactant per particle
    of ``driver`` (1 for the driver itself, 0 for the other drivers) and
    ``Delta E`` the energy the heavy particles gain: the excitation and
    ionization energies, negative for de-excitation and three-body
    recombination. Radiative recombination (the electron's kinetic energy) is
    not counted.
    """
    from opencr.core.reactions import RADIATIVE_KINDS
    from opencr.core.species import ELECTRON

    reduction = result.reduction
    column = reduction.drivers.index(driver)
    ne = reduction.electron_density
    share = {d: 0.0 for d in reduction.drivers}
    share[driver] = 1.0
    for name, row in zip(reduction.excited, reduction.coefficients, strict=True):
        share[name] = float(row[column])
    total = 0.0
    for k, reaction in zip(result.rates, result.model.network.reactions, strict=True):
        electrons = reaction.reactant_counts.get(ELECTRON.name, 0)
        if not electrons or reaction.kind in RADIATIVE_KINDS:
            continue
        heavy = [s.name for s in reaction.reactants if s.species != ELECTRON]
        if len(heavy) != 1:
            continue
        total += k * ne ** (electrons - 1) * share[heavy[0]] * reaction.energy_change
    return total


def quantity(result: Any, name: str) -> float:
    """Return the quantity ``name`` of a steady state (see the module docstring)."""
    kind, _, rest = name.partition(":")
    if kind == "ratio":
        top, _, bottom = rest.partition("/")
        return quantity(result, top) / quantity(result, bottom)
    if kind == "population":
        return float(result.population(rest))
    if kind == "effective_rate":
        source, _, target = rest.partition(">")
        return float(result.reduction.effective_rate(source, target))
    if kind == "emissivity":
        return float(result.fluxes[rest])
    if kind == "electron_cooling":
        return electron_cooling(result, rest)
    check_quantity(result.model, name)
    raise AssertionError("unreachable")  # pragma: no cover


def quantities(result: Any, names: Sequence[str]) -> np.ndarray:
    """Return the quantities ``names`` of a steady state as an array."""
    return np.array([quantity(result, n) for n in names])
