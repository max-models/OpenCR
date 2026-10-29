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

__all__ = ["KINDS", "check_quantity", "quantities", "quantity"]

#: The kinds of quantity.
KINDS = ("population", "effective_rate", "emissivity", "ratio")


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
    elif kind == "population":
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
    check_quantity(result.model, name)
    raise AssertionError("unreachable")  # pragma: no cover


def quantities(result: Any, names: Sequence[str]) -> np.ndarray:
    """Return the quantities ``names`` of a steady state as an array."""
    return np.array([quantity(result, n) for n in names])
