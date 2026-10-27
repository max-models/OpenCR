"""Importers for published data formats.

* :func:`lxcat_reactions`: electron-impact processes from an LXCat file
  (read by :func:`plasmacoll.read_lxcat`), as reactions between states of a
  network. LXCat's terms of use do not allow redistribution: the files stay
  with the user and OpenCR never bundles them.
* :func:`read_rate_table`: a two-column text file of temperature (eV) and rate
  coefficient (SI), as a :class:`~opencr.core.rates.Tabulated` rate model.

Every importer checks the data (:mod:`opencr.data.checks`) and attaches a
provenance.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path

import numpy as np
from plasmacoll import read_lxcat

from opencr.core.rates import CrossSectionRate, Tabulated
from opencr.core.reactions import Reaction
from opencr.core.species import ELECTRON_STATE, State
from opencr.data.checks import check_cross_section, check_rate_table, raise_problems
from opencr.data.provenance import Provenance, Reference

__all__ = ["LXCAT_KINDS", "lxcat_reactions", "read_rate_table"]

#: The OpenCR reaction kind of each LXCat block kind that changes a state.
LXCAT_KINDS = {
    "EXCITATION": "excitation",
    "IONIZATION": "ionization",
    "ATTACHMENT": "attachment",
}


def lxcat_reactions(
    path: str | Path,
    target: State,
    products: Callable[[str, str], tuple[State, ...] | None]
    | Mapping[str, tuple[State, ...]],
    database: str,
    retrieved: str | None = None,
) -> list[Reaction]:
    """Return the excitation, ionization and attachment processes of an LXCat file.

    Elastic and effective blocks change no population and are skipped.

    Args:
        path: The LXCat file.
        target: The target state of every process (e.g. ``Ar`` ground state).
        products: The product states of a block, by the block's product text
            (``"Ar*(11.55eV)"``) or as a function of ``(kind, product)``;
            the electron is added. A block whose products are ``None`` or absent
            is skipped.
        database: The database and its reference as the user's download cites
            it, e.g. ``"Biagi-v7.1, www.lxcat.net, retrieved 2026-10-27"``.
        retrieved: The download date (ISO 8601); today if not given.

    Returns:
        The reactions, named ``"<kind>:<target>-><products>"``.
    """
    reactions = []
    for process in read_lxcat(path):
        kind = LXCAT_KINDS.get(process.kind)
        if kind is None:
            continue
        if callable(products):
            states = products(process.kind, process.product)
        else:
            states = products.get(process.product)
        if states is None:
            continue
        cs = process.cross_section
        raise_problems(
            check_cross_section(cs.energy, cs.sigma, cs.threshold, process.product)
        )
        electrons = {"excitation": 1, "ionization": 2, "attachment": 0}[kind]
        provenance = Provenance(
            Reference(database, detail=process.comments.get("PROCESS", "")),
            retrieved=retrieved or date.today().isoformat(),
            units="m^2 against eV (incident electron energy)",
            interpolation="linear in sigma and energy",
            extrapolation="the last value above the table, zero below the threshold",
            licence="LXCat terms of use (no redistribution)",
        )
        name = f"{kind}:{target.name}->" + "+".join(s.name for s in states)
        reactions.append(
            Reaction(
                name,
                (ELECTRON_STATE, target),
                (ELECTRON_STATE,) * electrons + tuple(states),
                kind,
                CrossSectionRate(cross_section=cs),
                provenance,
            )
        )
    return reactions


def read_rate_table(
    path: str | Path,
    reference: Reference,
    depends_on: str = "electron",
    threshold: float = 0.0,
    unit_factor: float = 1.0,
    licence: str = "unknown",
) -> tuple[Tabulated, Provenance]:
    """Read a two-column table of temperature (eV) and rate coefficient.

    Lines starting with ``#`` are comments. ``unit_factor`` converts the
    rates to SI (e.g. 1e-6 for cm^3/s) and is recorded in the provenance.

    Returns:
        The rate model and its provenance.
    """
    table = np.loadtxt(path, comments="#", ndmin=2)
    if table.shape[1] != 2:
        raise_problems([f"{path}: expected two columns, found {table.shape[1]}"])
    temperatures, values = table[:, 0], table[:, 1] * unit_factor
    raise_problems(check_rate_table(temperatures, values, str(path)))
    conversions = (f"rates times {unit_factor:g} to SI",) if unit_factor != 1.0 else ()
    provenance = Provenance(
        reference,
        units="SI against eV",
        conversions=conversions,
        interpolation="linear in log k against log T",
        extrapolation="the end values outside the table",
        licence=licence,
    )
    rate = Tabulated(
        depends_on=depends_on,
        temperatures=tuple(temperatures),
        values=tuple(values),
        threshold=threshold,
    )
    return rate, provenance
