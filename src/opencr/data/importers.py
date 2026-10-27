"""Importers for published data formats.

* :func:`lxcat_reactions`: electron-impact processes from an LXCat file
  (read by :func:`plasmacoll.read_lxcat`), as reactions between states of a
  network. LXCat's terms of use do not allow redistribution: the files stay
  with the user and OpenCR never bundles them.
* :func:`read_rate_table`: a two-column text file of temperature (eV) and rate
  coefficient (SI), as a :class:`~opencr.core.rates.Tabulated` rate model.
* :func:`read_adf11`: an ADAS adf11 file of effective (collisional-radiative)
  coefficients, such as OPEN-ADAS ``scd12_h.dat`` and ``acd12_h.dat``, for
  comparisons. OPEN-ADAS data are for personal use and must not be
  redistributed: OpenCR reads the user's own downloads and never ships them.

Every importer checks the data (:mod:`opencr.data.checks`) and attaches a
provenance.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from plasmacoll import read_lxcat

from opencr.core.rates import CrossSectionRate, Tabulated
from opencr.core.reactions import Reaction
from opencr.core.species import ELECTRON_STATE, State
from opencr.data.checks import check_cross_section, check_rate_table, raise_problems
from opencr.data.provenance import Provenance, Reference

__all__ = ["LXCAT_KINDS", "Adf11", "lxcat_reactions", "read_adf11", "read_rate_table"]

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


@dataclass(frozen=True)
class Adf11:
    """Effective coefficients of one charge state from an ADAS adf11 file.

    Attributes:
        density: The electron densities in m^-3.
        temperature: The electron temperatures in eV.
        coefficient: The coefficients in m^3/s, shape ``(temperature, density)``.
        header: The first line of the file.
    """

    density: np.ndarray
    temperature: np.ndarray
    coefficient: np.ndarray
    header: str

    def __call__(self, temperature: float, density: float) -> float:
        """Return the coefficient interpolated bilinearly in log-log (m^3/s)."""
        from scipy.interpolate import RegularGridInterpolator

        interpolate = RegularGridInterpolator(
            (np.log(self.temperature), np.log(self.density)), np.log(self.coefficient)
        )
        return float(np.exp(interpolate([[np.log(temperature), np.log(density)]])[0]))


def read_adf11(path: str | Path, block: int = 0) -> Adf11:
    """Read one block (charge state) of an unresolved ADAS adf11 file.

    The file gives ``log10`` densities in cm^-3, ``log10`` temperatures in eV and
    ``log10`` coefficients in cm^3/s; they are returned in SI.
    """
    lines = Path(path).read_text().splitlines()
    sizes = lines[0].split()
    n_density, n_temperature = int(sizes[1]), int(sizes[2])
    numbers: list[float] = []
    index = 2
    while len(numbers) < n_density + n_temperature:
        numbers += [float(v) for v in lines[index].split()]
        index += 1
    starts = [i for i, line in enumerate(lines) if "Z1=" in line]
    if block >= len(starts):
        raise_problems([f"{path}: no block {block}"])
    values: list[float] = []
    index = starts[block] + 1
    while len(values) < n_density * n_temperature:
        values += [float(v) for v in lines[index].split()]
        index += 1
    table = np.asarray(values).reshape(n_temperature, n_density)
    return Adf11(
        density=10.0 ** np.asarray(numbers[:n_density]) * 1.0e6,
        temperature=10.0 ** np.asarray(numbers[n_density:]),
        coefficient=10.0**table * 1.0e-6,
        header=lines[0].strip(),
    )
