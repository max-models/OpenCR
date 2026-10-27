"""How the rate coefficient of a reaction is given.

A rate model describes the data, not how they are evaluated: the evaluation
(Maxwellian or any electron energy distribution, on any backend) is in
:mod:`opencr.kinetics.rates`. The rate of a reaction with reactants ``r_i``
is ``k prod n_i`` (a reactant that appears twice counts twice), so ``k`` is in
s^-1 for one reactant, m^3/s for two and m^6/s for three.

``depends_on`` names the temperature a model depends on: ``"electron"``
(``T_e``, or for a non-Maxwellian distribution the effective temperature
``2/3 <eps>``), ``"ion"``, ``"gas"`` or ``"none"``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from plasmacoll import CrossSection

from opencr.constants import ELECTRON_MASS

__all__ = [
    "TEMPERATURES",
    "Arrhenius",
    "Constant",
    "CrossSectionRate",
    "DetailedBalance",
    "Function",
    "RateModel",
    "Tabulated",
]

#: The temperatures a rate model can depend on.
TEMPERATURES = ("electron", "ion", "gas", "none")


@dataclass(frozen=True)
class RateModel:
    """Base class of the rate models; ``kind`` names the model in files."""

    kind: ClassVar[str] = ""
    depends_on: str = "electron"

    def __post_init__(self) -> None:
        """Check ``depends_on``."""
        if self.depends_on not in TEMPERATURES:
            raise ValueError(f"depends_on must be one of {TEMPERATURES}")

    def to_dict(self) -> dict[str, Any]:
        """Return the model as plain data (see :mod:`opencr.data.native`)."""
        raise NotImplementedError(f"{type(self).__name__} cannot be written to a file")


@dataclass(frozen=True)
class CrossSectionRate(RateModel):
    """A rate from a cross section averaged over a velocity distribution.

    For electrons (``depends_on="electron"``) the average is over the electron
    energy distribution of the conditions (Maxwellian by default); otherwise over
    a Maxwellian of relative velocities at the ion or gas temperature.

    Attributes:
        cross_section: The cross section (m^2) against energy (eV).
        mass: The reduced mass in kg of the colliding pair.
        incident_mass: For a table against the incident particle's energy with
            the target at rest (the lab frame), the incident mass in kg; the
            table is looked up at ``E incident_mass / mass``. ``None`` for a
            table in the centre-of-mass frame. For electrons both agree.
    """

    kind: ClassVar[str] = "cross_section"
    cross_section: CrossSection | None = None
    mass: float = ELECTRON_MASS
    incident_mass: float | None = None

    def __post_init__(self) -> None:
        """Check the masses and the cross section."""
        super().__post_init__()
        if not isinstance(self.cross_section, CrossSection):
            raise TypeError("cross_section must be a plasmacoll.CrossSection")
        if self.mass <= 0.0 or (
            self.incident_mass is not None and self.incident_mass <= 0.0
        ):
            raise ValueError("masses must be > 0")
        if self.depends_on == "none":
            raise ValueError("a cross-section rate depends on a temperature")

    @property
    def table(self) -> CrossSection:
        """The cross section (never ``None`` after construction)."""
        assert self.cross_section is not None
        return self.cross_section

    @property
    def threshold(self) -> float:
        """The threshold of the cross section in eV (in its own frame)."""
        return float(self.table.threshold)

    def to_dict(self) -> dict[str, Any]:
        """Return the table, threshold and masses."""
        import cunumpy as xp

        return {
            "kind": self.kind,
            "depends_on": self.depends_on,
            "energy": np.asarray(xp.to_numpy(self.table.energy)).tolist(),
            "sigma": np.asarray(xp.to_numpy(self.table.sigma)).tolist(),
            "threshold": self.threshold,
            "label": self.table.label,
            "mass": self.mass,
            "incident_mass": self.incident_mass,
        }


@dataclass(frozen=True)
class Arrhenius(RateModel):
    """``k = a T^b exp(-c / T)`` with ``T`` in eV and ``k`` in SI.

    Attributes:
        a: The prefactor in the SI unit of the reaction's order.
        b: The temperature exponent.
        c: The activation energy in eV.
    """

    kind: ClassVar[str] = "arrhenius"
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0

    def __post_init__(self) -> None:
        """Check the prefactor."""
        super().__post_init__()
        if self.a < 0.0:
            raise ValueError("the Arrhenius prefactor must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        """Return the three parameters."""
        return {
            "kind": self.kind,
            "depends_on": self.depends_on,
            "a": self.a,
            "b": self.b,
            "c": self.c,
        }


@dataclass(frozen=True)
class Tabulated(RateModel):
    """``k(T)`` from a table, interpolated linearly in ``log k`` against ``log T``.

    Outside the table the end values hold. A ``threshold`` (eV) removes the
    Boltzmann factor before interpolating, ``log(k exp(threshold / T))``,
    which keeps the interpolation accurate at temperatures far below a
    threshold.

    Attributes:
        temperatures: The table temperatures in eV, increasing.
        values: ``k`` in SI at ``temperatures``, non-negative.
        threshold: The energy in eV of the Boltzmann factor to remove.
    """

    kind: ClassVar[str] = "tabulated"
    temperatures: tuple[float, ...] = ()
    values: tuple[float, ...] = ()
    threshold: float = 0.0

    def __post_init__(self) -> None:
        """Check and freeze the table."""
        super().__post_init__()
        t = np.asarray(self.temperatures, dtype=float)
        k = np.asarray(self.values, dtype=float)
        if t.ndim != 1 or t.shape != k.shape or t.size < 2:
            raise ValueError("a rate table needs two 1D arrays of equal length >= 2")
        if np.any(np.diff(t) <= 0.0) or np.any(t <= 0.0):
            raise ValueError("table temperatures must be positive and increasing")
        if np.any(k < 0.0) or not np.all(np.isfinite(k)):
            raise ValueError("table rates must be finite and >= 0")
        object.__setattr__(self, "temperatures", tuple(t.tolist()))
        object.__setattr__(self, "values", tuple(k.tolist()))

    def to_dict(self) -> dict[str, Any]:
        """Return the table."""
        return {
            "kind": self.kind,
            "depends_on": self.depends_on,
            "temperatures": list(self.temperatures),
            "values": list(self.values),
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class Constant(RateModel):
    """A rate coefficient independent of temperature, e.g. an Einstein A in s^-1.

    Attributes:
        value: ``k`` in the SI unit of the reaction's order.
    """

    kind: ClassVar[str] = "constant"
    depends_on: str = "none"
    value: float = 0.0

    def __post_init__(self) -> None:
        """Check the value."""
        super().__post_init__()
        if not self.value >= 0.0:
            raise ValueError("a constant rate must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        """Return the value."""
        return {"kind": self.kind, "depends_on": self.depends_on, "value": self.value}


@dataclass(frozen=True)
class Function(RateModel):
    """``k(T)`` from a Python function of the temperature in eV (array in, array out).

    Written to files as a table on :attr:`file_temperatures`.

    Attributes:
        function: ``k`` in SI as a function of ``T`` in eV.
        label: What the function is, e.g. ``"Kramers radiative recombination"``.
        threshold: As for :class:`Tabulated`, used when the function is tabulated.
        file_temperatures: The temperatures (eV) of the table written to files.
    """

    kind: ClassVar[str] = "function"
    function: Callable[[np.ndarray], np.ndarray] | None = None
    label: str = ""
    threshold: float = 0.0
    file_temperatures: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.geomspace(0.01, 1.0e4, 241).tolist())
    )

    def __post_init__(self) -> None:
        """Check that a function is given."""
        super().__post_init__()
        if not callable(self.function):
            raise TypeError("function must be callable")

    def tabulate(self) -> Tabulated:
        """Return the function as a :class:`Tabulated` model on :attr:`file_temperatures`."""
        assert self.function is not None
        t = np.asarray(self.file_temperatures, dtype=float)
        k = np.asarray(self.function(t), dtype=float)
        return Tabulated(
            depends_on=self.depends_on,
            temperatures=tuple(t),
            values=tuple(k),
            threshold=self.threshold,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the function tabulated (see :meth:`tabulate`)."""
        data = self.tabulate().to_dict()
        data["label"] = self.label
        return data


@dataclass(frozen=True)
class DetailedBalance(RateModel):
    """The inverse of another reaction of the network, by detailed balance.

    De-excitation from excitation, three-body recombination from ionization,
    and any other inverse whose forward reaction has the products as
    reactants and an electron on both sides or one more on the forward
    product side. The evaluation (:mod:`opencr.kinetics.rates`) uses the
    state energies and weights: the Klein-Rosseland inverse cross section for
    an electron-impact excitation (valid for any energy distribution), and
    the Maxwellian relation (Boltzmann, Saha) otherwise.

    Attributes:
        forward: The name of the forward reaction.
    """

    kind: ClassVar[str] = "detailed_balance"
    forward: str = ""

    def __post_init__(self) -> None:
        """Check that the forward reaction is named."""
        super().__post_init__()
        if not self.forward:
            raise ValueError("DetailedBalance needs the name of the forward reaction")

    def to_dict(self) -> dict[str, Any]:
        """Return the forward reaction's name."""
        return {
            "kind": self.kind,
            "depends_on": self.depends_on,
            "forward": self.forward,
        }
