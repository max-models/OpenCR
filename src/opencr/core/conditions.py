"""The plasma conditions of a solve.

:class:`Conditions` holds everything a solve takes from outside the model:
the electron temperature (and optionally a non-Maxwellian energy
distribution), the electron density or quasi-neutrality, the densities of the
prescribed states (the drivers and backgrounds), the element totals of a
closed steady state, the heavy-particle temperatures and the radiation
trapping (escape factors given per line, or a :class:`RadiationTrapping`
configuration they are computed from). See the guide "The CR model" for how
a solve uses them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

__all__ = ["ESCAPE_MODELS", "Conditions", "RadiationTrapping"]

#: The escape-factor approximations of :mod:`opencr.kinetics.radiation`.
ESCAPE_MODELS = ("mewe", "holstein-slab", "holstein-cylinder")


@dataclass(frozen=True)
class RadiationTrapping:
    """The escape-factor configuration of a solve (see :mod:`opencr.kinetics.radiation`).

    Attributes:
        length: The absorption length ``L`` in m (the distance to the wall).
        model: One of :data:`ESCAPE_MODELS`.
        lines: The names of the ``spontaneous_emission`` reactions to trap, or
            ``None`` for all of them.
        tolerance: The self-consistent iteration stops when no escape factor
            changes by more than this.
        max_iterations: The limit of that iteration.
    """

    length: float
    model: str = "mewe"
    lines: tuple[str, ...] | None = None
    tolerance: float = 1.0e-6
    max_iterations: int = 100

    def __post_init__(self) -> None:
        """Check the length and the model."""
        if not self.length > 0.0:
            raise ValueError("length must be > 0 (m)")
        if self.model not in ESCAPE_MODELS:
            raise ValueError(f"model must be one of {ESCAPE_MODELS}")
        if self.lines is not None:
            object.__setattr__(self, "lines", tuple(self.lines))

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as plain data."""
        return {
            "length": self.length,
            "model": self.model,
            "lines": None if self.lines is None else list(self.lines),
        }


@dataclass(frozen=True)
class Conditions:
    """Plasma conditions, SI with temperatures in eV.

    Attributes:
        electron_temperature: ``kT_e`` in eV. With a non-Maxwellian ``eedf`` it is
            the temperature of the rates that are not cross sections and of
            detailed balance; :meth:`from_eedf` sets it to ``2/3 <eps>``.
        electron_density: ``n_e`` in m^-3, or ``None`` for quasi-neutrality,
            ``n_e = sum z_i n_i`` over every charged state (evolving and
            prescribed).
        eedf: The electron energy distribution (an
            :class:`opencr.kinetics.eedf.EEDF`), or ``None`` for a Maxwellian at
            ``electron_temperature``.
        densities: Densities in m^-3 of the prescribed states, by name: the
            drivers of a quasi-static solve (e.g. ``H(n=1)`` and ``H+``) and
            fixed backgrounds (e.g. ``H2`` in an atomic model).
        totals: For a closed steady state, the total density of each element in
            m^-3 over the evolving states, e.g. ``{"H": 1e19}``.
        ion_temperature: ``kT_i`` in eV; the gas temperature if ``None``.
        gas_temperature: ``kT`` of the neutral gas in eV (0.026 eV is 300 K).
        escape: Escape factors in [0, 1] by reaction name, overriding the
            reactions' own and anything ``trapping`` computes.
        trapping: Compute the escape factors of the lines self-consistently
            from the populations (:mod:`opencr.kinetics.radiation`); ``None``
            for an optically thin plasma (apart from ``escape``).
    """

    electron_temperature: float
    electron_density: float | None = None
    eedf: Any = None
    densities: Mapping[str, float] = field(default_factory=dict)
    totals: Mapping[str, float] = field(default_factory=dict)
    ion_temperature: float | None = None
    gas_temperature: float = 0.026
    escape: Mapping[str, float] = field(default_factory=dict)
    trapping: RadiationTrapping | None = None

    def __post_init__(self) -> None:
        """Check the values and freeze the mappings."""
        if not self.electron_temperature > 0.0:
            raise ValueError("electron_temperature must be > 0 (eV)")
        if self.electron_density is not None and not self.electron_density > 0.0:
            raise ValueError("electron_density must be > 0 (m^-3) or None")
        if not self.gas_temperature > 0.0:
            raise ValueError("gas_temperature must be > 0 (eV)")
        if self.ion_temperature is not None and not self.ion_temperature > 0.0:
            raise ValueError("ion_temperature must be > 0 (eV) or None")
        for name, value in (*self.densities.items(), *self.totals.items()):
            if not value >= 0.0:
                raise ValueError(f"density of {name!r} must be >= 0")
        for name, value in self.escape.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"escape factor of {name!r} must be in [0, 1]")
        if self.trapping is not None and not isinstance(
            self.trapping, RadiationTrapping
        ):
            raise TypeError("trapping must be a RadiationTrapping or None")
        for key in ("densities", "totals", "escape"):
            object.__setattr__(
                self,
                key,
                MappingProxyType({k: float(v) for k, v in getattr(self, key).items()}),
            )

    @classmethod
    def from_eedf(cls, eedf: Any, **kwargs: Any) -> Conditions:
        """Return conditions with ``eedf`` and its effective temperature ``2/3 <eps>``."""
        return cls(electron_temperature=eedf.effective_temperature, eedf=eedf, **kwargs)

    @property
    def quasi_neutral(self) -> bool:
        """Whether ``n_e`` follows from quasi-neutrality."""
        return self.electron_density is None

    @property
    def heavy_temperature(self) -> dict[str, float]:
        """The temperature of each kind of rate dependence (see :mod:`opencr.core.rates`)."""
        ion = (
            self.ion_temperature
            if self.ion_temperature is not None
            else self.gas_temperature
        )
        return {
            "electron": self.electron_temperature,
            "ion": ion,
            "gas": self.gas_temperature,
            "none": 1.0,
        }

    def replace(self, **changes: Any) -> Conditions:
        """Return a copy with ``changes`` applied."""
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return the conditions as plain data (the EEDF by its description)."""
        return {
            "electron_temperature": self.electron_temperature,
            "electron_density": self.electron_density,
            "eedf": None if self.eedf is None else repr(self.eedf),
            "densities": dict(self.densities),
            "totals": dict(self.totals),
            "ion_temperature": self.ion_temperature,
            "gas_temperature": self.gas_temperature,
            "escape": dict(self.escape),
            "trapping": None if self.trapping is None else self.trapping.to_dict(),
        }
