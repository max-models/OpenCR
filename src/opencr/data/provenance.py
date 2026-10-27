"""Where a piece of data comes from, how it was changed and what it may be used for.

Every reaction, and every dataset of a model, carries a :class:`Provenance`:
the reference, the persistent identifier, the version and the retrieval date,
the units and the conversions applied, the range of validity, the
interpolation and extrapolation, the uncertainty and the licence under which
it may be redistributed. :func:`opencr.data.checks.check_provenance` tells
what is missing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["DERIVED", "LICENCES_REDISTRIBUTABLE", "Provenance", "Reference", "Validity"]

#: Licences under which OpenCR may ship a dataset. Anything else is read from
#: the user's own files and never bundled.
LICENCES_REDISTRIBUTABLE = (
    "CC0-1.0",
    "CC-BY-4.0",
    "MIT",
    "public-domain",
    "published-fit",
)


@dataclass(frozen=True)
class Reference:
    """A publication or database.

    Attributes:
        citation: The full citation, e.g. ``"L. C. Johnson, Astrophys. J. 174, 227 (1972)"``.
        doi: The DOI without the ``https://doi.org/`` prefix, if any.
        url: A URL where the source is openly available, if any.
        detail: The equation, table or section used, e.g. ``"eq. (14), table 3"``.
    """

    citation: str
    doi: str | None = None
    url: str | None = None
    detail: str = ""

    def __str__(self) -> str:
        """The citation, the detail and the DOI in one line."""
        text = self.citation
        if self.detail:
            text += f", {self.detail}"
        if self.doi:
            text += f", doi:{self.doi}"
        return text


@dataclass(frozen=True)
class Validity:
    """The range of a quantity in which data may be used.

    Attributes:
        quantity: What the range is of, e.g. ``"electron_temperature"`` or ``"energy"``.
        minimum: The lowest valid value, in ``unit``.
        maximum: The highest valid value, in ``unit``.
        unit: The unit of the bounds, e.g. ``"eV"``.
    """

    quantity: str
    minimum: float = 0.0
    maximum: float = float("inf")
    unit: str = "eV"

    def __post_init__(self) -> None:
        """Check the bounds."""
        if not self.minimum <= self.maximum:
            raise ValueError(f"validity of {self.quantity}: minimum > maximum")

    def contains(self, value: float) -> bool:
        """Return whether ``value`` (in ``unit``) is inside the range."""
        return self.minimum <= value <= self.maximum


@dataclass(frozen=True)
class Provenance:
    """The origin and the conditions of use of one piece of data.

    Attributes:
        reference: The source.
        version: The version of the database or dataset, if it has one.
        retrieved: The date the data were obtained, ISO 8601 (``"2026-10-27"``).
        units: The units of the data as stored, e.g. ``"m^2 against eV"``.
        conversions: The conversions applied to the source, in order, e.g.
            ``("1e-16 cm^2 -> m^2",)``.
        validity: The ranges of validity.
        interpolation: How values between data points are obtained.
        extrapolation: What happens outside the data or the validity.
        uncertainty: The relative uncertainty (a fraction, e.g. 0.2), or ``None``
            if unknown.
        uncertainty_note: The uncertainty as the source states it.
        licence: The licence of the data; one of :data:`LICENCES_REDISTRIBUTABLE`
            for data OpenCR may ship, otherwise its name or ``"unknown"``.
        notes: Anything else, e.g. a correction of a misprint.
    """

    reference: Reference
    version: str = ""
    retrieved: str = ""
    units: str = ""
    conversions: tuple[str, ...] = ()
    validity: tuple[Validity, ...] = ()
    interpolation: str = ""
    extrapolation: str = ""
    uncertainty: float | None = None
    uncertainty_note: str = ""
    licence: str = "unknown"
    notes: str = ""

    def __post_init__(self) -> None:
        """Normalise the sequences to tuples and check the uncertainty."""
        object.__setattr__(self, "conversions", tuple(self.conversions))
        object.__setattr__(self, "validity", tuple(self.validity))
        if self.uncertainty is not None and self.uncertainty < 0.0:
            raise ValueError("uncertainty must be >= 0")

    @property
    def redistributable(self) -> bool:
        """Whether the licence allows OpenCR to ship the data."""
        return self.licence in LICENCES_REDISTRIBUTABLE

    def to_dict(self) -> dict[str, Any]:
        """Return the provenance as plain JSON-compatible data."""
        data = asdict(self)
        for item in data["validity"]:
            if item["maximum"] == float("inf"):
                item["maximum"] = None
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Provenance:
        """Return the provenance of :meth:`to_dict`'s output."""
        data = dict(data)
        reference = Reference(**data.pop("reference"))
        validity = tuple(
            Validity(
                **{
                    **item,
                    "maximum": float("inf")
                    if item.get("maximum") is None
                    else item["maximum"],
                }
            )
            for item in data.pop("validity", ())
        )
        data["conversions"] = tuple(data.get("conversions", ()))
        return cls(reference=reference, validity=validity, **data)


#: A provenance for values derived inside OpenCR (detailed balance, sums).
DERIVED = Provenance(
    Reference("derived in OpenCR"),
    licence="MIT",
    notes="computed from other data of the model",
)
