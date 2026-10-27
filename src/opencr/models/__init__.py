"""Predefined models, loaded by name with :func:`load_model`.

======================================  ===================================================
name                                    model
======================================  ===================================================
``hydrogen/atomic/basic``               H(n=1..n_max) and H+ (:mod:`~opencr.models.hydrogen_atomic`)
======================================  ===================================================

Options are passed on to the model's builder, e.g.
``load_model("hydrogen/atomic/basic", n_max=30)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opencr.model import Model

__all__ = ["available_models", "load_model", "register_model"]

_REGISTRY: dict[str, tuple[Callable[..., Model], str]] = {}


def register_model(name: str, builder: Callable[..., Model], description: str) -> None:
    """Make ``builder(**options)`` loadable as ``name``."""
    _REGISTRY[name] = (builder, description)


def available_models() -> dict[str, str]:
    """Return the names of the predefined models with one sentence each."""
    return {name: description for name, (_, description) in sorted(_REGISTRY.items())}


def load_model(name: str, **options: Any) -> Model:
    """Return the predefined model ``name`` built with ``options``."""
    if name not in _REGISTRY:
        raise KeyError(f"no model {name!r}; available: {sorted(_REGISTRY)}")
    builder, _ = _REGISTRY[name]
    return builder(**options)


def _atomic_hydrogen(isotope: str = "H", **options: Any) -> Model:
    from opencr.models.hydrogen_atomic import atomic_hydrogen_network

    network = atomic_hydrogen_network(isotope=isotope, **options)
    return Model(
        network,
        drivers=(f"{isotope}(n=1)", f"{isotope}+"),
        description="Atomic hydrogen resolved in n: electron-impact excitation, "
        "ionization, de-excitation and recombination, and spontaneous emission.",
    )


register_model(
    "hydrogen/atomic/basic",
    _atomic_hydrogen,
    "H(n=1..n_max) and H+ with electron collisions, recombination and emission",
)
