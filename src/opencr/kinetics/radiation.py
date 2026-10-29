"""Radiation trapping as an escape-factor approximation.

An optically thick line ``u -> l`` is re-absorbed on its way out of the
plasma, so only a fraction of the spontaneous emission leaves: the effective
transition probability is ``g A_ul`` with the escape factor ``g`` in [0, 1].
OpenCR applies escape factors to the ``spontaneous_emission`` reactions, from
the conditions (``Conditions.escape``, per line) or computed here from the
line-centre optical depth of a Doppler-broadened line over a length ``L``,

    tau_0 = kappa_0 L,   kappa_0 = lambda_0^3 g_u A_ul n_l / (8 pi^(3/2) g_l v_D),

with ``v_D = (2 kT / m)^(1/2)`` the emitters' thermal speed at the gas
temperature (Mihalas 1978, eq. 9-6 with a Doppler profile). The escape factor
is then one of the usual approximations (:data:`ESCAPE_MODELS`):

* ``"mewe"`` (default): ``g = (2 - exp(-tau_0 / 1000)) / (1 + tau_0)`` (Mewe
  1967, Brit. J. Appl. Phys. 18, 107), an interpolation that is right in both
  limits and is what most CR codes use (Yacora, Behringer's models);
* ``"holstein-slab"``: Holstein's asymptotic result for a Doppler profile,
  ``g = 1 / (tau_0 (pi ln tau_0)^(1/2))``, for an infinite slab of
  half-thickness ``L`` (Holstein 1947, Phys. Rev. 72, 1212);
* ``"holstein-cylinder"``: the same for an infinite cylinder of radius ``L``,
  ``g = 1.60 / (tau_0 (pi ln tau_0)^(1/2))``.

Holstein's forms hold for ``tau_0 >> 1``; below ``tau_0 = e`` they are set
to 1. ``L`` is the characteristic absorption length (the distance to the
wall, not the full plasma size), to be chosen by the user; this is a
zero-dimensional approximation of a transport problem and is explicit about
it: no escape factor is applied unless a :class:`RadiationTrapping` is given.

A self-consistent solve iterates: the escape factors depend on the lower
populations, which depend on the escape factors
(:meth:`opencr.model.Model.solve_steady_state`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from opencr.constants import ELEMENTARY_CHARGE, photon_wavelength
from opencr.core.conditions import ESCAPE_MODELS, RadiationTrapping
from opencr.core.network import ReactionNetwork

__all__ = [
    "ESCAPE_MODELS",
    "RadiationTrapping",
    "escape_factor",
    "escape_factors",
    "line_centre_optical_depth",
]


def line_centre_optical_depth(
    wavelength: float | np.ndarray,
    einstein_a: float | np.ndarray,
    weight_ratio: float | np.ndarray,
    lower_density: float | np.ndarray,
    length: float,
    temperature: float,
    mass: float,
) -> np.ndarray:
    """Return ``tau_0`` of a Doppler-broadened line over ``length``.

    Args:
        wavelength: The line's vacuum wavelength in nm.
        einstein_a: ``A_ul`` in s^-1 (without any escape factor).
        weight_ratio: ``g_u / g_l``.
        lower_density: ``n_l`` in m^-3.
        length: The absorption length ``L`` in m.
        temperature: The emitters' ``kT`` in eV.
        mass: The emitters' mass in kg.
    """
    speed = math.sqrt(2.0 * temperature * ELEMENTARY_CHARGE / mass)
    lam = np.asarray(wavelength, dtype=float) * 1.0e-9
    kappa = (
        lam**3
        * np.asarray(weight_ratio, dtype=float)
        * np.asarray(einstein_a, dtype=float)
        * np.asarray(lower_density, dtype=float)
        / (8.0 * math.pi**1.5 * speed)
    )
    return kappa * length


def escape_factor(optical_depth: float | np.ndarray, model: str = "mewe") -> np.ndarray:
    """Return the escape factor ``g(tau_0)`` of one of :data:`ESCAPE_MODELS`."""
    tau = np.asarray(optical_depth, dtype=float)
    if np.any(tau < 0.0):
        raise ValueError("the optical depth must be >= 0")
    if model == "mewe":
        return (2.0 - np.exp(-tau / 1000.0)) / (1.0 + tau)
    if model in ("holstein-slab", "holstein-cylinder"):
        factor = 1.0 if model == "holstein-slab" else 1.60
        thick = tau > math.e
        safe = np.where(thick, tau, math.e)
        g = factor / (safe * np.sqrt(math.pi * np.log(safe)))
        return np.where(thick, np.minimum(g, 1.0), 1.0)
    raise ValueError(f"model must be one of {ESCAPE_MODELS}")


def escape_factors(
    network: ReactionNetwork,
    populations: Mapping[str, float],
    trapping: RadiationTrapping,
    temperature: float,
) -> dict[str, float]:
    """Return the escape factor of every trapped line, by reaction name.

    Args:
        network: The network.
        populations: The density of every state in m^-3.
        trapping: The configuration.
        temperature: The gas (emitter) temperature in eV.
    """
    out: dict[str, float] = {}
    wanted = None if trapping.lines is None else set(trapping.lines)
    if wanted is not None:
        for name in wanted:
            if network.reaction(name).kind != "spontaneous_emission":
                raise ValueError(f"{name} is not a spontaneous emission")
    for reaction in network.reactions:
        if reaction.kind != "spontaneous_emission":
            continue
        if wanted is not None and reaction.name not in wanted:
            continue
        upper, lower = reaction.reactants[0], reaction.products[0]
        energy = reaction.photon_energy
        assert energy is not None
        a = getattr(reaction.rate, "value", None)
        if a is None:
            raise ValueError(
                f"{reaction.name}: trapping needs a constant Einstein coefficient"
            )
        tau = line_centre_optical_depth(
            photon_wavelength(energy),
            a,
            upper.weight / lower.weight,
            populations[lower.name],
            trapping.length,
            temperature,
            upper.species.mass,
        )
        out[reaction.name] = float(escape_factor(tau, trapping.model))
    return out
