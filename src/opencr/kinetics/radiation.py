r"""Radiation trapping as an escape-factor approximation.

An optically thick line ``u -> l`` is re-absorbed on its way out of the
plasma, so only a fraction of the spontaneous emission leaves: the effective
transition probability is ``g A_ul`` with the escape factor ``g`` in [0, 1].
OpenCR applies escape factors to the ``spontaneous_emission`` reactions, from
the conditions (``Conditions.escape``, per line) or computed here from the
line-centre optical depth of a Doppler-broadened line over a length ``L``,

$$
\tau_0 = \kappa_0 L, \qquad
\kappa_0 = \frac{\lambda_0^3}{8\pi^{3/2}}\,\frac{g_u}{g_l}\,\frac{A_{ul}\, n_l}{v_D},
$$

with ``v_D = (2 kT / m)^(1/2)`` the emitters' thermal speed at the gas
temperature (Mihalas 1978, eq. 9-6 with a Doppler profile). The escape factor
is then one of the usual approximations (`ESCAPE_MODELS`):

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
it: no escape factor is applied unless a `RadiationTrapping` is given.

A self-consistent solve iterates: the escape factors depend on the lower
populations, which depend on the escape factors
(`opencr.model.Model.solve_steady_state`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from methodcard import Rule, method

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


def _optical_depth_rules() -> dict:
    """The SymPy rule of `line_centre_optical_depth` (for the tests)."""
    import sympy

    names = "wavelength einstein_a weight_ratio lower_density length temperature mass"
    lam, a, ratio, n, length, t, m = sympy.symbols(names, positive=True)
    speed = sympy.sqrt(2 * t * sympy.Float(ELEMENTARY_CHARGE) / m)
    rule = (lam * sympy.Float(1e-9)) ** 3 * ratio * a * n * length
    rule /= 8 * sympy.pi ** sympy.Rational(3, 2) * speed
    ranges = {
        "wavelength": (100.0, 700.0),
        "einstein_a": (1e6, 1e9),
        "weight_ratio": (1.0, 5.0),
        "lower_density": (1e15, 1e19),
        "length": (0.01, 1.0),
        "temperature": (0.02, 1.0),
        "mass": (1.6e-27, 7e-26),
    }
    return {"optical_depth": Rule(rule, ranges=ranges)}


def _escape_factor_rules() -> dict:
    """The SymPy rules of `escape_factor` (for the tests)."""
    import sympy

    tau = sympy.Symbol("optical_depth", positive=True)
    holstein = 1 / (tau * sympy.sqrt(sympy.pi * sympy.log(tau)))

    def model(name: str):
        return lambda optical_depth: float(escape_factor(optical_depth, name))

    thick = {"optical_depth": (3.0, 1e5)}
    return {
        "mewe": Rule(
            (2 - sympy.exp(-tau / 1000)) / (1 + tau),
            implementation=model("mewe"),
            ranges={"optical_depth": (0.0, 1e4)},
        ),
        "holstein_slab": Rule(
            holstein, implementation=model("holstein-slab"), ranges=thick
        ),
        "holstein_cylinder": Rule(
            sympy.Float("1.60") * holstein,
            implementation=model("holstein-cylinder"),
            ranges=thick,
        ),
    }


@method(
    equations={
        "optical_depth": r"\tau_0 = \kappa_0 L, \qquad \kappa_0 = \frac{\lambda_0^3}{8\pi^{3/2}}\,\frac{g_u}{g_l}\,\frac{A_{ul}\, n_l}{v_D}, \qquad v_D = \sqrt{\frac{2kT}{m}}",
        "cross_section": r"\kappa_0 = \underbrace{\frac{\lambda_0^2}{8\pi}\,\frac{g_u}{g_l}\,A_{ul}}_{\text{integrated absorption}}\ \underbrace{\frac{\lambda_0}{\sqrt\pi\,v_D}}_{\phi(0)}\ n_l",
    },
    assumptions=(
        r"a Doppler-broadened line at the emitters' temperature; no stimulated emission",
        r"$L$ the absorption length (the distance to the wall), chosen by the user",
    ),
    references=(
        "D. Mihalas, Stellar Atmospheres, 2nd ed., W. H. Freeman (1978), eq. 9-6",
    ),
    symbolic=_optical_depth_rules,
)
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


@method(
    equations={
        "mewe": r"g_{\mathrm{mewe}} = \frac{2 - e^{-\tau_0/1000}}{1 + \tau_0}",
        "holstein_slab": r"g_{\mathrm{slab}} = \frac{1}{\tau_0\sqrt{\pi\ln\tau_0}} \quad (\tau_0 > e), \qquad 1 \ \text{below}",
        "holstein_cylinder": r"g_{\mathrm{cyl}} = \frac{1.60}{\tau_0\sqrt{\pi\ln\tau_0}} \quad (\tau_0 > e), \qquad 1 \ \text{below}",
    },
    assumptions=(
        r"Mewe (the default): 1 at $\tau_0 = 0$, about $1/\tau_0$ for $1 \ll \tau_0 \ll 1000$, $2/\tau_0$ for $\tau_0 \gg 1000$",
        r"Holstein: asymptotic forms for $\tau_0 \gg 1$, an infinite slab of half-thickness $L$ or a cylinder of radius $L$; at $\tau_0 = e$ they jump from 1 to 0.207 (slab) and 0.332 (cylinder)",
    ),
    references=(
        "R. Mewe, Brit. J. Appl. Phys. 18, 107 (1967)",
        "[Holstein, Phys. Rev. 72, 1212 (1947)](https://doi.org/10.1103/PhysRev.72.1212)",
    ),
    symbolic=_escape_factor_rules,
)
def escape_factor(optical_depth: float | np.ndarray, model: str = "mewe") -> np.ndarray:
    """Return the escape factor ``g(tau_0)`` of one of `ESCAPE_MODELS`."""
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


@method(
    equations={
        "line": r"\lambda_0 = \frac{hc}{E_u - E_l}, \qquad g = g(\tau_0), \qquad \tau_0 = \tau_0(\lambda_0, A_{ul}, g_u/g_l, n_l, L, T_{\mathrm{gas}}, m)",
    },
    assumptions=(
        r"every `spontaneous_emission` of `RadiationTrapping.lines` (all by default)",
        r"$A_{ul}$ the untrapped `Constant` coefficient (another rate model raises); $m$ the emitter's mass",
    ),
    components=(line_centre_optical_depth, escape_factor),
)
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
