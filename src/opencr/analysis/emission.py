"""Line emission and synthetic spectra.

Each spontaneous emission ``u -> l`` of a network is a line of photon energy
``E_u - E_l`` with emissivity ``eps = n_u A_ul`` (photons m^-3 s^-1; the escape
factor is already in ``A``). In a quasi-static solve the upper population is a
sum over the drivers, ``n_u = sum_d C_ud n_d``, so the emissivity splits into
photon emissivity coefficients ``PEC_d = A_ul C_ud / n_e`` (m^3/s) with
``eps = n_e sum_d PEC_d n_d``.

A synthetic spectrum puts each line on a wavelength grid as a Gaussian of the
Doppler width at the gas temperature (or a given instrumental FWHM, or both
added in quadrature).
"""

from __future__ import annotations

import math

import numpy as np
import xarray as xr
from methodcard import Rule, method

from opencr.constants import (
    ATOMIC_MASS,
    ELEMENTARY_CHARGE,
    SPEED_OF_LIGHT,
    air_wavelength,
    photon_wavelength,
)
from opencr.core.network import ReactionNetwork

__all__ = ["doppler_fwhm", "hydrogen_line_name", "line_table", "spectrum"]

_SERIES = {1: "Ly", 2: "H", 3: "Pa", 4: "Br", 5: "Pf", 6: "Hu"}
_GREEK = "αβγδεζηθικλμ"


def hydrogen_line_name(upper: int, lower: int) -> str:
    """Return the spectroscopic name of an H line, e.g. ``"H-α"`` for 3 -> 2, ``"Ly-β"`` for 3 -> 1.

    Lines past the Greek alphabet's twelfth member, or of series beyond
    Humphreys, are named ``"<upper>-<lower>"``.
    """
    step = upper - lower
    if lower in _SERIES and 1 <= step <= len(_GREEK):
        return f"{_SERIES[lower]}-{_GREEK[step - 1]}"
    return f"{upper}-{lower}"


def _label(network: ReactionNetwork, name: str) -> str:
    reaction = network.reaction(name)
    upper, lower = reaction.reactants[0], reaction.products[0]
    n_u, n_l = upper.quantum_numbers.get("n"), lower.quantum_numbers.get("n")
    if (
        upper.species.composition.keys() <= {"H", "D"}
        and n_u
        and n_l
        and upper.charge == 0
    ):
        return hydrogen_line_name(n_u, n_l)
    return f"{upper.name}->{lower.name}"


def _doppler_rules() -> dict:
    """The SymPy rule of `doppler_fwhm` (for the tests)."""
    import sympy

    lam, t, m = sympy.symbols("wavelength temperature mass", positive=True)
    e, c = sympy.Float(ELEMENTARY_CHARGE), sympy.Float(SPEED_OF_LIGHT)
    rule = lam * 2 * sympy.sqrt(2 * sympy.log(2)) * sympy.sqrt(t * e / m) / c
    ranges = {
        "wavelength": (100.0, 1000.0),
        "temperature": (0.02, 10.0),
        "mass": (1.6e-27, 7e-26),
    }
    return {"fwhm": Rule(rule, ranges=ranges)}


@method(
    equations={
        "fwhm": r"\Delta\lambda_{\mathrm{FWHM}} = \lambda\, 2\sqrt{2\ln 2}\, \frac{\sigma_v}{c}, \qquad \sigma_v = \sqrt{\frac{kT}{m}}",
    },
    assumptions=(
        r"a Maxwellian of the emitters at $T$: $\sigma_v$ per velocity component",
    ),
    symbolic=_doppler_rules,
)
def doppler_fwhm(wavelength: np.ndarray, temperature: float, mass: float) -> np.ndarray:
    """Return the Doppler FWHM in nm of lines at ``wavelength`` (nm), ``kT`` in eV, mass in kg."""
    sigma_v = math.sqrt(temperature * ELEMENTARY_CHARGE / mass)
    return (
        np.asarray(wavelength)
        * 2.0
        * math.sqrt(2.0 * math.log(2.0))
        * sigma_v
        / SPEED_OF_LIGHT
    )


@method(
    equations={
        "line": r"E = E_u - E_l, \qquad \lambda = \frac{hc}{E}, \qquad \epsilon = g\,A_{ul}\, n_u \ [\mathrm{photons\ m^{-3}\,s^{-1}}], \qquad P = \epsilon\, E\, e \ [\mathrm{W\,m^{-3}}]",
    },
    assumptions=(
        r"every spontaneous emission $u \to l$ is a line; the escape factor $g$ is already in the rate",
        r"photon energies from the state energies; vacuum wavelengths, or in air above 200 nm",
    ),
    components=(
        "opencr.constants.photon_wavelength",
        "opencr.constants.air_wavelength",
    ),
)
def line_table(
    network: ReactionNetwork,
    rates: np.ndarray,
    populations: dict[str, float],
    air: bool = False,
) -> xr.Dataset:
    """Return every line's wavelength, emissivity and power.

    Args:
        network: The network.
        rates: The rate coefficients (the escape factors applied).
        populations: Every state's density.
        air: Wavelengths in standard air above 200 nm instead of vacuum.

    Returns:
        A dataset over ``line`` (the reaction names) with ``wavelength`` (nm),
        ``photon_energy`` (eV), ``emissivity`` (photons m^-3 s^-1), ``power``
        (W m^-3), ``upper``, ``lower`` and ``label``.
    """
    names, energy, emissivity, upper, lower, labels = [], [], [], [], [], []
    for i, reaction in enumerate(network.reactions):
        if reaction.kind != "spontaneous_emission":
            continue
        names.append(reaction.name)
        energy.append(reaction.photon_energy)
        u = reaction.reactants[0].name
        emissivity.append(rates[i] * populations[u])
        upper.append(u)
        lower.append(reaction.products[0].name)
        labels.append(_label(network, reaction.name))
    energy_array = np.asarray(energy, dtype=float)
    wavelength = photon_wavelength(energy_array) if names else np.zeros(0)
    if air:
        wavelength = air_wavelength(wavelength)
    emissivity_array = np.asarray(emissivity, dtype=float)
    return xr.Dataset(
        {
            "wavelength": (
                "line",
                wavelength,
                {"units": "nm", "medium": "air" if air else "vacuum"},
            ),
            "photon_energy": ("line", energy_array, {"units": "eV"}),
            "emissivity": ("line", emissivity_array, {"units": "photons m^-3 s^-1"}),
            "power": (
                "line",
                emissivity_array * energy_array * ELEMENTARY_CHARGE,
                {"units": "W m^-3"},
            ),
            "upper": ("line", np.asarray(upper, dtype=object)),
            "lower": ("line", np.asarray(lower, dtype=object)),
            "label": ("line", np.asarray(labels, dtype=object)),
        },
        coords={"line": np.asarray(names, dtype=object)},
    )


@method(
    equations={
        "spectrum": r"\epsilon_\lambda(\lambda) = \sum_{\mathrm{lines}} \frac{\epsilon}{\sigma\sqrt{2\pi}}\, e^{-(\lambda - \lambda_0)^2/2\sigma^2}, \qquad \sigma = \frac{\Delta\lambda_{\mathrm{FWHM}}}{2\sqrt{2\ln 2}}",
        "width": r"\Delta\lambda_{\mathrm{FWHM}}^2 = \Delta\lambda_{\mathrm{Doppler}}^2 + \Delta\lambda_{\mathrm{instrument}}^2",
    },
    assumptions=(
        r"Gaussian lines only: no Stark, Zeeman or natural broadening",
        r"a line without either width raises",
    ),
    components=(doppler_fwhm,),
)
def spectrum(
    lines: xr.Dataset,
    wavelength: np.ndarray,
    temperature: float | None = None,
    mass: float = 1.00794 * ATOMIC_MASS,
    fwhm: float = 0.0,
    quantity: str = "power",
) -> xr.DataArray:
    """Return the spectral emissivity of ``lines`` on the grid ``wavelength`` (nm).

    Each line is a normalised Gaussian whose FWHM is the Doppler width at
    ``temperature`` (eV, of an emitter of ``mass``) and the instrumental
    ``fwhm`` (nm) added in quadrature.

    Args:
        lines: The output of `line_table`.
        wavelength: The grid in nm.
        temperature: The emitters' ``kT`` in eV; no Doppler broadening if ``None``.
        mass: The emitters' mass in kg.
        fwhm: The instrumental FWHM in nm.
        quantity: ``"power"`` (W m^-3 nm^-1) or ``"emissivity"`` (photons m^-3 s^-1 nm^-1).
    """
    if quantity not in ("power", "emissivity"):
        raise ValueError("quantity must be 'power' or 'emissivity'")
    grid = np.asarray(wavelength, dtype=float)
    centres = lines["wavelength"].values
    width2 = np.full(centres.shape, float(fwhm) ** 2)
    if temperature is not None:
        width2 = width2 + doppler_fwhm(centres, temperature, mass) ** 2
    if np.any(width2 <= 0.0):
        raise ValueError(
            "lines need a width: give a temperature or an instrumental fwhm"
        )
    sigma = np.sqrt(width2) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    profile = np.exp(-0.5 * ((grid[:, None] - centres[None, :]) / sigma) ** 2) / (
        sigma * math.sqrt(2.0 * math.pi)
    )
    values = profile @ lines[quantity].values
    unit = "W m^-3 nm^-1" if quantity == "power" else "photons m^-3 s^-1 nm^-1"
    return xr.DataArray(
        values,
        coords={"wavelength": ("wavelength", grid, {"units": "nm"})},
        dims="wavelength",
        name=f"spectral_{quantity}",
        attrs={"units": unit},
    )
