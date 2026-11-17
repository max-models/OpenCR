"""The Python side of the live demo: one steady-state solve of atomic hydrogen.

It runs in the browser (Pyodide); ``run`` takes the values of the sliders and returns the figure
of the level populations and the effective coefficients.
"""

import opencr

_models = {}


def run(electron_temperature=3.0, electron_density=1e19, n_max=15):
    """Solve atomic hydrogen at one set of conditions.

    Args:
        electron_temperature: Electron temperature in eV.
        electron_density: Electron density in m^-3; the protons have the same density.
        n_max: Highest principal quantum number of the model.

    Returns:
        The populations as a bar chart, and S_CR, alpha_CR and the H-alpha emissivity.
    """
    n_max = int(n_max)
    if n_max not in _models:
        _models[n_max] = opencr.load_model("hydrogen/atomic/basic", n_max=n_max)
    conditions = opencr.Conditions(
        electron_temperature=float(electron_temperature),
        electron_density=float(electron_density),
        densities={"H(n=1)": 1e19, "H+": float(electron_density)},
    )
    result = _models[n_max].solve_steady_state(conditions)
    reduction = result.reduction
    levels = list(range(1, n_max + 1))
    populations = [float(result.populations[f"H(n={n})"]) for n in levels]
    halpha = result.emission().sel(line="A:3-2")
    return {
        "figure": {
            "data": [{"type": "bar", "x": levels, "y": populations, "name": "H(n)"}],
            "layout": {
                "xaxis": {
                    "title": {"text": "principal quantum number n"},
                    "dtick": 1 if n_max <= 15 else 5,
                },
                "yaxis": {
                    "title": {"text": "population [m⁻³]"},
                    "type": "log",
                    "exponentformat": "power",
                },
                "showlegend": False,
            },
        },
        "numbers": {
            "Ionization S<sub>CR</sub> [m³ s⁻¹]": float(
                reduction.effective_rate("H(n=1)", "H+")
            ),
            "Recombination α<sub>CR</sub> [m³ s⁻¹]": float(
                reduction.effective_rate("H+", "H(n=1)")
            ),
            "H-alpha emissivity [photons m⁻³ s⁻¹]": float(halpha["emissivity"]),
        },
    }
