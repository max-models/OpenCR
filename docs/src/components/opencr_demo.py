"""The Python side of the live demo: one steady-state solve of atomic hydrogen.

It runs in the browser (Pyodide) and returns JSON for the chart and the numbers.
"""

import json

import opencr

_models = {}


def run(electron_temperature, electron_density, n_max):
    """
    Solve atomic hydrogen at one set of conditions.

    Parameters
    ----------
    electron_temperature : float
        Electron temperature in eV.
    electron_density : float
        Electron density in m^-3; the protons have the same density.
    n_max : int
        Highest principal quantum number of the model.

    Returns
    -------
    str
        JSON with the level populations, S_CR, alpha_CR and the H-alpha emissivity.
    """
    n_max = int(n_max)
    if n_max not in _models:
        _models[n_max] = opencr.load_model("hydrogen/atomic/basic", n_max=n_max)
    model = _models[n_max]
    conditions = opencr.Conditions(
        electron_temperature=float(electron_temperature),
        electron_density=float(electron_density),
        densities={"H(n=1)": 1e19, "H+": float(electron_density)},
    )
    result = model.solve_steady_state(conditions)
    reduction = result.reduction
    halpha = result.emission().sel(line="A:3-2")
    return json.dumps(
        {
            "levels": list(range(1, n_max + 1)),
            "populations": [
                float(result.populations[f"H(n={n})"]) for n in range(1, n_max + 1)
            ],
            "ionization": float(reduction.effective_rate("H(n=1)", "H+")),
            "recombination": float(reduction.effective_rate("H+", "H(n=1)")),
            "halpha": float(halpha["emissivity"]),
        }
    )
