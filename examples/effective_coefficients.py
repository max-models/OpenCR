"""Effective coefficients.

The effective ionization and recombination coefficients of atomic hydrogen:
the collisional-radiative coefficients S_CR and alpha_CR of the
``hydrogen/atomic/basic`` model against the electron temperature at three
densities, next to the ionization rate of the ground state alone. These are
the coefficients a fluid or global model takes from OpenCR.
"""

import numpy as np
import plasma_plots

import opencr

model = opencr.load_model("hydrogen/atomic/basic", n_max=20)
conditions = opencr.Conditions(1.0, 1e19, densities={"H(n=1)": 1.0, "H+": 1.0})
data = opencr.scan(
    model,
    conditions,
    lines=False,
    electron_temperature=np.geomspace(0.5, 200.0, 40),
    electron_density=[1e16, 1e19, 1e22],
)
s_cr = data["effective_rate"].sel(target="H+", source="H(n=1)")
alpha_cr = data["effective_rate"].sel(target="H(n=1)", source="H+")
temperatures = data["electron_temperature"].values
ground = [
    model.rate_coefficients(conditions.replace(electron_temperature=t))["ion:1"]
    for t in temperatures
]

with plasma_plots.figure(1, 2) as fig:
    for ne in data["electron_density"].values:
        label = f"n_e = {ne:.0e} m^-3"
        s_cr.sel(electron_density=ne).plasma.plot.lineout(ax=fig[0], title="S_CR")
        alpha_cr.sel(electron_density=ne).plasma.plot.lineout(
            ax=fig[1], title="alpha_CR"
        )
        for ax in fig:
            ax.lines[-1].set_label(label)
    fig[0].plot(temperatures, ground, "k--", label="ground state only")
    for ax in fig:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("m^3/s")
        ax.legend()
