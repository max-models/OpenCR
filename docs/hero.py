"""Draw the animated figure of the docs home page: the hydrogen levels and their emission.

The energy levels of atomic hydrogen (``hydrogen/atomic/basic``) up to n = 7 as a ladder (binding
energy on a log scale), with
the Lyman, Balmer and Paschen lines between them. The steady state of the model is solved for an
electron temperature sweeping from 1 to 8 eV and back (n_e = 1e18 m^-3, n_H = 1e19 m^-3); the
brightness of each line follows its emissivity and the width of each level its population, and
photons drop along the brightest lines. Hovering over a line names it. Rerun with
``make docs-hero`` (needs ``plasma_plots.hero``); it writes ``docs/src/assets/hero.svg``.
"""

from pathlib import Path

import numpy as np
from plasma_plots.hero import HeroFigure

import opencr

OUTPUT = Path(__file__).parent / "src" / "assets" / "hero.svg"
TOP = 7  # highest level drawn
SERIES = {
    1: (4.9, "Lyman"),
    2: (6.9, "Balmer"),
    3: (8.5, "Paschen"),
}  # lower level: x, name
TEMPERATURES = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 5.0, 3.0, 2.0, 1.5, 1.0]  # eV, one loop
PERIOD = 14.0  # seconds per loop


def energy(n):
    """Return the binding energy of level n in eV."""
    return 13.6 / n**2


def height(n):
    """Return the height of level n in the picture: minus the log10 of its binding energy."""
    return -np.log10(energy(n))


model = opencr.load_model("hydrogen/atomic/basic")
results = [
    model.solve_steady_state(
        opencr.Conditions(
            electron_temperature=te,
            electron_density=1.0e18,
            densities={"H(n=1)": 1.0e19, "H+": 1.0e18},
        )
    )
    for te in TEMPERATURES
]
emission = [result.emission() for result in results]

fig = HeroFigure(xlim=(0.0, 10.0), ylim=(-1.25, 0.68))
levels = range(1, TOP + 1)
log_pop = np.log10(
    [[r.populations[f"H(n={n})"] / n**2 for n in levels] for r in results]
)
log_pop = (log_pop - log_pop.min()) / np.ptp(log_pop)
for i, n in enumerate(levels):
    level = fig.line(
        [4.6, 9.8],
        [height(n)] * 2,
        tone="high",
        width=2.0,
        opacity=0.7,
        title=f"H(n={n}), {energy(n):.2f} eV below the ionization limit",
    )
    level.animate("stroke-width", 0.8 + 3.0 * log_pop[:, i], PERIOD)

lines = [(upper, lower) for lower in SERIES for upper in range(lower + 1, TOP + 1)]
keys = [f"A:{upper}-{lower}" for upper, lower in lines]
log_em = np.log10([[e.emissivity.sel(line=k).item() for k in keys] for e in emission])
log_em = np.clip((log_em - log_em.max() + 5.0) / 5.0, 0.0, 1.0)  # five decades
rng = np.random.default_rng(1)
for j, (upper, lower) in enumerate(lines):
    x0, series = SERIES[lower]
    x = x0 + 0.3 * (upper - lower - 1)
    label = emission[0].label.sel(line=keys[j]).item()
    wavelength = emission[0].wavelength.sel(line=keys[j]).item()
    line = fig.line(
        [x, x + 0.25],
        [height(upper), height(lower)],
        tone="accent",
        width=2.0,
        opacity=0.08 + 0.9 * log_em[4, j],
        title=f"{series} {label}, {wavelength:.1f} nm",
    )
    line.animate("opacity", 0.08 + 0.9 * log_em[:, j], PERIOD)
    if log_em.max(axis=0)[j] > 0.6:
        duration = rng.uniform(2.5, 5.0)
        fig.tracer(line, duration=duration, offset=rng.uniform(0, duration), r=3.0)
fig.save(OUTPUT)
