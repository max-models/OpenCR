"""Verification of the atomic hydrogen model against limits and published coefficients.

Acceptance criteria (see the guide "Verification"):

* Saha-Boltzmann: with every radiative process off, the populations are in
  Saha-Boltzmann equilibrium with the ions at any density (detailed balance),
  to 1e-8; with radiation, at n_e = 1e26 m^-3 and 1 eV, to 1e-3.
* Corona: at low density, the populations follow excitation from the ground
  state and radiative cascades (tests/integration).
* Effective coefficients: S_CR and alpha_CR within 25% of ADAS (Summers et al.
  1996, OPEN-ADAS scd12_h and acd12_h) for 2-100 eV and 1e17-1e21 m^-3. The
  ADAS files may not be redistributed: this test runs when OPENCR_ADAS_DIR
  points to the user's own downloads.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import opencr
import opencr.core
from opencr.constants import thermal_de_broglie_cubed
from opencr.data.importers import read_adf11
from opencr.kinetics.eedf import Maxwellian


@pytest.fixture(scope="module")
def model() -> opencr.Model:
    return opencr.load_model("hydrogen/atomic/basic", n_max=12)


def saha_ratio(model, result, temperature) -> dict[str, float]:
    net = model.network
    ion = net.state("H+")
    pops = result.populations
    ne = result.electron_density
    ratios = {}
    for state in net.states:
        if state.name == "H+":
            continue
        saha = (
            ne * pops["H+"] * state.weight / (2.0 * ion.weight)
            * thermal_de_broglie_cubed(temperature)
            * np.exp((ion.energy - state.energy) / temperature)
        )  # fmt: skip
        ratios[state.name] = pops[state.name] / saha
    return ratios


@pytest.mark.parametrize(
    ("temperature", "density"), [(0.5, 1e16), (2.0, 1e19), (20.0, 1e21)]
)
def test_without_radiation_detailed_balance_gives_saha_boltzmann(
    temperature, density
) -> None:
    model = opencr.load_model(
        "hydrogen/atomic/basic", n_max=8, radiative_recombination_=False
    )
    dark = {
        r.name: 0.0 for r in model.network.reactions if r.kind == "spontaneous_emission"
    }
    conditions = opencr.Conditions(
        temperature, density, totals={"H": 1e19}, escape=dark
    )
    result = model.solve_steady_state(conditions, mode="full")
    for name, ratio in saha_ratio(model, result, temperature).items():
        assert ratio == pytest.approx(1.0, rel=1e-8), name


def test_high_density_approaches_saha_boltzmann(model) -> None:
    result = model.solve_steady_state(
        opencr.Conditions(1.0, 1e26, totals={"H": 1e25}), mode="full"
    )
    for name, ratio in saha_ratio(model, result, 1.0).items():
        assert ratio == pytest.approx(1.0, rel=1e-3), name


def test_klein_rosseland_detailed_balance_holds_for_a_tabulated_maxwellian() -> None:
    # excitation and its Klein-Rosseland inverse only: Boltzmann populations
    full = opencr.load_model("hydrogen/atomic/basic", n_max=4).network
    keep = [r for r in full.reactions if r.kind in ("excitation", "deexcitation")]
    network = opencr.core.ReactionNetwork(full.states[:4], keep)
    model = opencr.Model(network)
    table = Maxwellian(2.0).tabulate(6000)
    conditions = opencr.Conditions.from_eedf(
        table, electron_density=1e19, densities={"H(n=1)": 1e19}
    )
    result = model.solve_steady_state(conditions)
    ground = network.state("H(n=1)")
    for n in range(2, 5):
        state = network.state(f"H(n={n})")
        boltzmann = (
            state.weight / ground.weight * np.exp(-(state.energy - ground.energy) / 2.0)
        )
        assert result.population(state.name) / 1e19 == pytest.approx(
            boltzmann, rel=5e-3
        ), n


def test_populations_stay_positive_and_converged_over_a_grid(model) -> None:
    conditions = opencr.Conditions(1.0, 1e18, densities={"H(n=1)": 1e19, "H+": 1e18})
    data = opencr.scan(
        model, conditions, lines=False,
        electron_temperature=np.geomspace(0.2, 200.0, 6), electron_density=np.geomspace(1e14, 1e24, 6),
    )  # fmt: skip
    assert np.all(data["population"].values >= 0.0)
    s_cr = data["effective_rate"].sel(target="H+", source="H(n=1)")
    # ionization grows with temperature at every density
    assert np.all(np.diff(s_cr.values, axis=0) > 0.0)


ADAS = os.environ.get("OPENCR_ADAS_DIR")


@pytest.mark.skipif(
    ADAS is None,
    reason="set OPENCR_ADAS_DIR to a folder with OPEN-ADAS scd12_h.dat and acd12_h.dat",
)
def test_effective_coefficients_against_adas() -> None:
    folder = Path(ADAS or ".")
    scd = read_adf11(folder / "scd12_h.dat")
    acd = read_adf11(folder / "acd12_h.dat")
    model = opencr.load_model("hydrogen/atomic/basic", n_max=20)
    worst = 0.0
    for ne in (1e17, 1e19, 1e21):
        for te in (2.0, 5.0, 10.0, 30.0, 100.0):
            q = model.effective_rates(
                opencr.Conditions(te, ne, densities={"H(n=1)": 1.0, "H+": 1.0})
            )
            s_ratio = q.effective_rate("H(n=1)", "H+") / scd(te, ne)
            a_ratio = q.effective_rate("H+", "H(n=1)") / acd(te, ne)
            worst = max(worst, abs(s_ratio - 1.0), abs(a_ratio - 1.0))
    assert worst < 0.25
