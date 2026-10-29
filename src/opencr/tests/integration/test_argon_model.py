"""The argon model with lumped 4s levels and a sputtered metal."""

from __future__ import annotations

import numpy as np
import pytest
from plasmacoll import metal_ionization_cross_section, rate_coefficient

import opencr
from opencr.constants import ELECTRON_MASS
from opencr.data import check_network_data
from opencr.models.argon import argon_network

WALLS = {"Ar+": 1e5, "Ar(m)": 1e4, "Cu+": 1e5, "Cu++": 1e5}


def test_networks() -> None:
    plain = opencr.load_model("argon/metastable")
    assert plain.drivers == ("Ar",)
    assert plain.network.problems() == [] and not check_network_data(plain.network)
    copper = opencr.load_model("argon/metastable", metal="Cu", wall_loss=WALLS)
    assert copper.drivers == ("Ar", "Cu")
    assert {"Cu", "Cu+", "Cu++"} <= set(copper.state_names)
    assert copper.network.problems() == [] and not check_network_data(copper.network)
    names = [r.name for r in argon_network(pooling=False).reactions]
    assert "pool:m-m" not in names
    with pytest.raises(ValueError):
        argon_network(metal="Fe")
    with pytest.raises(KeyError):
        argon_network(wall_loss={"Cu": 1.0})


def test_rates() -> None:
    model = opencr.load_model("argon/metastable", metal="Ti")
    rates = model.rate_coefficients(opencr.Conditions(3.0, 1e18))
    assert rates["ion:0"] == pytest.approx(2.34e-14 * 3.0**0.59 * np.exp(-17.44 / 3.0))
    lotz = rate_coefficient(metal_ionization_cross_section("Ti"), 3.0, ELECTRON_MASS)
    assert rates["ion:Ti"] == pytest.approx(float(lotz), rel=0.02)


def test_steady_state_and_stepwise_ionization() -> None:
    model = opencr.load_model("argon/metastable", metal="Cu", wall_loss=WALLS)
    conditions = opencr.Conditions(3.0, 1e18, densities={"Ar": 1e20, "Cu": 1e18})
    result = model.solve_steady_state(conditions)
    populations = result.populations
    assert 1e-5 < populations["Ar(m)"] / populations["Ar"] < 1e-1
    assert result.fluxes["penning:Cu"] > 0.0
    linear = opencr.load_model(
        "argon/metastable", pooling=False, wall_loss={"Ar(m)": 1e4, "Ar+": 1e5}
    )
    reduction = linear.effective_rates(
        opencr.Conditions(3.0, 1e18, densities={"Ar": 1e20})
    )
    direct = linear.rate_coefficients(opencr.Conditions(3.0, 1e18))["ion:0"]
    # the stepwise path through Ar(m) adds to the direct ionization
    assert reduction.population_coefficient("Ar+", "Ar") > 0.0
    assert -reduction.effective_rate("Ar", "Ar") > direct
