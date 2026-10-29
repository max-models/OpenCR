"""Quantities, sensitivities, uncertainty propagation and inverse modelling."""

from __future__ import annotations

import numpy as np
import pytest

import opencr
from opencr.analysis.quantities import check_quantity, quantity
from opencr.inference import (
    Observation,
    data_uncertainties,
    fit,
    forward_model,
    posterior_grid,
    propagate,
    sensitivities,
)

IONIZATION = "effective_rate:H(n=1)>H+"
BALMER = "ratio:emissivity:A:3-2/emissivity:A:4-2"


@pytest.fixture(scope="module")
def model() -> opencr.Model:
    return opencr.load_model("hydrogen/atomic/basic", n_max=6)


@pytest.fixture(scope="module")
def conditions() -> opencr.Conditions:
    return opencr.Conditions(5.0, 1e18, densities={"H(n=1)": 1e18, "H+": 1e18})


def test_quantities(model, conditions) -> None:
    result = model.solve_steady_state(conditions)
    assert quantity(result, "population:H(n=2)") == result.population("H(n=2)")
    assert quantity(result, IONIZATION) == result.reduction.effective_rate(
        "H(n=1)", "H+"
    )
    alpha = quantity(result, "emissivity:A:3-2")
    beta = quantity(result, "emissivity:A:4-2")
    assert quantity(result, BALMER) == pytest.approx(alpha / beta)
    for bad, error in (
        ("nothing", ValueError),
        ("ratio:population:H(n=2)", ValueError),
        ("effective_rate:H+", ValueError),
        ("emissivity:exc:1-2", ValueError),
        ("population:He", KeyError),
    ):
        with pytest.raises(error):
            check_quantity(model, bad)


def test_factors_scale_rates_and_keep_detailed_balance(model, conditions) -> None:
    base = model.scaled_rates(conditions)
    scaled = model.scaled_rates(conditions, {"exc:1-2": 2.0})
    names = model.evaluator.names
    for name in ("exc:1-2", "dex:2-1"):
        i = names.index(name)
        assert scaled[i] == pytest.approx(2.0 * base[i])
    with pytest.raises(KeyError):
        model.scaled_rates(conditions, {"nope": 2.0})


def test_sensitivities(model, conditions) -> None:
    s = sensitivities(
        model, conditions, ["population:H(n=2)", IONIZATION], ["exc:1-2", "A:2-1"]
    )
    # H(n=2) at low density is in corona balance: n2 ~ k_12 / A_21
    assert float(s.sel(output="population:H(n=2)", reaction="exc:1-2")) > 0.5
    assert float(s.sel(output="population:H(n=2)", reaction="A:2-1")) < -0.5
    full = sensitivities(model, conditions, [IONIZATION])
    assert "dex:2-1" not in full.reaction.values
    assert float(full.sel(output=IONIZATION, reaction="ion:1")) > 0.5


def test_propagation_linear_and_monte_carlo_agree(model, conditions) -> None:
    uncertainties = {"ion:1": 0.2, "exc:1-2": 0.2}
    linear = propagate(model, conditions, [IONIZATION], uncertainties)
    carlo = propagate(
        model, conditions, [IONIZATION], uncertainties, "monte-carlo", samples=300
    )
    assert linear.relative[0] == pytest.approx(carlo.relative[0], rel=0.2)
    assert linear.dominant(IONIZATION, 1)[0][0] == "ion:1"
    low, high = linear.interval()
    assert low[0] < linear.nominal[0] < high[0]
    assert "relative_uncertainty" in linear.to_dataset()
    with pytest.raises(ValueError):
        carlo.dominant(IONIZATION)
    with pytest.raises(ValueError):
        propagate(model, conditions, [IONIZATION], uncertainties, "exact")


def test_data_uncertainties_from_provenance(model) -> None:
    sigma = data_uncertainties(model)
    assert sigma["exc:1-2"] == pytest.approx(0.2)
    assert "dex:2-1" not in sigma
    assert data_uncertainties(model, default=0.5)["A:2-1"] == pytest.approx(0.01)


def test_inverse_recovers_temperature_and_density(model) -> None:
    base = opencr.Conditions(5.0, 1e18, densities={"H(n=1)": 1e18, "H+": 1e18})
    outputs = [BALMER, "ratio:emissivity:A:2-1/emissivity:A:3-2", "emissivity:A:3-2"]
    forward = forward_model(
        model, base, ["electron_temperature", "electron_density"], outputs
    )
    truth = np.array([3.0, 3e18])
    measured = forward(truth)
    observations = [
        Observation(name, float(value), 0.05)
        for name, value in zip(outputs, measured, strict=True)
    ]
    bounds = {"electron_temperature": (1.0, 20.0), "electron_density": (1e17, 1e20)}
    best = fit(forward, observations, bounds)
    assert best.success and best.chi2 < 1e-6
    assert best.parameters["electron_temperature"] == pytest.approx(3.0, rel=1e-3)
    assert best.parameters["electron_density"] == pytest.approx(3e18, rel=1e-3)
    assert all(np.isfinite(list(best.log_std.values())))
    grid = posterior_grid(forward, observations, bounds, points=9)
    assert float(grid["posterior"].sum()) == pytest.approx(1.0)
    assert grid.attrs["best:electron_temperature"] == pytest.approx(3.0, rel=0.3)
    with pytest.raises(ValueError):
        fit(forward, observations, {"electron_temperature": (2.0, 1.0)})
    with pytest.raises(ValueError):
        Observation(BALMER, -1.0)
    with pytest.raises(ValueError):
        Observation(BALMER, 1.0, 0.0)
