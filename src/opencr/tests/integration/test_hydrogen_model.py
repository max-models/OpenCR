"""The atomic hydrogen model: its network, its data and its limits."""

from __future__ import annotations

import numpy as np
import pytest

import opencr
from opencr.data import check_network_data
from opencr.models.hydrogen_atomic import atomic_hydrogen_network


@pytest.fixture(scope="module")
def model() -> opencr.Model:
    return opencr.load_model("hydrogen/atomic/basic", n_max=12)


def test_network_contents(model) -> None:
    network = model.network
    counts = network.manifest()["counts"]["by_kind"]
    pairs = 12 * 11 // 2
    assert counts == {
        "excitation": pairs,
        "deexcitation": pairs,
        "spontaneous_emission": pairs,
        "ionization": 12,
        "three_body_recombination": 12,
        "radiative_recombination": 12,
    }
    assert model.drivers == ("H(n=1)", "H+")
    assert repr(model) == "Model('hydrogen/atomic/basic', 13 states, 234 reactions)"
    manifest = model.manifest()
    assert manifest["drivers"] == ["H(n=1)", "H+"] and manifest["assumptions"]
    assert any("Johnson" in source for source in manifest["sources"])
    assert check_network_data(network) == []


def test_options() -> None:
    network = atomic_hydrogen_network(
        4,
        deexcitation=False,
        three_body_recombination=False,
        radiative_recombination_=False,
    )
    kinds = {r.kind for r in network.reactions}
    assert kinds == {"excitation", "spontaneous_emission", "ionization"}
    deuterium = opencr.load_model("hydrogen/atomic/basic", n_max=3, isotope="D")
    assert deuterium.drivers == ("D(n=1)", "D+") and deuterium.name == "d/atomic/basic"
    with pytest.raises(ValueError):
        atomic_hydrogen_network(1)
    with pytest.raises(KeyError, match="available"):
        opencr.load_model("nothing")
    assert "hydrogen/atomic/basic" in opencr.available_models()
    with pytest.raises(KeyError):
        opencr.Model(network, drivers=["H(n=99)"])


def test_quasi_static_needs_the_drivers(model) -> None:
    with pytest.raises(ValueError, match="drivers"):
        model.solve_steady_state(
            opencr.Conditions(2.0, 1e19, densities={"H(n=1)": 1e19})
        )
    with pytest.raises(ValueError, match="mode"):
        model.solve_steady_state(opencr.Conditions(2.0, 1e19), mode="other")


def test_coronal_limit_with_cascades(model) -> None:
    # at low density: n_m sum_l A_ml = n_e n_1 k_1m + sum_j>m n_j A_jm
    ne, n1 = 1.0e10, 1.0e18
    conditions = opencr.Conditions(5.0, ne, densities={"H(n=1)": n1, "H+": 0.0})
    result = model.solve_steady_state(conditions)
    k = model.rate_coefficients(conditions)
    n_max = 12
    expected = np.zeros(n_max + 1)
    for m in range(n_max, 1, -1):
        loss = sum(k[f"A:{m}-{low}"] for low in range(1, m))
        gain = ne * n1 * k[f"exc:1-{m}"] + sum(
            expected[j] * k[f"A:{j}-{m}"] for j in range(m + 1, n_max + 1)
        )
        expected[m] = gain / loss
    for m in range(2, n_max + 1):
        assert result.population(f"H(n={m})") == pytest.approx(expected[m], rel=1e-4), m


def test_full_steady_state_is_the_ionization_balance(model) -> None:
    conditions = opencr.Conditions(5.0, 1e19, totals={"H": 1e19})
    full = model.solve_steady_state(conditions, mode="full")
    reduction = model.effective_rates(
        opencr.Conditions(5.0, 1e19, densities={"H(n=1)": 1.0, "H+": 1.0})
    )
    ratio = reduction.effective_rate("H(n=1)", "H+") / reduction.effective_rate(
        "H+", "H(n=1)"
    )
    pops = full.populations
    # the excited states hold a little of the atoms, so the ratio is within their share
    assert pops["H+"] / pops["H(n=1)"] == pytest.approx(ratio, rel=1e-3)
    assert sum(pops.values()) == pytest.approx(1e19, rel=1e-9)
    assert full.mode == "full" and full.report.conservation["H"] < 1e-10


def test_quasi_neutral_full_steady_state(model) -> None:
    conditions = opencr.Conditions(3.0, None, totals={"H": 1e19})
    result = model.solve_steady_state(conditions, mode="full")
    assert result.electron_density == pytest.approx(result.population("H+"))
    assert result.report.converged
    # the same as a solve at that fixed n_e
    fixed = model.solve_steady_state(
        conditions.replace(electron_density=result.electron_density), mode="full"
    )
    assert fixed.population("H+") == pytest.approx(result.population("H+"), rel=1e-6)


def test_result_reduction(model) -> None:
    conditions = opencr.Conditions(2.0, 1e19, densities={"H(n=1)": 1e19, "H+": 1e18})
    result = model.solve_steady_state(conditions)
    assert "quasi-static" in repr(result) and result.mode == "quasi-static"
    pops = result.reduction.populations(dict(conditions.densities))
    assert pops[0] == pytest.approx(result.population("H(n=2)"), rel=1e-10)
    start = model.solve_steady_state(conditions, initial={"H(n=2)": 1.0})
    assert start.population("H(n=2)") == pytest.approx(result.population("H(n=2)"))


def test_n_max_is_limited() -> None:
    with pytest.raises(ValueError, match="n_max"):
        atomic_hydrogen_network(61)


def test_high_levels_use_vriens_smeets() -> None:
    from opencr.core.rates import CrossSectionRate, Function

    network = atomic_hydrogen_network(30)
    assert isinstance(network.reaction("exc:23-24").rate, CrossSectionRate)
    assert isinstance(network.reaction("exc:1-25").rate, Function)
    assert isinstance(network.reaction("exc:24-30").rate, Function)
    model = opencr.Model(network, drivers=("H(n=1)", "H+"))
    low = opencr.Model(atomic_hydrogen_network(24), drivers=("H(n=1)", "H+"))
    conditions = opencr.Conditions(2.0, 1e19, densities={"H(n=1)": 1e19, "H+": 1e19})
    high = model.effective_rates(conditions)
    base = low.effective_rates(conditions)
    # the levels above 24 add a little recombination and leave the low levels alone
    recombination = high.effective_rate("H+", "H(n=1)")
    assert recombination == pytest.approx(base.effective_rate("H+", "H(n=1)"), rel=0.2)
    assert high.population_coefficient("H(n=3)", "H(n=1)") == pytest.approx(
        base.population_coefficient("H(n=3)", "H(n=1)"), rel=1e-3
    )
    # Saha-Boltzmann without radiation holds with the Vriens-Smeets rates too
    thermal = opencr.Model(atomic_hydrogen_network(30, radiative_recombination_=False))
    lines = {r.name: 0.0 for r in thermal.network.reactions if r.name.startswith("A:")}
    result = thermal.solve_steady_state(
        opencr.Conditions(2.0, 1e20, densities={"H(n=1)": 1e18}, escape=lines)
    )
    ratio = result.population("H(n=30)") / result.population("H(n=29)")
    weights = (2 * 30**2) / (2 * 29**2)
    gap = 13.598 * (1 / 29**2 - 1 / 30**2)
    assert ratio == pytest.approx(weights * np.exp(-gap / 2.0), rel=1e-6)
