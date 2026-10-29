"""Radiation trapping: optical depths, escape factors and the self-consistent solve."""

from __future__ import annotations

import math

import numpy as np
import pytest

import opencr
from opencr.constants import thermal_de_broglie_cubed
from opencr.core import (
    Conditions,
    Function,
    RadiationTrapping,
    Reaction,
    ReactionNetwork,
)
from opencr.data.hydrogen import hydrogen_levels
from opencr.kinetics.radiation import (
    escape_factor,
    escape_factors,
    line_centre_optical_depth,
)
from opencr.solvers.diagnostics import ConvergenceError


def test_lyman_alpha_line_centre_cross_section() -> None:
    # sigma_0 = kappa_0 / n_l of Lyman-alpha at 300 K: 4.5e-17 m^2 (4.5e-13 cm^2)
    tau = line_centre_optical_depth(
        121.567, 6.265e8, 4.0, 1e19, 1.0, 0.02585, 1.6735e-27
    )
    assert tau / 1e19 == pytest.approx(4.5e-17, rel=0.02)
    assert (
        line_centre_optical_depth(121.567, 6.265e8, 4.0, 0.0, 1.0, 0.03, 1.7e-27) == 0.0
    )


def test_escape_factor_models() -> None:
    tau = np.array([0.0, 1.0, 10.0, 1e4])
    mewe = escape_factor(tau)
    assert mewe[0] == 1.0 and mewe[1] == pytest.approx(0.5, rel=1e-3)
    assert mewe[-1] == pytest.approx(2.0 / 1e4, rel=1e-3)
    slab = escape_factor(tau, "holstein-slab")
    assert slab[0] == 1.0 and slab[1] == 1.0  # below tau = e: unity
    assert slab[-1] == pytest.approx(1.0 / (1e4 * math.sqrt(math.pi * math.log(1e4))))
    cylinder = escape_factor(tau, "holstein-cylinder")
    assert cylinder[-1] == pytest.approx(1.60 * slab[-1])
    assert np.all(np.diff(mewe) < 0) and np.all(np.diff(slab) <= 0)
    with pytest.raises(ValueError, match="model"):
        escape_factor(1.0, "sobolev")
    with pytest.raises(ValueError, match=">= 0"):
        escape_factor(-1.0)


def test_configuration_is_checked() -> None:
    with pytest.raises(ValueError, match="length"):
        RadiationTrapping(0.0)
    with pytest.raises(ValueError, match="model"):
        RadiationTrapping(1.0, model="none")
    trapping = RadiationTrapping(0.1, lines=["A:2-1"])
    assert trapping.lines == ("A:2-1",)
    assert trapping.to_dict() == {"length": 0.1, "model": "mewe", "lines": ["A:2-1"]}
    with pytest.raises(TypeError, match="RadiationTrapping"):
        Conditions(1.0, 1e18, trapping="thick")  # type: ignore[arg-type]
    conditions = Conditions(1.0, 1e18, trapping=trapping)
    assert conditions.to_dict()["trapping"]["length"] == 0.1


def test_escape_factors_of_a_network() -> None:
    model = opencr.load_model("hydrogen/atomic/basic", n_max=4)
    populations = {n: 0.0 for n in model.state_names}
    populations["H(n=1)"] = 1e20
    factors = escape_factors(model.network, populations, RadiationTrapping(10.0), 0.026)
    assert set(factors) == {
        r.name for r in model.network.reactions if r.kind == "spontaneous_emission"
    }
    assert factors["A:2-1"] < 1e-3 and factors["A:3-2"] == 1.0  # n=2 is empty
    only = escape_factors(
        model.network, populations, RadiationTrapping(0.01, lines=["A:2-1"]), 0.026
    )
    assert list(only) == ["A:2-1"]
    with pytest.raises(ValueError, match="not a spontaneous emission"):
        escape_factors(
            model.network,
            populations,
            RadiationTrapping(0.01, lines=["exc:1-2"]),
            0.026,
        )
    (g, x), ion = hydrogen_levels(2)
    network = ReactionNetwork(
        [g, x],
        [
            Reaction(
                "rad",
                (x,),
                (g,),
                "spontaneous_emission",
                Function(function=lambda t: 1e8 + 0.0 * t, depends_on="gas"),
            )
        ],
    )
    with pytest.raises(ValueError, match="constant Einstein"):
        escape_factors(
            network,
            {"H(n=1)": 1e19, "H(n=2)": 0.0},
            RadiationTrapping(1.0),
            0.026,
        )


@pytest.fixture(scope="module")
def model() -> opencr.Model:
    return opencr.load_model("hydrogen/atomic/basic", n_max=8)


def test_trapping_raises_the_excited_populations(model) -> None:
    base = Conditions(3.0, 1e18, densities={"H(n=1)": 1e19, "H+": 1e18})
    thin = model.solve_steady_state(base)
    thick = model.solve_steady_state(base.replace(trapping=RadiationTrapping(0.1)))
    assert thick.population("H(n=2)") > 5.0 * thin.population("H(n=2)")
    assert thick.conditions.escape["A:2-1"] < 0.1
    assert min(thick.conditions.escape, key=thick.conditions.escape.get) == "A:2-1"
    assert "escape factors" in thick.report.summary()
    # the result is self-consistent: recomputing the factors changes nothing
    again = escape_factors(
        model.network, thick.populations, RadiationTrapping(0.1), base.gas_temperature
    )
    assert again == pytest.approx(dict(thick.conditions.escape), abs=1e-6)
    # emission uses the trapped rates
    assert thick.emission()["emissivity"].sel(line="A:2-1") < thin.emission()[
        "emissivity"
    ].sel(line="A:2-1") * thick.population("H(n=2)") / thin.population("H(n=2)")


def test_given_escape_factors_override_the_computed_ones(model) -> None:
    base = Conditions(3.0, 1e18, densities={"H(n=1)": 1e19, "H+": 1e18})
    result = model.solve_steady_state(
        base.replace(trapping=RadiationTrapping(10.0), escape={"A:2-1": 1.0})
    )
    assert result.conditions.escape["A:2-1"] == 1.0
    assert result.conditions.escape["A:3-1"] < 0.1


def test_fully_trapped_lines_give_saha_boltzmann() -> None:
    # with every line trapped and no radiative recombination, only collisions remain
    model = opencr.load_model(
        "hydrogen/atomic/basic", n_max=5, radiative_recombination_=False
    )
    temperature = 2.0
    conditions = Conditions(
        temperature, None, totals={"H": 1e19}, trapping=RadiationTrapping(1e15)
    )
    result = model.solve_steady_state(conditions, mode="full")
    ne = result.electron_density
    ion = model.network.state("H+")
    for state in model.network.states:
        if state.name == "H+":
            continue
        saha = (
            ne * result.population("H+") * state.weight / (2.0 * ion.weight)
            * thermal_de_broglie_cubed(temperature)
            * np.exp((ion.energy - state.energy) / temperature)
        )  # fmt: skip
        assert result.population(state.name) == pytest.approx(saha, rel=1e-7)


def test_the_iteration_can_fail(model) -> None:
    base = Conditions(3.0, 1e18, densities={"H(n=1)": 1e19, "H+": 1e18})
    with pytest.raises(ConvergenceError, match="escape factors did not converge"):
        model.solve_steady_state(
            base.replace(trapping=RadiationTrapping(1.0, max_iterations=1))
        )


def test_damping_when_the_fixed_point_oscillates(model, monkeypatch) -> None:
    import opencr.kinetics.radiation as radiation

    calls = []
    original = radiation.escape_factors

    def noisy(network, populations, trapping, temperature):
        factors = original(network, populations, trapping, temperature)
        calls.append(factors["A:2-1"])
        if len(calls) < 4:  # bounce the Lyman-alpha factor to force damping
            factors["A:2-1"] = 0.9 if len(calls) % 2 else 0.1
        return factors

    monkeypatch.setattr(radiation, "escape_factors", noisy)
    base = Conditions(3.0, 1e18, densities={"H(n=1)": 1e19, "H+": 1e18})
    result = model.solve_steady_state(base.replace(trapping=RadiationTrapping(1.0)))
    assert len(calls) >= 5 and result.report.converged
