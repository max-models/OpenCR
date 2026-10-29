"""The molecular hydrogen models: networks, limits and consistency with plasmacoll."""

from __future__ import annotations

import numpy as np
import pytest
from plasmacoll import rate_coefficient
from plasmacoll.janev import JANEV_FITS, dissociative_attachment
from plasmacoll.vibrational import VibrationalDistribution

import opencr
from opencr.constants import ELECTRON_MASS
from opencr.core.network import ReactionNetwork
from opencr.data import check_network_data
from opencr.data.molecular_hydrogen import (
    ion_states,
    molecular_states,
    vibrational_energy,
)
from opencr.models.hydrogen_molecular import molecular_hydrogen_network

DRIVERS = {"H2(X,v=0)": 1e19, "H(n=1)": 1e18, "H+": 5e16, "H2+": 5e16}


@pytest.fixture(scope="module")
def basic() -> opencr.Model:
    return opencr.load_model("hydrogen/molecular/basic")


@pytest.fixture(scope="module")
def ions() -> opencr.Model:
    return opencr.load_model("hydrogen/molecular/ions", wall_loss={"H3+": 1e5})


def test_registered() -> None:
    names = opencr.available_models()
    assert "hydrogen/molecular/basic" in names and "hydrogen/molecular/ions" in names


def test_networks_are_consistent(basic, ions) -> None:
    for model in (basic, ions):
        assert model.network.problems() == []
        assert not check_network_data(model.network)
    counts = basic.network.manifest()["counts"]["by_kind"]
    assert counts["excitation"] == 14 + 13 + 15 * 5
    assert counts["deexcitation"] == 14 + 13
    assert counts["dissociation"] == 15 * 3
    assert counts["spontaneous_emission"] == 5
    assert ions.network.manifest()["counts"]["by_kind"]["dissociative_attachment"] == 15


def test_energy_scale() -> None:
    states = {s.name: s for s in molecular_states()}
    assert states["H2(X,v=0)"].energy == pytest.approx(-4.478)
    assert vibrational_energy(1) - vibrational_energy(0) == pytest.approx(0.516)
    assert states["H2(B)"].energy - states["H2(X,v=0)"].energy == pytest.approx(12.754)
    charged = ion_states()
    # H2+ -> H+ + H costs the dissociation energy of H2+, 2.65 eV
    assert charged["H+"].energy - charged["H2+"].energy == pytest.approx(2.65, abs=0.01)
    with pytest.raises(ValueError):
        vibrational_energy(15)
    with pytest.raises(ValueError):
        molecular_states(15)


def test_vibrational_ladder_relaxes_to_the_electron_temperature() -> None:
    # e-V excitation and its detailed-balance inverse alone: Boltzmann at T_e
    full = molecular_hydrogen_network(electronic=False)
    ladder = ReactionNetwork(
        [s for s in full.states if s.name.startswith("H2(X")],
        [r for r in full.reactions if r.name.startswith("eV:")],
        name="ladder",
    )
    temperature = 1.5
    result = opencr.Model(ladder, drivers=("H2(X,v=0)",)).solve_steady_state(
        opencr.Conditions(temperature, 1e17, densities={"H2(X,v=0)": 1e19})
    )
    for v in (1, 5, 14):
        expected = 1e19 * np.exp(
            -(vibrational_energy(v) - vibrational_energy(0)) / temperature
        )
        assert result.population(f"H2(X,v={v})") == pytest.approx(expected, rel=1e-6)


def test_ground_state_rates_match_plasmacoll(basic) -> None:
    rates = basic.rate_coefficients(opencr.Conditions(5.0, 1e17))
    for name, key in (
        ("ion:0", "e_h2_ionization"),
        ("exc:0-B", "e_h2_singlet_B"),
        ("diss:0-b", "e_h2_triplet_b"),
        ("eV:0-1", "e_h2_vibrational_1"),
    ):
        expected = rate_coefficient(JANEV_FITS[key].cross_section(), 5.0, ELECTRON_MASS)
        assert rates[name] == pytest.approx(float(expected), rel=0.02)
    # the shifted cross sections of excited levels have lower thresholds
    assert rates["ion:5"] > rates["ion:0"]


def test_dissociative_attachment_matches_plasmacoll_mixture(ions) -> None:
    distribution = VibrationalDistribution.h2_ground_state_from_kelvin(5000.0)
    total = 1e19
    densities = {
        f"H2(X,v={v})": total * float(p) for v, p in enumerate(distribution.populations)
    }
    temperature = 1.0
    rates = ions.rate_coefficients(opencr.Conditions(temperature, 1e17))
    production = sum(rates[f"da:{v}"] * densities[f"H2(X,v={v})"] for v in range(15))
    mixture = rate_coefficient(
        dissociative_attachment(distribution), temperature, ELECTRON_MASS
    )
    assert production == pytest.approx(float(mixture) * total, rel=0.03)


def test_basic_steady_state(basic) -> None:
    result = basic.solve_steady_state(opencr.Conditions(3.0, 1e17, densities=DRIVERS))
    populations = result.populations
    assert all(value >= 0.0 for value in populations.values())
    # the ladder falls off with v, the excited electronic states are tiny
    ladder = [populations[f"H2(X,v={v})"] for v in range(15)]
    assert all(a > b for a, b in zip(ladder, ladder[1:], strict=False))
    assert populations["H2(B)"] < 1e-6 * populations["H2(X,v=0)"]
    reduction = result.reduction
    assert reduction.effective_rate("H2(X,v=0)", "H2+") > 0.0
    assert reduction.effective_rate("H2(X,v=0)", "H+") > 0.0
    lines = result.emission()
    assert "A:d-a" in lines["line"].values


def test_ions_steady_state(ions) -> None:
    conditions = opencr.Conditions(1.0, 1e17, densities=DRIVERS)
    result = ions.solve_steady_state(conditions)
    assert result.report.converged
    h_minus = result.population("H-")
    assert 1e13 < h_minus < 1e17
    fluxes = result.fluxes
    production = sum(fluxes[f"da:{v}"] for v in range(15))
    losses = sum(
        fluxes[name]
        for name in fluxes
        if name in ("det:e", "mutual_neutralization", "hminus_h_associative_detachment",
                    "hminus_h_detachment", "mn:H2+", "mn:H3+")
    )  # fmt: skip
    assert production == pytest.approx(losses, rel=1e-6)
    # the wall loss lowers H3+
    free = opencr.load_model("hydrogen/molecular/ions").solve_steady_state(conditions)
    assert result.population("H3+") < free.population("H3+")


def test_wall_loss_checks_the_state() -> None:
    with pytest.raises(KeyError):
        molecular_hydrogen_network(wall_loss={"He": 1.0})


def test_without_electronic_states() -> None:
    network = molecular_hydrogen_network(v_max=3, electronic=False)
    assert "H2(B)" not in network.state_names
    assert network.problems() == []
