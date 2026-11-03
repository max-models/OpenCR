"""Astropy quantities in the setup: converted once to the documented units."""

from __future__ import annotations

import numpy as np
import pytest

u = pytest.importorskip("astropy.units")

import opencr  # noqa: E402
from opencr.analysis.scan import _fields_in_units  # noqa: E402
from opencr.backends.batched import batched_scan  # noqa: E402
from opencr.cli import main  # noqa: E402
from opencr.core import Conditions, RadiationTrapping  # noqa: E402
from opencr.core.rates import (  # noqa: E402
    Arrhenius,
    CrossSectionRate,
    Function,
    Tabulated,
)
from opencr.core.species import Species, State  # noqa: E402
from opencr.inference import Observation, fit, posterior_grid  # noqa: E402
from opencr.kinetics.eedf import Generalized, Maxwellian, TabulatedEEDF  # noqa: E402
from opencr.ml import generate_dataset  # noqa: E402
from opencr.models.argon import argon_network  # noqa: E402
from opencr.models.hydrogen_molecular import molecular_hydrogen_network  # noqa: E402
from opencr.tables import effective_rate_table  # noqa: E402

#: kT of 1 eV in K.
EV_IN_K = 11604.518
DRIVERS = {"H(n=1)": 1e19, "H+": 1e18}


@pytest.fixture(scope="module")
def model() -> opencr.Model:
    return opencr.load_model("hydrogen/atomic/basic", n_max=3)


@pytest.mark.parametrize(
    "temperature",
    [2.0, 2.0 * u.eV, "2 eV", f"{2 * EV_IN_K} K", 2 * EV_IN_K * u.K, "2000 meV"],
)
def test_conditions_temperature(temperature) -> None:
    conditions = Conditions(temperature)
    assert conditions.electron_temperature == pytest.approx(2.0, rel=1e-6)
    assert isinstance(conditions.electron_temperature, float)


def test_conditions_densities_and_gas() -> None:
    conditions = Conditions(
        "2 eV",
        1e12 * u.cm**-3,
        densities={"H(n=1)": "1e13 cm-3", "H+": 1e18},
        totals={"H": 1e19 * u.m**-3},
        ion_temperature="600 K",
        gas_temperature=300 * u.K,
    )
    assert conditions.electron_density == pytest.approx(1e18)
    assert dict(conditions.densities) == pytest.approx({"H(n=1)": 1e19, "H+": 1e18})
    assert conditions.totals["H"] == pytest.approx(1e19)
    assert conditions.ion_temperature == pytest.approx(600 / EV_IN_K)
    assert conditions.gas_temperature == pytest.approx(300 / EV_IN_K)
    assert Conditions(1.0).electron_density is None
    assert Conditions(1.0).gas_temperature == 0.026
    # replace converts too, and keeps the converted numbers
    changed = conditions.replace(electron_temperature="5 eV")
    assert changed.electron_temperature == pytest.approx(5.0)
    assert changed.densities == conditions.densities


@pytest.mark.parametrize(
    "kwargs",
    [
        {"electron_temperature": "1 m"},
        {"electron_temperature": 1.0, "electron_density": "1e18 m"},
        {"electron_temperature": 1.0, "densities": {"H": 1 * u.s}},
        {"electron_temperature": 1.0, "gas_temperature": "1 Pa"},
    ],
)
def test_conditions_wrong_dimension(kwargs) -> None:
    with pytest.raises(ValueError, match="must be in units of"):
        Conditions(**kwargs)


def test_radiation_trapping_length() -> None:
    assert RadiationTrapping("5 cm").length == pytest.approx(0.05)
    assert RadiationTrapping(2 * u.mm).length == pytest.approx(2e-3)
    assert RadiationTrapping(0.1).length == 0.1
    with pytest.raises(ValueError, match="length must be in units of m"):
        RadiationTrapping("1 eV")


def test_eedfs() -> None:
    assert Maxwellian(f"{3 * EV_IN_K} K").temperature == pytest.approx(3.0, rel=1e-6)
    assert Maxwellian(3.0).temperature == 3.0
    assert Generalized(4500 * u.meV, shape=2.0).mean == pytest.approx(4.5)
    energy = np.linspace(0.0, 30.0, 50)
    shape = np.exp(-energy / 2.0)
    tabulated = TabulatedEEDF(energy * u.eV, shape)
    assert np.allclose(TabulatedEEDF(energy / 1e3 * u.keV, shape).energy, energy)
    assert np.allclose(tabulated.energy, energy)
    with pytest.raises(ValueError, match="temperature must be in units of eV"):
        Maxwellian("1 s")


def test_species_and_states() -> None:
    hydrogen = Species("H", 0, "1.00794 u", {"H": 1})
    assert hydrogen.mass == pytest.approx(1.6737e-27, rel=1e-4)
    assert Species("H", 0, 1.6737e-27 * u.kg, {"H": 1}).mass == 1.6737e-27
    level = State(hydrogen, "n=2", "10.2 eV", 8.0)
    assert level.energy == pytest.approx(10.2)
    assert State(hydrogen, "n=2", 10.2e-3 * u.keV, 8.0).energy == pytest.approx(10.2)
    with pytest.raises(ValueError, match="mass must be in units of kg"):
        Species("H", 0, "1 eV")


def test_rate_models() -> None:
    from plasmacoll import CrossSection

    table = CrossSection(energy=[10.0, 20.0, 50.0], sigma=[0.0, 1e-20, 5e-21])
    rate = CrossSectionRate(
        depends_on="gas", cross_section=table, mass="1 u", incident_mass="2 u"
    )
    assert rate.mass == pytest.approx(1.6605e-27, rel=1e-4)
    assert rate.incident_mass == pytest.approx(2 * 1.6605e-27, rel=1e-4)
    assert Arrhenius(a=1e-14, c=f"{EV_IN_K} K").c == pytest.approx(1.0, rel=1e-6)
    tabulated = Tabulated(
        temperatures=[1.0, 10.0] * u.eV, values=(1e-15, 1e-14), threshold="10 eV"
    )
    assert tabulated.temperatures == (1.0, 10.0) and tabulated.threshold == 10.0
    function = Function(
        function=lambda t: 1e-15 * t,
        threshold=0.5 * u.eV,
        file_temperatures=[1.0, 10.0] * u.keV,
    )
    assert function.file_temperatures == (1e3, 1e4) and function.threshold == 0.5
    with pytest.raises(ValueError, match="c must be in units of eV"):
        Arrhenius(c="1 m")


def test_model_builders() -> None:
    plain = argon_network(penning_rate=5e-16, wall_loss={"Ar+": 1e3}, max_energy=1e3)
    quantities = argon_network(
        penning_rate="5e-10 cm3/s", wall_loss={"Ar+": "1 1/ms"}, max_energy="1 keV"
    )
    plain_rates = {r.name: r.rate for r in plain.reactions}
    rates = {r.name: r.rate for r in quantities.reactions}
    assert rates.keys() == plain_rates.keys()
    for name, rate in rates.items():
        if hasattr(rate, "value"):
            assert rate.value == pytest.approx(plain_rates[name].value), name
    molecular = molecular_hydrogen_network(
        v_max=1, electronic=False, wall_loss={"H+": 1e4 / u.s}, max_energy="1 keV"
    )
    assert next(r for r in molecular.reactions if r.kind == "wall_loss").rate.value == (
        pytest.approx(1e4)
    )
    model = opencr.load_model("hydrogen/atomic/basic", n_max=3, max_energy="1 keV")
    assert len(model.state_names) == 4
    with pytest.raises(ValueError, match="penning_rate must be in units of m3/s"):
        argon_network(penning_rate="1 eV")


def test_steady_and_transient(model) -> None:
    conditions = Conditions(2.0, 1e18, densities=DRIVERS)
    plain = model.solve_steady_state(conditions, initial={"H(n=2)": 1e10})
    given = model.solve_steady_state(conditions, initial={"H(n=2)": "1e4 cm-3"})
    assert given.populations == pytest.approx(plain.populations)
    times = np.linspace(0.0, 1e-6, 5)
    reference = model.solve_transient(conditions, {}, times, atol=1.0)
    for value in (times * u.s, "[0, 0.25, 0.5, 0.75, 1] us", list(times * 1e6 * u.us)):
        result = model.solve_transient(
            conditions, {"H(n=2)": 0 * u.cm**-3}, value, atol="1e-6 cm-3"
        )
        assert np.allclose(result.times, times)
        assert np.allclose(
            result.population("H(n=3)"), reference.population("H(n=3)"), rtol=1e-5
        )
    with pytest.raises(ValueError, match="times must be in units of s"):
        model.solve_transient(conditions, {}, [0, 1] * u.m)


def test_scans(model) -> None:
    conditions = Conditions(2.0, 1e18, densities=DRIVERS)
    data = opencr.scan(
        model,
        conditions,
        electron_temperature=[2.0, 4.0] * u.eV,
        **{"density:H(n=1)": ["1e13 cm-3"]},
    )
    assert np.allclose(data["electron_temperature"], [2.0, 4.0])
    assert np.allclose(data["density:H(n=1)"], [1e19])
    table = effective_rate_table(
        model,
        conditions,
        ["effective_rate:H(n=1)>H+"],
        temperatures=[2.0, 4.0] * u.eV,
        densities="[1e12, 1e13] cm-3",
    )
    assert np.allclose(table.dataset["electron_density"], [1e18, 1e19])
    batched = batched_scan(model, conditions, [2.0, 4.0] * u.eV, ["1e12 cm-3"])
    assert np.allclose(batched["electron_density"], [1e18])
    assert np.allclose(batched["electron_temperature"], [2.0, 4.0])


def test_dataset_and_inference(model) -> None:
    conditions = Conditions(2.0, 1e18, densities=DRIVERS)
    data = generate_dataset(
        model,
        conditions,
        {"electron_temperature": ("1 eV", 3 * u.eV)},
        ["effective_rate:H(n=1)>H+"],
        samples=3,
        fractions=(1.0, 0.0, 0.0),
    )
    assert np.allclose(data["low"], [1.0]) and np.allclose(data["high"], [3.0])
    observations = [Observation("x", 2.0)]
    bounds = {"electron_density": ("1e12 cm-3", 1e13 * u.cm**-3), "scale": (1.0, 4.0)}

    def forward(theta):
        return np.array([theta[1]])

    grid = posterior_grid(forward, observations, bounds, points=3)
    assert np.allclose(grid["electron_density"], [1e18, np.sqrt(1e37), 1e19])
    best = fit(
        forward,
        observations,
        bounds,
        start={"electron_density": "3e12 cm-3", "scale": 1.5},
    )
    assert best.parameters["scale"] == pytest.approx(2.0, rel=1e-3)
    assert _fields_in_units({"scale": "2"}) == {"scale": 2.0}
    with pytest.raises(ValueError, match="scale must be a plain number"):
        _fields_in_units({"scale": "2 eV"})


def test_cli_quantities(capsys) -> None:
    args = ["effective", "hydrogen/atomic/basic", "--n-max", "3"]
    assert main([*args, "--te", "10", "--ne", "1e19"]) == 0
    plain = capsys.readouterr().out
    assert main([*args, "--te", f"{10 * EV_IN_K} K", "--ne", "1e13 cm-3"]) == 0
    given = capsys.readouterr().out
    assert given.splitlines()[0].split(":")[0] == plain.splitlines()[0].split(":")[0]
    first = [float(line.split(": ")[1].split()[0]) for line in (plain, given)]
    assert first[1] == pytest.approx(first[0], rel=1e-4)
