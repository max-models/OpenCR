"""Effective-rate tables and the electron cooling coefficient."""

from __future__ import annotations

import numpy as np
import pytest

import opencr
from opencr.analysis.quantities import quantity
from opencr.tables import EffectiveRateTable, effective_rate_table

OUTPUTS = [
    "effective_rate:H(n=1)>H+",
    "effective_rate:H+>H(n=1)",
    "electron_cooling:H(n=1)",
    "electron_cooling:H+",
]


@pytest.fixture(scope="module")
def model() -> opencr.Model:
    return opencr.load_model("hydrogen/atomic/basic", n_max=8)


@pytest.fixture(scope="module")
def conditions() -> opencr.Conditions:
    return opencr.Conditions(5.0, 1e18, densities={"H(n=1)": 1e18, "H+": 1e18})


def test_electron_cooling(model, conditions) -> None:
    result = model.solve_steady_state(conditions)
    cooling = quantity(result, "electron_cooling:H(n=1)")
    ionization = quantity(result, "effective_rate:H(n=1)>H+")
    # at least the ionization energy per CR ionization, plus the radiated excitation
    assert cooling > 13.6 * ionization
    assert cooling < 13.6 * ionization + 20.0 * 1e-13
    # three-body recombination heats the electrons
    cold = model.solve_steady_state(conditions.replace(electron_temperature=0.3))
    assert quantity(cold, "electron_cooling:H+") < 0.0


def test_table_interpolates_and_round_trips(model, conditions, tmp_path) -> None:
    temperatures = np.geomspace(1.0, 50.0, 12)
    densities = np.geomspace(1e17, 1e20, 4)
    table = effective_rate_table(model, conditions, OUTPUTS, temperatures, densities)
    assert table.quantities == tuple(OUTPUTS)
    assert table.dataset.attrs["model"] == "hydrogen/atomic/basic"
    # on the grid it is exact, between the nodes close to a direct solve
    node = table(OUTPUTS[0], temperatures[3], densities[1])
    assert node == pytest.approx(table.dataset["value"].values[0, 3, 1], rel=1e-12)
    point = conditions.replace(electron_temperature=7.0, electron_density=3e18)
    direct = quantity(model.solve_steady_state(point), OUTPUTS[0])
    assert table(OUTPUTS[0], 7.0, 3e18) == pytest.approx(direct, rel=0.05)
    # outside the grid the end values hold; arrays broadcast
    assert table(OUTPUTS[0], 1e3, 1e17) == pytest.approx(table(OUTPUTS[0], 50.0, 1e17))
    assert table(OUTPUTS[2], np.array([2.0, 3.0]), 1e18).shape == (2,)
    rate = table.function(OUTPUTS[1], 1e18)
    assert rate(5.0) == pytest.approx(table(OUTPUTS[1], 5.0, 1e18))
    path = table.to_netcdf(tmp_path / "h.nc")
    again = EffectiveRateTable.from_netcdf(path)
    assert again(OUTPUTS[0], 7.0, 3e18) == pytest.approx(table(OUTPUTS[0], 7.0, 3e18))
    with pytest.raises(KeyError):
        table("population:H(n=2)", 1.0, 1e18)
    with pytest.raises(KeyError):
        table.function("nope", 1e18)
    with pytest.raises(ValueError):
        EffectiveRateTable(table.dataset.drop_vars("value"))
