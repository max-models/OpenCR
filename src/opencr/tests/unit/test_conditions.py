"""Plasma conditions."""

from __future__ import annotations

import pytest

from opencr.core import Conditions


def test_conditions_defaults_and_temperatures() -> None:
    conditions = Conditions(2.0, 1e18, densities={"H(n=1)": 1e19})
    assert not conditions.quasi_neutral and conditions.densities["H(n=1)"] == 1e19
    assert conditions.heavy_temperature == {
        "electron": 2.0,
        "ion": 0.026,
        "gas": 0.026,
        "none": 1.0,
    }
    assert Conditions(2.0, ion_temperature=1.0).heavy_temperature["ion"] == 1.0
    assert Conditions(2.0).quasi_neutral
    changed = conditions.replace(electron_temperature=5.0)
    assert (
        changed.electron_temperature == 5.0
        and changed.densities == conditions.densities
    )
    data = conditions.to_dict()
    assert data["densities"] == {"H(n=1)": 1e19} and data["eedf"] is None
    with pytest.raises(TypeError):
        conditions.densities["H+"] = 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"electron_temperature": 0.0},
        {"electron_temperature": 1.0, "electron_density": -1.0},
        {"electron_temperature": 1.0, "gas_temperature": 0.0},
        {"electron_temperature": 1.0, "ion_temperature": -1.0},
        {"electron_temperature": 1.0, "densities": {"H": -1.0}},
        {"electron_temperature": 1.0, "totals": {"H": float("nan")}},
        {"electron_temperature": 1.0, "escape": {"Ly-a": 1.5}},
    ],
)
def test_bad_conditions_are_refused(kwargs) -> None:
    with pytest.raises(ValueError):
        Conditions(**kwargs)


def test_from_eedf() -> None:
    class Fake:
        effective_temperature = 3.0

        def __repr__(self) -> str:
            return "Fake()"

    conditions = Conditions.from_eedf(Fake(), electron_density=1e17)
    assert (
        conditions.electron_temperature == 3.0
        and conditions.to_dict()["eedf"] == "Fake()"
    )
