"""Fluxes, dominant processes, emission, spectra, scans and export."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

import opencr
from opencr.analysis.emission import doppler_fwhm, hydrogen_line_name, spectrum


@pytest.fixture(scope="module")
def model() -> opencr.Model:
    return opencr.load_model("hydrogen/atomic/basic", n_max=8)


@pytest.fixture(scope="module")
def result(model) -> opencr.Result:
    conditions = opencr.Conditions(3.0, 1e19, densities={"H(n=1)": 1e19, "H+": 1e18})
    return model.solve_steady_state(conditions)


def test_line_names() -> None:
    assert hydrogen_line_name(3, 2) == "H-α" and hydrogen_line_name(4, 2) == "H-β"
    assert hydrogen_line_name(2, 1) == "Ly-α" and hydrogen_line_name(5, 3) == "Pa-β"
    assert hydrogen_line_name(30, 1) == "30-1" and hydrogen_line_name(9, 8) == "9-8"


def test_fluxes_balance_every_state(result) -> None:
    fluxes = result.fluxes
    for state in result.system.evolving:
        net = sum(
            r.net[state] * fluxes[r.name]
            for r in result.model.network.reactions_of(state)
        )
        gross = sum(
            abs(r.net[state] * fluxes[r.name])
            for r in result.model.network.reactions_of(state)
        )
        assert abs(net) < 1e-9 * gross


def test_dominant_processes(result) -> None:
    top = result.dominant_processes("H(n=2)", 3)
    assert len(top) == 3 and abs(top[0]["rate"]) >= abs(top[1]["rate"])
    assert (
        top[0]["name"] == "A:2-1" and top[0]["rate"] < 0 and 0 < top[0]["fraction"] <= 1
    )
    assert {"name", "equation", "rate", "fraction"} <= set(top[0])


def test_emission_lines(result) -> None:
    lines = result.emission()
    halpha = lines.sel(line="A:3-2")
    assert str(halpha["label"].values) == "H-α"
    assert float(halpha["wavelength"]) == pytest.approx(656.47, abs=0.02)  # vacuum
    assert float(
        result.emission(air=True).sel(line="A:3-2")["wavelength"]
    ) == pytest.approx(656.28, abs=0.02)
    assert float(lines.sel(line="A:2-1")["wavelength"]) == pytest.approx(
        121.57, abs=0.01
    )
    k = result.model.rate_coefficients(result.conditions)
    assert float(halpha["emissivity"]) == pytest.approx(
        result.population("H(n=3)") * k["A:3-2"]
    )
    assert float(halpha["power"]) == pytest.approx(
        float(halpha["emissivity"]) * 1.889 * 1.602176634e-19, rel=1e-3
    )


def test_pecs_split_the_emission_by_driver(result) -> None:
    pec = result.photon_emissivity_coefficients()
    lines = result.emission()
    ne = result.electron_density
    densities = np.array([result.population("H(n=1)"), result.population("H+")])
    reconstructed = ne * pec.values @ densities
    assert reconstructed == pytest.approx(lines["emissivity"].values, rel=1e-8)
    assert list(pec["driver"].values) == ["H(n=1)", "H+"]
    # at 3 eV and these densities, H-alpha is mostly excitation
    halpha = pec.sel(line="A:3-2")
    assert (
        float(halpha.sel(driver="H(n=1)")) * densities[0]
        > float(halpha.sel(driver="H+")) * densities[1]
    )


def test_spectrum_integrates_to_the_line_powers(result) -> None:
    grid = np.linspace(640.0, 670.0, 30001)
    profile = result.emission_spectrum(grid, fwhm=0.05)
    total = np.trapezoid(profile.values, grid)
    assert total == pytest.approx(
        float(result.emission().sel(line="A:3-2")["power"]), rel=1e-4
    )
    assert profile.attrs["units"] == "W m^-3 nm^-1"
    photons = result.emission_spectrum(quantity="emissivity", fwhm=0.1)
    assert photons.attrs["units"].startswith("photons")
    with pytest.raises(ValueError):
        spectrum(result.emission(), grid, quantity="other")
    with pytest.raises(ValueError, match="width"):
        spectrum(result.emission(), grid)
    # Doppler width of H-alpha at 1 eV is about 0.05 nm
    assert doppler_fwhm(656.3, 1.0, 1.67e-27) == pytest.approx(0.0506, rel=0.02)


def test_dataset_and_export(result, tmp_path) -> None:
    data = result.to_dataset()
    assert data["population"].sel(state="H(n=1)") == 1e19
    assert json.loads(data.attrs["manifest"])["name"] == "hydrogen/atomic/basic"
    assert json.loads(data.attrs["conditions"])["electron_temperature"] == 3.0
    path = result.export(tmp_path / "result.nc")
    loaded = xr.open_dataset(path, engine="scipy")
    assert float(loaded["population"].sel(state="H(n=2)")) == pytest.approx(
        result.population("H(n=2)")
    )


def test_scan(model) -> None:
    conditions = opencr.Conditions(1.0, 1e19, densities={"H(n=1)": 1e19, "H+": 1e18})
    data = opencr.scan(
        model,
        conditions,
        electron_temperature=[1.0, 5.0],
        electron_density=[1e18, 1e20],
    )
    assert data["population"].dims == (
        "electron_temperature",
        "electron_density",
        "state",
    )
    point = model.solve_steady_state(
        conditions.replace(electron_temperature=5.0, electron_density=1e20)
    )
    assert float(
        data["population"].sel(
            electron_temperature=5.0, electron_density=1e20, state="H(n=3)"
        )
    ) == pytest.approx(point.population("H(n=3)"))
    s_cr = data["effective_rate"].sel(target="H+", source="H(n=1)")
    assert float(
        s_cr.sel(electron_temperature=5.0, electron_density=1e20)
    ) == pytest.approx(point.reduction.effective_rate("H(n=1)", "H+"))
    assert data["pec"].dims == (
        "electron_temperature",
        "electron_density",
        "line",
        "driver",
    )
    by_density = opencr.scan(
        model, conditions, lines=False, effective=False, **{"density:H+": [1e17, 1e18]}
    )
    assert "emissivity" not in by_density and "effective_rate" not in by_density
    full = opencr.scan(
        model,
        opencr.Conditions(3.0, 1e19, totals={"H": 1e19}),
        mode="full",
        **{"total:H": [1e18, 1e19]},
    )
    assert "effective_rate" not in full and full["population"].sum(
        "state"
    ).values == pytest.approx([1e18, 1e19])
    with pytest.raises(ValueError):
        opencr.scan(model, conditions)
