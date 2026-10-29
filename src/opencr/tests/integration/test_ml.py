"""Datasets and the scikit-learn baseline surrogate."""

from __future__ import annotations

import numpy as np
import pytest

import opencr
from opencr.inference import Observation, fit

pytest.importorskip("sklearn")

from opencr.ml import Surrogate, generate_dataset, manifest_digest, split_indices  # noqa: E402

OUTPUTS = ["effective_rate:H(n=1)>H+", "effective_rate:H+>H(n=1)"]
BOUNDS = {"electron_temperature": (1.0, 30.0), "electron_density": (1e17, 1e20)}


@pytest.fixture(scope="module")
def model() -> opencr.Model:
    return opencr.load_model("hydrogen/atomic/basic", n_max=6)


@pytest.fixture(scope="module")
def dataset(model):
    base = opencr.Conditions(5.0, 1e18, densities={"H(n=1)": 1e18, "H+": 1e18})
    return generate_dataset(model, base, BOUNDS, OUTPUTS, samples=160, seed=3)


def test_dataset_provenance_and_split(model, dataset) -> None:
    assert dataset.attrs["manifest_sha256"] == manifest_digest(model)
    assert dataset.attrs["seed"] == 3 and dataset.attrs["solve_time"] > 0.0
    x = dataset["inputs"].values
    assert np.all(x >= dataset["low"].values) and np.all(x <= dataset["high"].values)
    counts = np.bincount(dataset["split"].values)
    assert counts.tolist() == [112, 24, 24]
    np.testing.assert_array_equal(split_indices(160, seed=4), dataset["split"].values)
    with pytest.raises(ValueError):
        split_indices(10, (0.5, 0.5))


def test_surrogate_errors_speed_and_domain(model, dataset) -> None:
    surrogate = Surrogate().fit(dataset)
    report = surrogate.evaluate(dataset)
    assert float(report["median_error"].max()) < 0.05
    assert report.attrs["speed_up"] > 1.0
    inside = surrogate.predict([[5.0, 1e18], [100.0, 1e18]])
    assert inside.in_domain.tolist() == [True, False]
    assert surrogate.provenance["model"] == model.name
    with pytest.raises(RuntimeError):
        Surrogate().predict([[5.0, 1e18]])


def test_surrogate_as_forward_model(dataset) -> None:
    surrogate = Surrogate().fit(dataset)
    truth = surrogate.forward(np.array([4.0, 2e18]))
    observations = [
        Observation(n, float(v), 0.05) for n, v in zip(OUTPUTS, truth, strict=True)
    ]
    best = fit(surrogate.forward, observations, BOUNDS)
    assert best.parameters["electron_temperature"] == pytest.approx(4.0, rel=0.05)
