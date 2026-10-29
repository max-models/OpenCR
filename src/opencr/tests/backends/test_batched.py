"""The batched backends against the NumPy/SciPy reference."""

from __future__ import annotations

import cunumpy
import numpy as np
import pytest

import opencr
from opencr.backends import BACKENDS, CUNUMPY, NUMPY, Backend, get_backend
from opencr.backends.batched import BatchedQuasiStatic, batched_scan
from opencr.core import (
    ELECTRON_STATE,
    Arrhenius,
    Conditions,
    Constant,
    Function,
    Reaction,
    ReactionNetwork,
    Tabulated,
)
from opencr.data.hydrogen import hydrogen_levels
from opencr.kinetics.eedf import Generalized
from opencr.kinetics.effective import SingularSystemError

E = ELECTRON_STATE


@pytest.fixture(scope="module")
def model() -> opencr.Model:
    return opencr.load_model("hydrogen/atomic/basic", n_max=8)


def test_backends() -> None:
    assert get_backend("numpy") is NUMPY and get_backend(CUNUMPY) is CUNUMPY
    assert set(BACKENDS) == {"numpy", "cunumpy"}
    assert NUMPY.device == "cpu" and repr(NUMPY) == "Backend('numpy', cpu)"
    assert CUNUMPY.device in ("cpu", "gpu")
    with pytest.raises(ValueError, match="backend"):
        get_backend("torch")
    assert isinstance(CUNUMPY.to_numpy(CUNUMPY.asarray([1.0])), np.ndarray)
    assert isinstance(Backend("numpy", np).to_numpy([1, 2]), np.ndarray)


@pytest.mark.parametrize("backend", ["numpy", "cunumpy"])
def test_batched_rates_match_the_per_point_evaluation(backend) -> None:
    (g, x), ion = hydrogen_levels(2)
    network = ReactionNetwork(
        [g, x, ion],
        [
            Reaction(
                "exc", (E, g), (E, x), "excitation", Arrhenius(a=1e-14, b=0.5, c=10.2)
            ),
            Reaction(
                "rad",
                (x,),
                (g,),
                "spontaneous_emission",
                Constant(value=1e8),
                escape=0.5,
            ),
            Reaction(
                "ion",
                (E, g),
                (E, E, ion),
                "ionization",
                Tabulated(
                    temperatures=(1.0, 10.0, 100.0),
                    values=(1e-16, 1e-14, 2e-14),
                    threshold=13.6,
                ),
            ),
            Reaction(
                "rr",
                (E, ion),
                (g,),
                "radiative_recombination",
                Function(function=lambda t: 1e-19 / np.sqrt(t), label="rr"),
            ),
            Reaction(
                "gas",
                (g, x),
                (g, g),
                "quenching",
                Arrhenius(a=1e-16, depends_on="gas"),
            ),
        ],
    )
    model = opencr.Model(network)
    conditions = Conditions(1.0, 1e18, escape={"rad": 0.25})
    temperatures = np.geomspace(0.3, 300.0, 7)
    xp = get_backend(backend).xp
    batched = model.evaluator.coefficients_batched(conditions, temperatures, xp)
    k = get_backend(backend).to_numpy(batched)
    assert k.shape == (7, 5)
    for i, t in enumerate(temperatures):
        point = model.evaluator.coefficients(conditions.replace(electron_temperature=t))
        assert k[i] == pytest.approx(point, rel=1e-12)
    assert k[0, 1] == pytest.approx(0.25e8)
    assert model.evaluator.coefficients_scan(conditions, temperatures) == pytest.approx(
        k
    )
    with pytest.raises(ValueError, match="> 0"):
        model.evaluator.coefficients_batched(conditions, [1.0, -1.0], xp)
    eedf = Conditions.from_eedf(Generalized(2.0, 2.0))
    with pytest.raises(ValueError, match="Maxwellian"):
        model.evaluator.coefficients_batched(eedf, temperatures, xp)


@pytest.mark.parametrize("backend", ["numpy", "cunumpy"])
def test_batched_reduction_matches_the_reference(model, backend) -> None:
    temperatures = np.array([1.0, 5.0, 30.0])
    densities = np.array([1e17, 1e20])
    data = batched_scan(model, Conditions(1.0, 1e18), temperatures, densities, backend)
    assert data.attrs["backend"] == backend
    assert data["effective_rate"].dims == (
        "electron_temperature",
        "electron_density",
        "target",
        "source",
    )
    for i, t in enumerate(temperatures):
        for j, n in enumerate(densities):
            conditions = Conditions(t, n, densities={"H(n=1)": 1e19, "H+": 1e18})
            reference = model.effective_rates(conditions)
            assert data["effective_rate"].values[i, j] == pytest.approx(
                reference.effective / n, rel=1e-9
            )
            assert data["population_coefficient"].values[i, j] == pytest.approx(
                reference.coefficients / n, rel=1e-9
            )
            pec = model.solve_steady_state(conditions).photon_emissivity_coefficients()
            assert data["pec"].values[i, j] == pytest.approx(pec.values, rel=1e-9)
            assert list(data["line"].values) == list(pec["line"].values)


def test_reduction_accessors(model) -> None:
    conditions = Conditions(1.0, 1e18, densities={"H(n=1)": 1e19, "H+": 1e18})
    reduction = BatchedQuasiStatic(model).run(conditions, [2.0, 20.0], 1e19)
    reference = model.effective_rates(
        conditions.replace(electron_temperature=20.0, electron_density=1e19)
    )
    assert reduction.effective_rate("H(n=1)", "H+")[1] == pytest.approx(
        reference.effective_rate("H(n=1)", "H+")
    )
    assert reduction.population_coefficient("H(n=3)", "H+")[1] == pytest.approx(
        reference.population_coefficient("H(n=3)", "H+")
    )
    assert reduction.rates.shape == (2, len(model.network.reactions))
    assert reduction.to_dataset()["pec"].shape == (2, 28, 2)


def test_structure_errors(model) -> None:
    (g, x), ion = hydrogen_levels(2)
    nonlinear = ReactionNetwork(
        [g, x, ion],
        [
            Reaction("exc", (E, g), (E, x), "excitation", Constant(value=1e-15)),
            Reaction("rad", (x,), (g,), "spontaneous_emission", Constant(value=1e8)),
            Reaction(
                "pair", (x, x), (g, ion, E), "penning_ionization", Constant(value=1e-16)
            ),
            Reaction(
                "rr", (E, ion), (g,), "radiative_recombination", Constant(value=1e-19)
            ),
        ],
    )
    with pytest.raises(ValueError, match="one heavy reactant"):
        BatchedQuasiStatic(opencr.Model(nonlinear))
    lossless = ReactionNetwork(
        [g, x, ion],
        [
            Reaction("exc", (E, g), (E, x), "excitation", Constant(value=1e-15)),
            Reaction(
                "rr", (E, ion), (g,), "radiative_recombination", Constant(value=1e-19)
            ),
        ],
    )
    batched = BatchedQuasiStatic(opencr.Model(lossless, drivers=("H(n=1)", "H+")))
    with pytest.raises(SingularSystemError, match="no loss process"):
        batched.run(Conditions(1.0, 1e18), 1.0, 1e18)
    with pytest.raises(ValueError, match="no drivers"):
        BatchedQuasiStatic(opencr.Model(lossless)).run(Conditions(1.0, 1e18), 1.0, 1e18)
    with pytest.raises(ValueError, match="> 0"):
        BatchedQuasiStatic(model).run(Conditions(1.0, 1e18), 1.0, 0.0)
    everything = Conditions(1.0, 1e18, densities={n: 1.0 for n in model.state_names})
    reduction = BatchedQuasiStatic(model).run(everything, [1.0, 2.0], 1e18)
    assert reduction.excited == () and reduction.coefficients.shape == (2, 0, 9)


def test_singular_matrix_is_reported(model, monkeypatch) -> None:
    def failing(*args, **kwargs):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np.linalg, "solve", failing)
    with pytest.raises(SingularSystemError, match="singular"):
        BatchedQuasiStatic(model, NUMPY).run(Conditions(1.0, 1e18), 1.0, 1e18)


def test_cunumpy_namespace_is_used() -> None:
    assert CUNUMPY.xp is cunumpy
