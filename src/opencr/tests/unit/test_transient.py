"""Time-dependent populations: analytic decays, relaxation to steady states, conservation."""

from __future__ import annotations

import numpy as np
import pytest

import opencr
from opencr.core import ELECTRON_STATE, Conditions, Constant, Reaction, ReactionNetwork
from opencr.data.hydrogen import hydrogen_levels
from opencr.kinetics.system import PopulationSystem
from opencr.solvers.transient import solve_transient

E = ELECTRON_STATE
(G, X), ION = hydrogen_levels(2)


def two_level() -> ReactionNetwork:
    return ReactionNetwork(
        [G, X, ION],
        [
            Reaction("exc", (E, G), (E, X), "excitation", Constant(value=1e-15)),
            Reaction("rad", (X,), (G,), "spontaneous_emission", Constant(value=1e8)),
            Reaction("ion", (E, G), (E, E, ION), "ionization", Constant(value=2e-17)),
            Reaction(
                "rr", (E, ION), (G,), "radiative_recombination", Constant(value=1e-19)
            ),
        ],
    )


def test_pure_decay_is_exponential() -> None:
    network = two_level()
    rates = np.array([0.0, 1e8, 0.0, 0.0])
    conditions = Conditions(1.0, 1e18, densities={"H(n=1)": 1e19, "H+": 0.0})
    system = PopulationSystem(network, rates, conditions, ["H(n=2)"])
    times = np.linspace(0.0, 5e-8, 11)
    n, report = solve_transient(system, [1e15], times, rtol=1e-10, atol=1.0)
    assert n[:, 0] == pytest.approx(1e15 * np.exp(-1e8 * times), rel=1e-7)
    assert report.success and report.evaluations > 0 and report.minimum > 0
    assert "reached the final time" in report.summary()


@pytest.mark.parametrize("method", ["BDF", "Radau", "LSODA"])
def test_relaxation_to_the_closed_steady_state(method) -> None:
    model = opencr.Model(two_level(), drivers=("H(n=1)", "H+"))
    conditions = Conditions(2.0, 1e19, totals={"H": 1e20})
    steady = model.solve_steady_state(conditions, mode="full")
    times = np.geomspace(1e-9, 10.0, 40)
    history = model.solve_transient(
        conditions, {"H(n=1)": 1e20}, times, method=method, rtol=1e-8
    )
    final = history.at()
    for state in model.state_names:
        assert final.population(state) == pytest.approx(
            steady.population(state), rel=1e-5
        )
    assert history.report.conservation["H"] < 1e-8
    assert final.report.residual < 1e-4 and final.mode == "transient"
    assert history.population("H+")[0] < history.population("H+")[-1]
    data = history.to_dataset()
    assert data["population"].shape == (40, 3) and data.attrs["solve"] == "transient"
    assert np.all(data["electron_density"].values == 1e19)
    assert "40 times" in repr(history)
    assert history.conditions == conditions
    assert history.conditions_at(1.0) == conditions


def test_quasi_neutral_transient_conserves_charge() -> None:
    model = opencr.load_model("hydrogen/atomic/basic", n_max=6)
    conditions = Conditions(3.0, None, totals={"H": 1e19})
    times = np.geomspace(1e-9, 1e-3, 20)
    history = model.solve_transient(
        conditions, {"H(n=1)": 0.9e19, "H+": 0.1e19}, times, rtol=1e-8
    )
    assert history.electron_density == pytest.approx(history.population("H+"))
    assert history.report.conservation["H"] < 1e-8
    assert list(history.report.conservation) == ["H"]


def test_prescribed_drivers_with_time_dependent_conditions() -> None:
    model = opencr.load_model("hydrogen/atomic/basic", n_max=5)
    base = Conditions(1.0, 1e18, densities={"H(n=1)": 1e19, "H+": 1e18})

    def conditions(t: float) -> Conditions:
        return base.replace(electron_temperature=1.0 + 9.0 * min(t / 1e-6, 1.0))

    times = np.array([0.0, 1e-7, 1e-6, 1e-5, 1e-4])
    history = model.solve_transient(conditions, {}, times)
    # at 1e-4 s the excited states have reached the quasi-static state at 10 eV
    steady = model.solve_steady_state(conditions(1.0))
    assert history.population("H(n=3)")[-1] == pytest.approx(
        steady.population("H(n=3)"), rel=1e-4
    )
    assert history.population("H(n=1)")[-1] == 1e19  # prescribed
    assert history.conditions_at(1.0).electron_temperature == 10.0
    assert history.at(-1).conditions.electron_temperature == 10.0
    assert history.report.conservation == {}  # the drivers take the atoms


def test_invalid_inputs() -> None:
    model = opencr.Model(two_level(), drivers=("H(n=1)", "H+"))
    conditions = Conditions(2.0, 1e19, totals={"H": 1e20})
    with pytest.raises(ValueError, match="stiff"):
        model.solve_transient(conditions, {}, [0.0, 1.0], method="RK45")
    with pytest.raises(ValueError, match="increasing"):
        model.solve_transient(conditions, {}, [1.0, 0.0])
    with pytest.raises(ValueError, match="increasing"):
        model.solve_transient(conditions, {}, [1.0])
    with pytest.raises(ValueError, match=">= 0"):
        model.solve_transient(conditions, {"H+": -1.0}, [0.0, 1.0])
    with pytest.raises(KeyError, match="unknown states"):
        model.solve_transient(conditions, {"He": 1.0}, [0.0, 1.0])
    system = model.system(conditions)
    with pytest.raises(ValueError, match="densities"):
        solve_transient(system, [1.0], [0.0, 1.0])

    def changing(t: float) -> Conditions:
        return conditions if t == 0.0 else conditions.replace(densities={"H+": 1.0})

    with pytest.raises(ValueError, match="same at every time"):
        model.solve_transient(changing, {"H(n=1)": 1e20}, [0.0, 1.0])


def test_a_failed_integration_raises_with_the_report(monkeypatch) -> None:
    import opencr.solvers.transient as module

    class Failed:
        success = False
        message = "step size too small"
        nfev = 3
        njev = 1
        nlu = 1
        t = np.zeros(1)
        y = np.zeros((1, 1))

    monkeypatch.setattr(module, "solve_ivp", lambda *a, **k: Failed())
    model = opencr.Model(two_level(), drivers=("H(n=1)", "H+"))
    system = model.system(Conditions(2.0, 1e19, totals={"H": 1e20}))
    with pytest.raises(RuntimeError, match="step size too small"):
        solve_transient(system, np.ones(3), [0.0, 1.0])


def test_small_negative_densities_are_reported_and_zeroed(monkeypatch) -> None:
    import opencr.solvers.transient as module

    class Solution:
        success = True
        message = ""
        nfev = 3
        njev = 1
        nlu = 1
        t = np.array([0.0, 1.0])
        y = np.array([[1.0, 1.0], [0.0, -1e-9], [0.0, 0.0]])

    monkeypatch.setattr(module, "solve_ivp", lambda *a, **k: Solution())
    model = opencr.Model(two_level(), drivers=("H(n=1)", "H+"))
    system = model.system(Conditions(2.0, 1e19, totals={"H": 1e20}))
    n, report = solve_transient(system, np.array([1.0, 0.0, 0.0]), [0.0, 1.0])
    assert np.all(n >= 0.0) and "negative densities" in report.summary()
    assert report.factorisations == 1
