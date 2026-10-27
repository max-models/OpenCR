"""Steady states: linear, closed and nonlinear systems, and their failures."""

from __future__ import annotations

import numpy as np
import pytest

from opencr.core import (
    ELECTRON_STATE,
    Conditions,
    Constant,
    Reaction,
    ReactionNetwork,
    State,
    species_from_formula,
)
from opencr.data.hydrogen import hydrogen_levels
from opencr.kinetics.effective import SingularSystemError, quasi_static
from opencr.kinetics.system import PopulationSystem
from opencr.solvers.diagnostics import (
    ConvergenceError,
    SolverReport,
    UnderdeterminedError,
)
from opencr.solvers.steady import constraint_rows, solve_steady

E = ELECTRON_STATE
(G, X, Y), ION = hydrogen_levels(3)
HM = State(species_from_formula("H-"), "", -0.754, 1.0)


def rates(network):
    return np.array([r.rate.value for r in network.reactions])


def ionization_balance() -> ReactionNetwork:
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


def test_quasi_static_linear_solve_matches_the_reduction() -> None:
    network = ionization_balance()
    conditions = Conditions(2.0, 1e19, densities={"H(n=1)": 1e19, "H+": 1e18})
    system = PopulationSystem(network, rates(network), conditions, ["H(n=2)"])
    n, report = solve_steady(system)
    reduction = quasi_static(network, rates(network), conditions)
    assert n == pytest.approx(
        reduction.populations(dict(conditions.densities)), rel=1e-12
    )
    assert report.converged and report.method == "sparse LU" and report.residual < 1e-12
    assert "converged" in report.summary()


def test_closed_linear_ionization_balance() -> None:
    network = ionization_balance()
    conditions = Conditions(2.0, 1e19, totals={"H": 1e20})
    system = PopulationSystem(network, rates(network), conditions, network.state_names)
    n, report = solve_steady(system, conditions.totals, mode="full")
    # ion/neutral = k_ion / k_rr (the excited state is a small correction)
    assert n[2] / n[0] == pytest.approx(2e-17 / 1e-19, rel=1e-12)
    assert n.sum() == pytest.approx(1e20, rel=1e-12)
    assert report.conservation["H"] < 1e-12 and report.mode == "full"


def test_quasi_neutral_closed_balance_is_solved_by_continuation() -> None:
    network = ionization_balance()
    conditions = Conditions(2.0, None, totals={"H": 1e20})
    system = PopulationSystem(network, rates(network), conditions, network.state_names)
    assert not system.is_linear
    n, report = solve_steady(system, conditions.totals, mode="full")
    assert report.converged and "quasi-neutral" in report.method
    # the continuation from a starting point finds the same state
    again, other = solve_steady(system, conditions.totals, initial=n * 1.5, mode="full")
    assert "Newton" in other.method and again == pytest.approx(n, rel=1e-6)
    assert n[2] / n[0] == pytest.approx(200.0, rel=1e-6)
    assert system.electron_density(n) == pytest.approx(n[2])
    assert n.sum() == pytest.approx(1e20, rel=1e-9)


def test_nonlinear_negative_ion_balance_matches_a_long_relaxation() -> None:
    network = ReactionNetwork(
        [G, ION, HM],
        [
            Reaction("att", (E, G), (HM,), "attachment", Constant(value=1e-17)),
            Reaction("det", (E, HM), (E, E, G), "detachment", Constant(value=3e-14)),
            Reaction(
                "mn", (ION, HM), (G, G), "mutual_neutralization", Constant(value=4e-13)
            ),
            Reaction("ion", (E, G), (E, E, ION), "ionization", Constant(value=1e-18)),
        ],
    )
    conditions = Conditions(2.0, 1e17, densities={"H(n=1)": 1e20})
    system = PopulationSystem(network, rates(network), conditions, ["H+", "H-"])
    n, report = solve_steady(system, initial=np.array([1e17, 1e15]))
    assert report.converged
    assert (
        np.max(
            np.abs(system.rhs(n)) / np.array([1e-18 * 1e17 * 1e20, 1e-17 * 1e17 * 1e20])
        )
        < 1e-8
    )
    # analytic: H+ made by ionization, lost to mutual neutralization
    ion_source = 1e-18 * 1e17 * 1e20
    assert 4e-13 * n[0] * n[1] == pytest.approx(ion_source, rel=1e-8)
    assert n[1] == pytest.approx(
        (1e-17 * 1e17 * 1e20 - ion_source) / (3e-14 * 1e17), rel=1e-8
    )
    # the same from the default starting point
    default, _ = solve_steady(system)
    assert default == pytest.approx(n, rel=1e-7)


def test_failures() -> None:
    network = ionization_balance()
    system = PopulationSystem(
        network, rates(network), Conditions(2.0, 1e19), network.state_names
    )
    with pytest.raises(UnderdeterminedError, match=r"\['H'\]"):
        solve_steady(system)
    lossless = rates(network)
    lossless[1] = 0.0
    sub = PopulationSystem(
        network,
        lossless,
        Conditions(2.0, 1e19, densities={"H(n=1)": 1.0, "H+": 1.0}),
        ["H(n=2)"],
    )
    with pytest.raises(SingularSystemError, match="no loss"):
        solve_steady(sub)
    cycle = ReactionNetwork(
        [G, X, Y],
        [
            Reaction("exc", (E, G), (E, X), "excitation", Constant(value=1e-15)),
            Reaction("x", (E, X), (E, Y), "excitation", Constant(value=1e-14)),
            Reaction("y", (Y,), (X,), "spontaneous_emission", Constant(value=1e7)),
        ],
    )
    trapped = PopulationSystem(
        cycle,
        rates(cycle),
        Conditions(2.0, 1e19, densities={"H(n=1)": 1.0}),
        ["H(n=2)", "H(n=3)"],
    )
    with pytest.raises(SingularSystemError, match="singular"):
        solve_steady(trapped)
    quasi = PopulationSystem(
        network,
        rates(network),
        Conditions(2.0, None, totals={"H": 1e20}),
        network.state_names,
    )
    with pytest.raises(ConvergenceError) as error:
        solve_steady(quasi, {"H": 1e20}, initial=np.full(3, 1e19), max_iterations=1)
    assert not error.value.report.converged


def test_constraint_rows_pick_distinct_states() -> None:
    network = ionization_balance()
    system = PopulationSystem(
        network, rates(network), Conditions(2.0, 1e19), network.state_names
    )
    rows, matrix, values, names = constraint_rows(system, {"H": 5.0})
    assert rows.tolist() == [0] and names == ("H",) and values.tolist() == [5.0]


def test_report_summary_lists_problems() -> None:
    report = SolverReport(
        "full",
        "Newton",
        converged=False,
        conservation={"H": 1e-3},
        clipped=["H(n=9)"],
        messages=["slow"],
    )
    text = report.summary()
    assert (
        "NOT converged" in text
        and "H(n=9)" in text
        and "slow" in text
        and "conservation" in text
    )


def test_a_plasma_without_ionization_has_no_quasi_neutral_state() -> None:
    network = ReactionNetwork(
        [G, ION],
        [
            Reaction(
                "rr", (E, ION), (G,), "radiative_recombination", Constant(value=1e-19)
            )
        ],
    )
    system = PopulationSystem(
        network,
        rates(network),
        Conditions(1.0, None, totals={"H": 1e19}),
        network.state_names,
    )
    with pytest.raises(ConvergenceError, match="recombines completely"):
        solve_steady(system, {"H": 1e19})
