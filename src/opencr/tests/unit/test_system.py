"""The population equations, their Jacobian and the quasi-static reduction."""

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
from opencr.kinetics.system import MissingDensityError, PopulationSystem

E = ELECTRON_STATE
(G, X, Y), ION = hydrogen_levels(3)
HM = State(species_from_formula("H-"), "", -0.754, 1.0)


def nonlinear() -> ReactionNetwork:
    return ReactionNetwork(
        [G, X, Y, ION, HM],
        [
            Reaction("exc", (E, G), (E, X), "excitation", Constant(value=2e-15)),
            Reaction("ion", (E, X), (E, E, ION), "ionization", Constant(value=1e-14)),
            Reaction(
                "3b",
                (E, E, ION),
                (E, Y),
                "three_body_recombination",
                Constant(value=1e-38),
            ),
            Reaction("a32", (Y,), (X,), "spontaneous_emission", Constant(value=4e7)),
            Reaction("att", (E, G), (HM,), "attachment", Constant(value=1e-17)),
            Reaction(
                "mn", (ION, HM), (G, X), "mutual_neutralization", Constant(value=3e-13)
            ),
        ],
    )


def finite_difference(system: PopulationSystem, n: np.ndarray) -> np.ndarray:
    jac = np.zeros((system.size, system.size))
    for j in range(system.size):
        step = 1e-6 * max(abs(n[j]), 1.0)
        up, down = n.copy(), n.copy()
        up[j] += step
        down[j] -= step
        jac[:, j] = (system.rhs(up) - system.rhs(down)) / (2 * step)
    return jac


@pytest.mark.parametrize("ne", [1e18, None])
def test_jacobian_matches_finite_differences(ne) -> None:
    network = nonlinear()
    k = np.array([r.rate.value for r in network.reactions])
    conditions = Conditions(3.0, ne, densities={"H(n=1)": 1e19})
    evolving = ["H(n=2)", "H(n=3)", "H+", "H-"]
    system = PopulationSystem(network, k, conditions, evolving)
    assert (
        system.prescribed == ("H(n=1)",)
        and system.max_order == 2
        and not system.is_linear
    )
    n = np.array([1e15, 3e14, 5e17, 2e16])
    exact = system.jacobian(n).toarray()
    assert exact == pytest.approx(
        finite_difference(system, n), rel=1e-6, abs=1e-9 * np.abs(exact).max()
    )
    if ne is None:
        assert system.electron_density(n) == pytest.approx(5e17 - 2e16)
    with pytest.raises(ValueError, match="not linear"):
        system.linear_parts()


def test_reaction_rates_and_densities() -> None:
    network = nonlinear()
    k = np.array([r.rate.value for r in network.reactions])
    system = PopulationSystem(
        network,
        k,
        Conditions(3.0, 1e18, densities={"H(n=1)": 1e19}),
        ["H(n=2)", "H(n=3)", "H+", "H-"],
    )
    n = np.array([1.0, 2.0, 3.0, 4.0])
    w = system.reaction_rates(n)
    assert w[0] == pytest.approx(2e-15 * 1e18 * 1e19)
    assert w[2] == pytest.approx(1e-38 * 1e36 * 3.0)
    assert w[5] == pytest.approx(3e-13 * 3.0 * 4.0)
    assert system.densities(n)["H(n=1)"] == 1e19 and system.densities(n)["H-"] == 4.0
    with pytest.raises(MissingDensityError):
        PopulationSystem(network, k, Conditions(3.0, 1e18), ["H(n=2)"])
    with pytest.raises(KeyError):
        PopulationSystem(network, k, Conditions(3.0, 1e18), ["nothing"])
    with pytest.raises(ValueError, match="one rate"):
        PopulationSystem(network, k[:2], Conditions(3.0, 1e18), network.state_names)


def two_level(
    a: float = 4.7e8, k12: float = 1e-15, k21: float = 3e-14
) -> ReactionNetwork:
    return ReactionNetwork(
        [G, X],
        [
            Reaction("exc", (E, G), (E, X), "excitation", Constant(value=k12)),
            Reaction("dex", (E, X), (E, G), "deexcitation", Constant(value=k21)),
            Reaction("rad", (X,), (G,), "spontaneous_emission", Constant(value=a)),
        ],
    )


def test_linear_parts_and_conservation() -> None:
    network = two_level()
    k = np.array([1e-15, 3e-14, 4.7e8])
    system = PopulationSystem(network, k, Conditions(2.0, 1e19), network.state_names)
    assert system.is_linear
    m, b = system.linear_parts()
    n = np.array([1e19, 1e12])
    assert m @ n + b == pytest.approx(system.rhs(n))
    names, matrix = system.conservation()
    assert names == ("H",) and matrix.tolist() == [[1.0, 1.0]]
    # with the ground state prescribed, H is exchanged with it: not conserved
    sub = PopulationSystem(
        network, k, Conditions(2.0, 1e19, densities={"H(n=1)": 1.0}), ["H(n=2)"]
    )
    assert sub.conservation()[0] == ()


def test_two_level_quasi_static_is_analytic() -> None:
    a, k12, k21, ne, n1 = 4.7e8, 1e-15, 3e-14, 1e19, 1e20
    network = two_level(a, k12, k21)
    k = np.array([k12, k21, a])
    reduction = quasi_static(network, k, Conditions(2.0, ne, densities={"H(n=1)": n1}))
    expected = ne * k12 / (a + ne * k21)
    assert reduction.coefficients[0, 0] == pytest.approx(expected, rel=1e-12)
    assert reduction.population_coefficient("H(n=2)", "H(n=1)") == pytest.approx(
        expected / ne
    )
    assert reduction.populations({"H(n=1)": n1})[0] == pytest.approx(expected * n1)
    # the ground state is conserved: no effective loss
    assert reduction.effective_rate("H(n=1)", "H(n=1)") == pytest.approx(0.0, abs=1e-25)


def test_effective_ionization_through_an_excited_state() -> None:
    # 1 -> 2 by excitation, 2 -> ion by ionization, 2 -> 1 by emission: S_CR = k12 kion ne / (A + kion ne)
    network = ReactionNetwork(
        [G, X, ION],
        [
            Reaction("exc", (E, G), (E, X), "excitation", Constant(value=1e-15)),
            Reaction("ion", (E, X), (E, E, ION), "ionization", Constant(value=1e-13)),
            Reaction("rad", (X,), (G,), "spontaneous_emission", Constant(value=1e8)),
            Reaction(
                "rr", (E, ION), (G,), "radiative_recombination", Constant(value=1e-19)
            ),
        ],
    )
    k = np.array([1e-15, 1e-13, 1e8, 1e-19])
    ne = 1e20
    reduction = quasi_static(
        network, k, Conditions(1.0, ne, densities={"H(n=1)": 1.0, "H+": 1.0})
    )
    s_cr = 1e-15 * 1e-13 * ne / (1e8 + 1e-13 * ne)
    assert reduction.effective_rate("H(n=1)", "H+") == pytest.approx(s_cr, rel=1e-12)
    assert reduction.effective_rate("H+", "H(n=1)") == pytest.approx(1e-19, rel=1e-12)
    assert reduction.effective_rate("H(n=1)", "H(n=1)") == pytest.approx(
        -s_cr, rel=1e-12
    )


def test_quasi_static_errors() -> None:
    network = two_level()
    k = np.array([1e-15, 3e-14, 4.7e8])
    with pytest.raises(ValueError, match="given electron density"):
        quasi_static(network, k, Conditions(2.0, densities={"H(n=1)": 1.0}))
    with pytest.raises(ValueError, match="no drivers"):
        quasi_static(network, k, Conditions(2.0, 1e19))
    lossless = np.array([1e-15, 0.0, 0.0])
    with pytest.raises(SingularSystemError, match="no loss"):
        quasi_static(
            network, lossless, Conditions(2.0, 1e19, densities={"H(n=1)": 1.0})
        )
    network3 = nonlinear()
    k3 = np.array([r.rate.value for r in network3.reactions])
    with pytest.raises(ValueError, match="one quasi-static reactant"):
        quasi_static(network3, k3, Conditions(2.0, 1e19, densities={"H(n=1)": 1.0}))
    only_drivers = quasi_static(
        network, k, Conditions(2.0, 1e19, densities={"H(n=1)": 1.0, "H(n=2)": 1.0})
    )
    assert only_drivers.coefficients.shape == (0, 2)


def test_quasi_static_with_a_driver_partner() -> None:
    # mn (ION + HM) is linear in HM once ION is a driver: the reduction agrees
    # with the steady state at the given driver densities
    from opencr.model import Model

    network = nonlinear()
    conditions = Conditions(2.0, 1e19, densities={"H(n=1)": 1e19, "H+": 1e17})
    reduction = Model(network).effective_rates(conditions)
    result = Model(network).solve_steady_state(conditions)
    populations = reduction.populations(dict(conditions.densities))
    for name, value in zip(reduction.excited, populations, strict=True):
        assert value == pytest.approx(result.population(name), rel=1e-9)
    # the drivers' own rates: H+ is lost by mutual neutralization with H-
    # and by recombination, gained by ionization
    rate = result.fluxes
    gain = rate["ion"] - rate["3b"] - rate["mn"]
    n_p = np.array([conditions.densities[d] for d in reduction.drivers])
    assert (reduction.effective @ n_p)[reduction.drivers.index("H+")] == pytest.approx(
        gain, rel=1e-9
    )


def test_closed_cycle_without_path_is_singular() -> None:
    # 2 <-> 3 exchange each other but never return to the driver
    network = ReactionNetwork(
        [G, X, Y],
        [
            Reaction("exc", (E, G), (E, X), "excitation", Constant(value=1e-15)),
            Reaction("x", (E, X), (E, Y), "excitation", Constant(value=1e-14)),
            Reaction("y", (Y,), (X,), "spontaneous_emission", Constant(value=1e7)),
        ],
    )
    k = np.array([1e-15, 1e-14, 1e7])
    with pytest.raises(SingularSystemError, match="singular"):
        quasi_static(network, k, Conditions(2.0, 1e19, densities={"H(n=1)": 1.0}))


def test_source_by_prescribed() -> None:
    network = nonlinear()
    k = np.array([r.rate.value for r in network.reactions])
    conditions = Conditions(
        3.0, 1e18, densities={"H(n=1)": 1e19, "H+": 1e18, "H-": 1e17}
    )
    system = PopulationSystem(network, k, conditions, ["H(n=2)", "H(n=3)"])
    parts = system.source_by_prescribed()
    assert set(parts) == {"H(n=1)", "H+", "H-", "(mixed)"}  # H+ + H- needs both
    _, b = system.linear_parts()
    assert sum(parts.values()) == pytest.approx(b)
