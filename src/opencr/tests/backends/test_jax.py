"""The JAX backend against the reference: rates, AD Jacobians, vmapped reductions, gradients."""

from __future__ import annotations

import numpy as np
import pytest

import opencr
from opencr.backends.batched import batched_scan
from opencr.core import (
    ELECTRON_STATE,
    Conditions,
    Constant,
    Reaction,
    ReactionNetwork,
)
from opencr.data.hydrogen import hydrogen_levels
from opencr.kinetics.eedf import Generalized
from opencr.kinetics.effective import SingularSystemError

jax = pytest.importorskip("jax")
from opencr.backends.jax_backend import JaxModel  # noqa: E402

E = ELECTRON_STATE


@pytest.fixture(scope="module")
def model() -> opencr.Model:
    return opencr.load_model("hydrogen/atomic/basic", n_max=5)


@pytest.fixture(scope="module")
def collisional() -> opencr.Model:
    # no Function rate: the JAX rates are then exactly the reference's
    return opencr.load_model(
        "hydrogen/atomic/basic", n_max=5, radiative_recombination_=False
    )


def test_rates_match_the_reference(model, collisional) -> None:
    conditions = Conditions(2.0, 1e18, escape={"A:2-1": 0.3})
    exact = JaxModel(collisional, conditions)
    for t in (0.7, 5.0, 60.0):
        reference = collisional.evaluator.coefficients(
            conditions.replace(electron_temperature=t)
        )
        assert np.asarray(exact.rates(t)) == pytest.approx(reference, rel=1e-10)
    batch = np.asarray(exact.rates(np.array([0.7, 5.0, 60.0])))
    assert batch.shape == (3, len(collisional.network.reactions))
    assert batch[1] == pytest.approx(np.asarray(exact.rates(5.0)), rel=1e-12)
    # a Function rate (radiative recombination) is tabulated: within 1e-3
    tabulated = JaxModel(model, conditions)
    reference = model.evaluator.coefficients(
        conditions.replace(electron_temperature=5.0)
    )
    assert np.asarray(tabulated.rates(5.0)) == pytest.approx(reference, rel=1e-3)
    with pytest.raises(ValueError, match="EEDF"):
        JaxModel(model, Conditions.from_eedf(Generalized(2.0, 2.0)))


def test_ad_jacobian_matches_the_analytic_one(model) -> None:
    # quasi-neutral, every state evolving: the n_e terms are in the Jacobian
    conditions = Conditions(2.0, None, totals={"H": 1e19})
    jm = JaxModel(model, conditions)
    system = model.system(conditions.replace(electron_temperature=3.0))
    n = np.geomspace(1e10, 1e19, system.size)
    assert np.asarray(jm.rhs(n, 3.0)) == pytest.approx(system.rhs(n), rel=1e-6)
    analytic = system.jacobian(n).toarray()
    assert np.asarray(jm.jacobian(n, 3.0)) == pytest.approx(
        analytic, rel=1e-6, abs=1e-6 * np.abs(analytic).max()
    )
    # given n_e and prescribed drivers
    given = Conditions(2.0, 1e18, densities={"H(n=1)": 1e19, "H+": 1e18})
    jm = JaxModel(model, given)
    system = model.system(given)
    n = np.geomspace(1e10, 1e15, system.size)
    analytic = system.jacobian(n).toarray()
    assert np.asarray(jm.jacobian(n, 2.0, 1e18)) == pytest.approx(
        analytic, rel=1e-6, abs=1e-6 * np.abs(analytic).max()
    )
    # finite differences of the JAX rhs agree with the AD Jacobian
    step = 1e-6 * n
    numeric = np.column_stack(
        [
            (
                np.asarray(jm.rhs(n + h * e, 2.0, 1e18))
                - np.asarray(jm.rhs(n - h * e, 2.0, 1e18))
            )
            / (2 * h)
            for h, e in zip(step, np.eye(system.size), strict=True)
        ]
    )
    assert numeric == pytest.approx(
        analytic, rel=1e-4, abs=1e-5 * np.abs(analytic).max()
    )


def test_nonlinear_network_has_a_jacobian_but_no_reduction() -> None:
    (g, x), ion = hydrogen_levels(2)
    network = ReactionNetwork(
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
    model = opencr.Model(network)
    conditions = Conditions(2.0, None, totals={"H": 1e19})
    jm = JaxModel(model, conditions)
    system = model.system(conditions)
    n = np.array([1e19, 1e15, 1e17])
    assert np.asarray(jm.jacobian(n, 2.0)) == pytest.approx(
        system.jacobian(n).toarray(), rel=1e-10
    )
    with pytest.raises(ValueError, match="one heavy reactant"):
        jm.reduction(2.0, 1e18)
    with pytest.raises(ValueError, match="one heavy reactant"):
        jm.effective_rate("H(n=1)", "H+", 2.0, 1e18)


def test_vmapped_reduction_matches_the_batched_path(collisional) -> None:
    conditions = Conditions(1.0, 1e18)
    temperatures = np.array([1.0, 5.0, 30.0])
    densities = np.array([1e17, 1e20])
    grid_t, grid_n = np.meshgrid(temperatures, densities, indexing="ij")
    jm = JaxModel(collisional, conditions)
    reduction = jm.reduction(grid_t, grid_n)
    data = batched_scan(collisional, conditions, temperatures, densities)
    shape = (3, 2)
    assert reduction.effective.reshape(*shape, 2, 2) / grid_n[
        ..., None, None
    ] == pytest.approx(data["effective_rate"].values, rel=1e-8)
    assert reduction.pec.reshape(*shape, *reduction.pec.shape[1:]) == pytest.approx(
        data["pec"].values, rel=1e-8
    )
    assert (
        reduction.drivers == ("H(n=1)", "H+")
        and reduction.to_dataset()["pec"].ndim == 3
    )
    everything = Conditions(
        1.0, 1e18, densities={n: 1.0 for n in collisional.state_names}
    )
    with pytest.raises(ValueError, match="nothing to reduce"):
        JaxModel(collisional, everything).reduction(2.0, 1e18)


def test_singular_reduction_is_reported() -> None:
    (g, x), ion = hydrogen_levels(2)
    lossless = ReactionNetwork(
        [g, x, ion],
        [
            Reaction("exc", (E, g), (E, x), "excitation", Constant(value=1e-15)),
            Reaction(
                "rr", (E, ion), (g,), "radiative_recombination", Constant(value=1e-19)
            ),
        ],
    )
    jm = JaxModel(
        opencr.Model(lossless, drivers=("H(n=1)", "H+")), Conditions(1.0, 1e18)
    )
    with pytest.raises(SingularSystemError):
        jm.reduction(1.0, 1e18)


def test_gradient_of_an_effective_rate(collisional) -> None:
    jm = JaxModel(collisional, Conditions(1.0, 1e18))
    s_cr = jm.effective_rate("H(n=1)", "H+", 5.0, 1e18)
    reference = collisional.effective_rates(
        Conditions(5.0, 1e18, densities={"H(n=1)": 1.0, "H+": 1.0})
    ).effective_rate("H(n=1)", "H+")
    assert float(s_cr) == pytest.approx(reference, rel=1e-8)
    gradient = jax.grad(jm.effective_rate, argnums=(2, 3))("H(n=1)", "H+", 5.0, 1e18)

    def at(t: float, n: float) -> float:
        return float(jm.effective_rate("H(n=1)", "H+", t, n))

    d_t = (at(5.0 * (1 + 1e-6), 1e18) - at(5.0 * (1 - 1e-6), 1e18)) / (1e-5)
    d_n = (at(5.0, 1e18 * (1 + 1e-6)) - at(5.0, 1e18 * (1 - 1e-6))) / (2e12)
    assert float(gradient[0]) == pytest.approx(d_t, rel=1e-5)
    assert float(gradient[1]) == pytest.approx(d_n, rel=1e-4)
