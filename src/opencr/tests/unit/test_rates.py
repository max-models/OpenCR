"""Rate coefficients: exact Maxwellian integrals, EEDFs and detailed balance."""

from __future__ import annotations

import math

import numpy as np
import pytest
from plasmacoll import CrossSection, rate_coefficient
from plasmacoll.atomic_hydrogen import excitation_fit, ionization_fit

from opencr.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, thermal_de_broglie_cubed
from opencr.core import (
    ELECTRON_STATE,
    Arrhenius,
    Conditions,
    Constant,
    CrossSectionRate,
    DetailedBalance,
    Function,
    Reaction,
    ReactionNetwork,
    State,
    Tabulated,
    species_from_formula,
)
from opencr.data.hydrogen import hydrogen_levels
from opencr.kinetics.eedf import Generalized, Maxwellian, TabulatedEEDF
from opencr.kinetics.rates import (
    MaxwellianIntegral,
    RateEvaluator,
    inverse_cross_section,
)

E = ELECTRON_STATE
(G, X, Y), ION = hydrogen_levels(3)


@pytest.mark.parametrize(
    "fit",
    [
        excitation_fit(1, 2),
        excitation_fit(1, 3),
        excitation_fit(3, 5),
        ionization_fit(1),
        ionization_fit(5),
    ],
)
@pytest.mark.parametrize("kt", [0.3, 1.0, 10.0, 200.0])
def test_exact_maxwellian_matches_plasmacolls_quadrature(fit, kt) -> None:
    cs = fit.cross_section()
    integral = MaxwellianIntegral.from_rate(CrossSectionRate(cross_section=cs))
    k = math.exp(integral.log_scaled(np.array(kt)) - integral.threshold / kt)
    assert k == pytest.approx(
        rate_coefficient(cs, kt, ELECTRON_MASS, num=40000), rel=2e-3
    )


def test_a_constant_cross_section_gives_sigma_times_the_mean_speed() -> None:
    integral = MaxwellianIntegral.from_rate(
        CrossSectionRate(cross_section=CrossSection.constant(1e-20))
    )
    t = np.array([0.1, 1.0, 100.0])
    mean_speed = np.sqrt(8 * t * ELEMENTARY_CHARGE / (math.pi * ELECTRON_MASS))
    assert np.exp(integral.log_scaled(t)) == pytest.approx(
        1e-20 * mean_speed, rel=1e-12
    )


def test_heavy_particle_lab_frame() -> None:
    m = 1.67e-27
    cs = CrossSection.constant(1e-19, threshold=1.0)
    rate = CrossSectionRate(
        depends_on="ion", cross_section=cs, mass=m / 2, incident_mass=m
    )
    integral = MaxwellianIntegral.from_rate(rate)
    assert integral.threshold == pytest.approx(0.5)
    k = math.exp(integral.log_scaled(np.array(2.0)) - integral.threshold / 2.0)
    assert k == pytest.approx(
        rate_coefficient(cs, 2.0, m / 2, incident_mass=m, num=40000), rel=2e-3
    )


def network() -> ReactionNetwork:
    exc = excitation_fit(1, 2).cross_section()
    ion = ionization_fit(1).cross_section()
    return ReactionNetwork(
        [G, X, Y, ION],
        [
            Reaction(
                "exc", (E, G), (E, X), "excitation", CrossSectionRate(cross_section=exc)
            ),
            Reaction(
                "dex", (E, X), (E, G), "deexcitation", DetailedBalance(forward="exc")
            ),
            Reaction(
                "ion",
                (E, G),
                (E, E, ION),
                "ionization",
                CrossSectionRate(cross_section=ion),
            ),
            Reaction(
                "3b",
                (E, E, ION),
                (E, G),
                "three_body_recombination",
                DetailedBalance(forward="ion"),
            ),
            Reaction(
                "arr", (E, X), (E, Y), "excitation", Arrhenius(a=1e-14, b=0.5, c=1.9)
            ),
            Reaction(
                "arr-1", (E, Y), (E, X), "deexcitation", DetailedBalance(forward="arr")
            ),
            Reaction(
                "tab",
                (E, Y),
                (E, E, ION),
                "ionization",
                Tabulated(
                    temperatures=(0.1, 1.0, 10.0),
                    values=(1e-20, 1e-15, 1e-13),
                    threshold=1.5,
                ),
            ),
            Reaction(
                "rr",
                (E, ION),
                (Y,),
                "radiative_recombination",
                Function(function=lambda t: 1e-19 / np.sqrt(t)),
            ),
            Reaction(
                "a21",
                (X,),
                (G,),
                "spontaneous_emission",
                Constant(value=4.7e8),
                escape=0.5,
            ),
            Reaction("a32", (Y,), (X,), "spontaneous_emission", Constant(value=4.4e7)),
        ],
    )


def test_detailed_balance_maxwellian() -> None:
    evaluator = RateEvaluator(network())
    assert evaluator.names[0] == "exc"
    for kt in (0.5, 2.0, 20.0):
        k = dict(
            zip(evaluator.names, evaluator.coefficients(Conditions(kt)), strict=True)
        )
        gap = X.energy - G.energy
        assert k["dex"] / k["exc"] == pytest.approx(
            G.weight / X.weight * math.exp(gap / kt), rel=1e-10
        )
        saha = (
            G.weight
            / (E.weight * ION.weight)
            * thermal_de_broglie_cubed(kt)
            * math.exp(ION.energy / kt)
        )
        assert k["3b"] / k["ion"] == pytest.approx(saha, rel=1e-10)
        assert k["arr-1"] / k["arr"] == pytest.approx(
            X.weight / Y.weight * math.exp((Y.energy - X.energy) / kt), rel=1e-10
        )
        assert k["arr"] == pytest.approx(1e-14 * kt**0.5 * math.exp(-1.9 / kt))
        assert k["rr"] == pytest.approx(1e-19 / math.sqrt(kt))
        assert k["a21"] == pytest.approx(0.5 * 4.7e8)


def test_detailed_balance_at_very_low_temperature_stays_finite() -> None:
    k = RateEvaluator(network()).coefficients(Conditions(0.02))
    assert np.all(np.isfinite(k))
    names = RateEvaluator(network()).names
    assert k[names.index("exc")] < 1e-200 and k[names.index("dex")] > 0.0


def test_tabulated_interpolation_removes_the_boltzmann_factor() -> None:
    t = np.geomspace(0.1, 10.0, 5)
    values = 1e-14 * np.exp(-3.0 / t)
    rate = Tabulated(temperatures=tuple(t), values=tuple(values), threshold=3.0)
    net = ReactionNetwork([G, X], [Reaction("t", (E, G), (E, X), "excitation", rate)])
    evaluator = RateEvaluator(net)
    for kt in (0.15, 0.7, 4.0, 50.0):
        expected = 1e-14 * math.exp(-3.0 / min(kt, 10.0))
        if kt > 10.0:
            expected = 1e-14 * math.exp(-3.0 / 10.0) * math.exp(3.0 / 10.0 - 3.0 / kt)
        assert evaluator.coefficients(Conditions(kt))[0] == pytest.approx(
            expected, rel=1e-10
        )


def test_scan_matches_single_evaluations() -> None:
    evaluator = RateEvaluator(network())
    temperatures = np.array([0.5, 3.0, 30.0])
    scan = evaluator.coefficients_scan(
        Conditions(1.0, escape={"a21": 0.25}), temperatures
    )
    for row, kt in zip(scan, temperatures, strict=True):
        assert row == pytest.approx(
            evaluator.coefficients(Conditions(kt, escape={"a21": 0.25})), rel=1e-12
        )
    with pytest.raises(ValueError):
        evaluator.coefficients_scan(Conditions(1.0), np.array([0.0]))


def test_tabulated_maxwellian_eedf_gives_the_maxwellian_rates() -> None:
    evaluator = RateEvaluator(network())
    kt = 3.0
    maxwell = evaluator.coefficients(Conditions(kt))
    table = Maxwellian(kt).tabulate(4000)
    via_eedf = evaluator.coefficients(Conditions.from_eedf(table))
    names = evaluator.names
    for name in ("exc", "dex", "ion"):
        i = names.index(name)
        assert via_eedf[i] == pytest.approx(maxwell[i], rel=5e-3), name


def test_druyvesteyn_has_fewer_fast_electrons() -> None:
    evaluator = RateEvaluator(network())
    names = evaluator.names
    maxwell = evaluator.coefficients(Conditions(2.0))
    druyvesteyn = evaluator.coefficients(Conditions.from_eedf(Generalized(3.0, 2.0)))
    assert druyvesteyn[names.index("ion")] < 0.1 * maxwell[names.index("ion")]
    # rates that are not cross sections use the effective temperature, 2/3 <eps> = 2 eV
    assert druyvesteyn[names.index("arr")] == pytest.approx(maxwell[names.index("arr")])


def test_inverse_cross_section_is_klein_rosseland() -> None:
    cs = CrossSection.constant(2e-20, threshold=10.0)
    inverse = inverse_cross_section(cs, 0.25, 10.0)
    assert inverse(np.array([5.0]))[0] == pytest.approx(0.25 * 15.0 / 5.0 * 2e-20)


def test_unsupported_inverses_are_refused() -> None:
    h2 = State(species_from_formula("H2"), "", -4.5, 1.0)
    bad = [
        Reaction("dis", (E, h2), (E, G, G), "dissociation", Arrhenius(a=1e-15, c=10.0)),
        Reaction("asso", (E, G, G), (E, h2), "other", DetailedBalance(forward="dis")),
    ]
    with pytest.raises(NotImplementedError, match="heavy particles"):
        RateEvaluator(ReactionNetwork([G, h2], bad))
    constant = [
        Reaction("c", (E, G), (E, X), "excitation", Constant(value=1e-15)),
        Reaction("c-1", (E, X), (E, G), "deexcitation", DetailedBalance(forward="c")),
    ]
    with pytest.raises(NotImplementedError, match="no temperature"):
        RateEvaluator(ReactionNetwork([G, X], constant))
    heavy = [
        Reaction(
            "h",
            (E, G),
            (E, E, ION),
            "ionization",
            Arrhenius(depends_on="gas", a=1e-15, c=13.6),
        ),
        Reaction(
            "h-1",
            (E, E, ION),
            (E, G),
            "three_body_recombination",
            DetailedBalance(forward="h"),
        ),
    ]
    with pytest.raises(NotImplementedError, match="Saha"):
        RateEvaluator(ReactionNetwork([G, ION], heavy))


def test_eedfs() -> None:
    energy = np.linspace(0.0, 200.0, 20001)
    for eedf in (Maxwellian(2.0), Generalized(3.0, 2.0), Generalized(3.0, 1.0)):
        f = eedf.eepf(energy)
        assert np.trapezoid(np.sqrt(energy) * f, energy) == pytest.approx(1.0, rel=1e-4)
        assert np.trapezoid(energy**1.5 * f, energy) == pytest.approx(
            eedf.mean_energy, rel=1e-4
        )
        assert eedf.effective_temperature == pytest.approx(2.0)
        assert repr(eedf)
    assert Generalized(3.0, 1.0).eepf(energy) == pytest.approx(
        Maxwellian(2.0).eepf(energy)
    )
    assert Generalized(3.0, 1.0).is_maxwellian and not Generalized(3.0).is_maxwellian
    table = TabulatedEEDF(energy, 5.0 * Maxwellian(2.0).eepf(energy))
    assert table.mean_energy == pytest.approx(3.0, rel=1e-4)
    assert (
        table.tabulate() is table
        and table.grid() is table.energy
        and "mean" in repr(table)
    )
    assert table.eepf(np.array([1e6]))[0] == 0.0
    for bad in (
        (energy[:1], energy[:1]),
        (energy[::-1], energy),
        (energy, -energy),
        (energy, 0 * energy),
    ):
        with pytest.raises(ValueError):
            TabulatedEEDF(*bad)
    for bad in (lambda: Maxwellian(0.0), lambda: Generalized(-1.0)):
        with pytest.raises(ValueError):
            bad()
    from opencr.kinetics.eedf import EEDF

    with pytest.raises(NotImplementedError):
        EEDF().eepf(energy)
    with pytest.raises(NotImplementedError):
        EEDF().mean_energy  # noqa: B018


def test_eedf_from_boltzmann() -> None:
    class Solution:
        energy = np.linspace(0.0, 50.0, 501)
        eepf = np.exp(-np.linspace(0.0, 50.0, 501) / 2.0)

    table = TabulatedEEDF.from_boltzmann(Solution())
    assert table.effective_temperature == pytest.approx(2.0, rel=5e-3)
