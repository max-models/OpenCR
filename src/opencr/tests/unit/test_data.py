"""The data layer: checks, the native format, importers and the hydrogen data."""

from __future__ import annotations

import json

import numpy as np
import pytest
from plasmacoll import CrossSection

from opencr.core import (
    ELECTRON_STATE,
    Arrhenius,
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
from opencr.data import DataError, check_network_data, load_network, save_network
from opencr.data.checks import (
    check_cross_section,
    check_levels,
    check_provenance,
    check_rate_table,
    raise_problems,
)
from opencr.data.hydrogen import (
    IMPACT_PROVENANCE,
    einstein_a,
    hydrogen_levels,
    level_energy,
    radiative_recombination,
    scaled_exp1,
)
from opencr.data.importers import lxcat_reactions, read_adf11, read_rate_table
from opencr.data.native import network_from_dict, network_to_dict, rate_from_dict
from opencr.data.provenance import Provenance, Reference

E = ELECTRON_STATE
LEVELS, ION = hydrogen_levels(3)
G, X, Y = LEVELS


def network() -> ReactionNetwork:
    sigma = CrossSection.constant(1e-20, threshold=10.2)
    return ReactionNetwork(
        [G, X, Y, ION],
        [
            Reaction(
                "exc",
                (E, G),
                (E, X),
                "excitation",
                CrossSectionRate(cross_section=sigma),
                IMPACT_PROVENANCE,
            ),
            Reaction(
                "dex", (E, X), (E, G), "deexcitation", DetailedBalance(forward="exc")
            ),
            Reaction(
                "ion",
                (E, Y),
                (E, E, ION),
                "ionization",
                Arrhenius(a=1e-14, b=0.5, c=1.5),
                IMPACT_PROVENANCE,
            ),
            Reaction(
                "rr",
                (E, ION),
                (G,),
                "radiative_recombination",
                Function(function=lambda t: 1e-19 / t, label="1/T"),
                IMPACT_PROVENANCE,
            ),
            Reaction(
                "rad",
                (X,),
                (G,),
                "spontaneous_emission",
                Constant(value=einstein_a(2, 1)),
                IMPACT_PROVENANCE,
                escape=0.5,
            ),
            Reaction(
                "tab",
                (E, G),
                (E, Y),
                "excitation",
                Tabulated(
                    temperatures=(1.0, 10.0), values=(1e-16, 1e-14), threshold=12.0
                ),
                IMPACT_PROVENANCE,
            ),
        ],
        name="test",
        assumptions=["n-bundled"],
    )


def test_native_round_trip(tmp_path) -> None:
    original = network()
    path = save_network(original, tmp_path / "net.json")
    loaded = load_network(path)
    assert loaded.state_names == original.state_names
    assert [r.name for r in loaded.reactions] == [r.name for r in original.reactions]
    assert loaded.assumptions == ("n-bundled",)
    assert isinstance(
        loaded.reaction("rr").rate, Tabulated
    )  # functions are written as tables
    assert loaded.reaction("rad").escape == 0.5
    assert loaded.reaction("exc").provenance == IMPACT_PROVENANCE
    assert loaded.reaction("ion").rate == original.reaction("ion").rate
    assert loaded.state("H(n=2)") == X
    cs = loaded.reaction("exc").rate.cross_section
    assert cs.threshold == pytest.approx(10.2)


def test_native_format_is_checked(tmp_path) -> None:
    data = network_to_dict(network())
    with pytest.raises(DataError, match="format"):
        network_from_dict({**data, "format": "other"})
    with pytest.raises(DataError, match="version"):
        network_from_dict({**data, "version": 99})
    with pytest.raises(DataError, match="missing 'states'"):
        network_from_dict({k: v for k, v in data.items() if k != "states"})
    bad = json.loads(json.dumps(data))
    bad["states"][0]["species"] = "Q"
    with pytest.raises(DataError, match="unknown species"):
        network_from_dict(bad)
    bad = json.loads(json.dumps(data))
    bad["reactions"][0]["products"] = ["e", "H(n=9)"]
    with pytest.raises(DataError, match="unknown states"):
        network_from_dict(bad)
    with pytest.raises(DataError, match="unknown rate model"):
        rate_from_dict({"kind": "magic"})
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    with pytest.raises(DataError, match="not JSON"):
        load_network(path)


def test_cross_section_checks() -> None:
    assert check_cross_section([0.0, 1.0], [0.0, 1e-20]) == []
    assert "equal length" in check_cross_section([0.0, 1.0], [1.0])[0]
    problems = check_cross_section(
        [2.0, 1.0, -1.0], [1e-20, -1.0, 1e-10], threshold=1.5
    )
    text = " ".join(problems)
    for expected in (
        "not increasing",
        "negative energies",
        "negative cross",
        "below the threshold",
        "cm^2",
    ):
        assert expected in text
    assert "missing" in check_cross_section([0.0, np.nan], [1.0, 1.0])[0]
    assert "fewer than two" in check_cross_section([1.0], [1.0])[0]
    assert "negative threshold" in " ".join(
        check_cross_section([0.0, 1.0], [0.0, 0.0], -1.0)
    )


def test_rate_table_checks() -> None:
    assert check_rate_table([1.0, 2.0], [1e-14, 2e-14]) == []
    assert "equal length" in check_rate_table([1.0], [1.0, 2.0])[0]
    text = " ".join(check_rate_table([0.0, -1.0], [1.0, -1.0]))
    assert "increasing" in text and "negative rates" in text
    assert "missing" in check_rate_table([1.0, 2.0], [np.inf, 1.0])[0]
    assert "fewer" in check_rate_table([1.0], [1.0])[0]


def test_level_and_provenance_checks() -> None:
    assert check_levels(LEVELS) == []
    h = species_from_formula("H")
    wrong = [State(h, "n=1", 0.0, 2.0, {"n": 1}), State(h, "n=2", -1.0, 8.0, {"n": 2})]
    assert "do not increase" in check_levels(wrong)[0]
    assert "twice" in check_levels([G, G])[0]
    assert "finite" in check_levels([State(h, "x", np.nan)])[0]
    assert check_provenance(None)[0].endswith("no provenance")
    text = " ".join(check_provenance(Provenance(Reference(""))))
    assert "citation" in text and "licence" in text and "units" in text
    assert check_provenance(IMPACT_PROVENANCE) == []
    with pytest.raises(DataError):
        raise_problems(["x"])
    raise_problems([])
    assert check_network_data(network()) == []
    unsourced = ReactionNetwork(
        [G, X], [Reaction("r", (X,), (G,), "spontaneous_emission", Constant(value=1.0))]
    )
    assert "no provenance" in check_network_data(unsourced)[0]


def test_read_rate_table(tmp_path) -> None:
    path = tmp_path / "rate.txt"
    path.write_text("# T [eV]  k [cm^3/s]\n1 1e-9\n10 1e-8\n")
    rate, provenance = read_rate_table(
        path, Reference("me"), unit_factor=1e-6, licence="MIT"
    )
    assert rate.values == pytest.approx((1e-15, 1e-14))
    assert provenance.conversions == ("rates times 1e-06 to SI",)
    path.write_text("1 2 3\n4 5 6\n")
    with pytest.raises(DataError, match="two columns"):
        read_rate_table(path, Reference("me"))
    path.write_text("2 1\n1 1\n")
    with pytest.raises(DataError, match="increasing"):
        read_rate_table(path, Reference("me"))


LXCAT = """\
EXCITATION
Ar -> Ar*(11.5eV)
 1.150000e+1
PROCESS: E + Ar -> E + Ar*(11.5eV), Excitation
-----------------------------
 1.150000e+1	0.000000e+0
 2.000000e+1	1.000000e-21
-----------------------------

IONIZATION
Ar -> Ar^+
 1.576000e+1
-----------------------------
 1.576000e+1	0.000000e+0
 5.000000e+1	2.000000e-20
-----------------------------

ELASTIC
Ar
 1.360000e-5
-----------------------------
 0.000000e+0	7.500000e-20
 1.000000e+1	1.000000e-19
-----------------------------
"""


def test_lxcat_reactions(tmp_path) -> None:
    path = tmp_path / "ar.txt"
    path.write_text(LXCAT)
    ar = species_from_formula("Ar")
    ground = State(ar, "", 0.0, 1.0)
    star = State(ar, "4s", 11.5, 12.0)
    ion = State(species_from_formula("Ar+"), "", 15.76, 6.0)
    reactions = lxcat_reactions(
        path,
        ground,
        {"Ar*(11.5eV)": (star,), "Ar^+": (ion,)},
        "test set",
        retrieved="2026-10-27",
    )
    assert [r.kind for r in reactions] == ["excitation", "ionization"]
    assert reactions[1].products == (E, E, ion)
    assert "no redistribution" in reactions[0].provenance.licence
    assert reactions[0].provenance.reference.detail.startswith("E + Ar")
    only_ion = lxcat_reactions(
        path,
        ground,
        lambda kind, product: (ion,) if kind == "IONIZATION" else None,
        "x",
    )
    assert len(only_ion) == 1


def test_hydrogen_levels() -> None:
    levels, ion = hydrogen_levels(4)
    assert [s.name for s in levels] == ["H(n=1)", "H(n=2)", "H(n=3)", "H(n=4)"]
    assert (
        levels[1].weight == 8.0
        and ion.name == "H+"
        and ion.energy == pytest.approx(13.598, abs=1e-3)
    )
    assert level_energy(2) == pytest.approx(10.1988, abs=1e-4)
    deuterium, d_ion = hydrogen_levels(2, isotope="D")
    assert deuterium[0].name == "D(n=1)" and d_ion.name == "D+"
    with pytest.raises(ValueError):
        hydrogen_levels(0)
    with pytest.raises(ValueError):
        level_energy(0)


@pytest.mark.parametrize(
    ("upper", "lower", "exact"),
    # n-bundled Einstein coefficients of H, NIST ASD (Wiese and Fuhr 2009)
    [
        (2, 1, 4.6986e8),
        (3, 1, 5.5751e7),
        (3, 2, 4.4101e7),
        (4, 2, 8.4193e6),
        (4, 3, 8.9860e6),
    ],
)
def test_einstein_coefficients(upper, lower, exact) -> None:
    assert einstein_a(upper, lower) == pytest.approx(exact, rel=0.01)


def test_einstein_coefficient_input() -> None:
    with pytest.raises(ValueError):
        einstein_a(1, 2)


def test_radiative_recombination() -> None:
    # the 1s coefficient at 1e4 K (0.8617 eV): 1.58e-13 cm^3/s (Osterbrock and Ferland)
    assert radiative_recombination(1, 0.8617) == pytest.approx(1.58e-19, rel=0.05)
    # case A, summed over all levels: 4.18e-13 cm^3/s at 1e4 K
    total = sum(radiative_recombination(n, 0.8617) for n in range(1, 400))
    assert total == pytest.approx(4.18e-19, rel=0.06)
    # decreasing with temperature
    t = np.array([0.1, 1.0, 10.0])
    assert np.all(np.diff(radiative_recombination(5, t)) < 0.0)
    with pytest.raises(ValueError):
        radiative_recombination(0, 1.0)


def test_scaled_exp1_is_continuous() -> None:
    from scipy.special import exp1

    x = np.array([0.1, 10.0, 49.9, 50.1, 300.0])
    values = scaled_exp1(x)
    assert values[:3] == pytest.approx(np.exp(x[:3]) * exp1(x[:3]))
    assert values[3] == pytest.approx(np.exp(50.1) * exp1(50.1), rel=1e-6)
    assert values[4] == pytest.approx(1.0 / 301.0, rel=1e-3)


def test_lazy_data_exports() -> None:
    import opencr.data

    assert opencr.data.Reference is Reference
    with pytest.raises(AttributeError):
        opencr.data.nothing  # noqa: B018


ADF11 = """\
    1    3    2    1    1     /HYDROGEN           /TEST
--------------------------------------------------------------------------------
   8.00000   9.00000  10.00000
   0.00000   1.00000
---------------------/ IPRT= 1  / IGRD= 1  /--------/ Z1= 1   / DATE= 01/01/26
 -10.00000 -10.00000 -10.00000
  -9.00000  -9.00000  -8.00000
C-----------------------------------------------------------------------
"""


def test_read_adf11(tmp_path) -> None:
    path = tmp_path / "test.dat"
    path.write_text(ADF11)
    table = read_adf11(path)
    assert table.density == pytest.approx([1e14, 1e15, 1e16])
    assert table.temperature == pytest.approx([1.0, 10.0])
    assert table.coefficient[1, 2] == pytest.approx(1e-14)
    assert table(10.0, 1e14) == pytest.approx(1e-15)
    assert table(np.sqrt(10.0), 1e14) == pytest.approx(np.sqrt(1e-16 * 1e-15))
    assert "TEST" in table.header
    with pytest.raises(DataError):
        read_adf11(path, block=3)
