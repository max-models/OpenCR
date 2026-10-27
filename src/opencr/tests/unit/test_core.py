"""Species, states, rate models, reactions and networks."""

from __future__ import annotations

import numpy as np
import pytest
from plasmacoll import CrossSection

from opencr.constants import ELECTRON_MASS
from opencr.core import (
    ELECTRON,
    ELECTRON_STATE,
    Arrhenius,
    Constant,
    CrossSectionRate,
    DetailedBalance,
    Function,
    NetworkError,
    Reaction,
    ReactionNetwork,
    Species,
    State,
    Tabulated,
    species_from_formula,
)
from opencr.data.provenance import Provenance, Reference, Validity

H = species_from_formula("H")
HP = species_from_formula("H+")
G = State(H, "n=1", 0.0, 2.0, {"n": 1})
X = State(H, "n=2", 10.2, 8.0, {"n": 2})
ION = State(HP, "", 13.6, 1.0)
E = ELECTRON_STATE
SIGMA = CrossSection.constant(1e-20, threshold=10.2)


def two_level(**kwargs) -> ReactionNetwork:
    reactions = [
        Reaction(
            "exc", (E, G), (E, X), "excitation", CrossSectionRate(cross_section=SIGMA)
        ),
        Reaction("dex", (E, X), (E, G), "deexcitation", DetailedBalance(forward="exc")),
        Reaction("rad", (X,), (G,), "spontaneous_emission", Constant(value=4.7e8)),
    ]
    return ReactionNetwork([G, X], reactions, name="two-level", **kwargs)


def test_formulas() -> None:
    h2p = species_from_formula("H2+")
    assert (h2p.charge, dict(h2p.composition)) == (1, {"H": 2})
    hm = species_from_formula("H-")
    assert hm.charge == -1 and hm.mass > H.mass
    assert species_from_formula("Ar++").charge == 2
    assert species_from_formula("e") is ELECTRON
    for bad in ("Xx", "H2*", "H+-"):
        with pytest.raises(ValueError):
            species_from_formula(bad)


def test_species_and_state_checks() -> None:
    with pytest.raises(ValueError):
        Species("", 0, 1.0)
    with pytest.raises(ValueError):
        Species("H", 0.5, 1.0)
    with pytest.raises(ValueError):
        Species("H", 0, -1.0)
    with pytest.raises(ValueError):
        Species("H", 0, 1.0, {"H": 0})
    with pytest.raises(ValueError):
        State(H, "n=1", weight=0.0)
    assert G.name == "H(n=1)" and ION.name == "H+" and G.charge == 0
    assert G == State(H, "n=1", 0.0, 2.0, {"n": 1}) and G != X
    assert hash(G) == hash(State(H, "n=1"))
    assert G.to_dict()["quantum_numbers"] == {"n": 1}
    assert (
        H == species_from_formula("H")
        and H != HP
        and hash(H) == hash(species_from_formula("H"))
    )
    assert (G == 1) is False and (H == 1) is False


def test_rate_models_check_their_data() -> None:
    with pytest.raises(ValueError):
        Constant(value=-1.0)
    with pytest.raises(ValueError):
        Arrhenius(a=-1.0)
    with pytest.raises(ValueError):
        Constant(value=1.0, depends_on="photons")
    with pytest.raises(TypeError):
        CrossSectionRate(cross_section=None)
    with pytest.raises(ValueError):
        CrossSectionRate(cross_section=SIGMA, mass=-1.0)
    with pytest.raises(ValueError):
        CrossSectionRate(cross_section=SIGMA, depends_on="none")
    with pytest.raises(ValueError):
        Tabulated(temperatures=(1.0,), values=(1.0,))
    with pytest.raises(ValueError):
        Tabulated(temperatures=(2.0, 1.0), values=(1.0, 1.0))
    with pytest.raises(ValueError):
        Tabulated(temperatures=(1.0, 2.0), values=(1.0, -1.0))
    with pytest.raises(TypeError):
        Function(function=None)
    with pytest.raises(ValueError):
        DetailedBalance(forward="")
    rate = CrossSectionRate(cross_section=SIGMA)
    assert rate.threshold == pytest.approx(10.2) and rate.mass == ELECTRON_MASS
    data = rate.to_dict()
    assert data["kind"] == "cross_section" and data["threshold"] == pytest.approx(10.2)
    table = Function(function=lambda t: 1e-14 * t, threshold=1.0).tabulate()
    assert table.values[0] == pytest.approx(1e-16)
    assert Function(function=np.sqrt, label="root").to_dict()["label"] == "root"
    from opencr.core.rates import RateModel

    with pytest.raises(NotImplementedError):
        RateModel().to_dict()


def test_reactions_check_balance_and_order() -> None:
    reaction = Reaction(
        "ion", (E, G), (E, E, ION), "ionization", Arrhenius(a=1e-14, c=13.6)
    )
    assert reaction.order == 2
    assert reaction.net == {"e": 1, "H(n=1)": -1, "H+": 1}
    assert reaction.equation == "e + H(n=1) -> e + e + H+"
    assert reaction.energy_change == pytest.approx(13.6)
    assert reaction.photon_energy is None and not reaction.is_inverse
    rad = Reaction("rad", (X,), (G,), "spontaneous_emission", Constant(value=1.0))
    assert rad.photon_energy == pytest.approx(10.2) and rad.equation.endswith("+ hv")
    with pytest.raises(ValueError, match="charge"):
        Reaction("bad", (E, G), (E, ION), "ionization", Constant(value=1.0))
    with pytest.raises(ValueError, match="elements"):
        Reaction("bad", (G,), (), "excitation", Constant(value=1.0))
    with pytest.raises(ValueError, match="kind"):
        Reaction("bad", (X,), (G,), "magic", Constant(value=1.0))
    with pytest.raises(ValueError, match="name"):
        Reaction("", (X,), (G,), "other", Constant(value=1.0))
    with pytest.raises(ValueError, match="reactants"):
        Reaction("bad", (), (), "other", Constant(value=1.0))
    with pytest.raises(TypeError):
        Reaction("bad", (X,), (G,), "other", 1.0)
    with pytest.raises(ValueError, match="escape"):
        Reaction(
            "bad", (X,), (G,), "spontaneous_emission", Constant(value=1.0), escape=2.0
        )
    with pytest.raises(ValueError, match="one reactant"):
        Reaction("bad", (E, X), (E, G), "spontaneous_emission", Constant(value=1.0))
    with pytest.raises(ValueError, match="electron"):
        Reaction("bad", (G,), (X,), "excitation", CrossSectionRate(cross_section=SIGMA))
    # wall losses need not balance
    Reaction("wall", (ION,), (), "wall_loss", Constant(value=1e3))


def test_network_lookups_and_manifest() -> None:
    network = two_level(assumptions=["optically thin"])
    assert repr(network) == "ReactionNetwork('two-level', 2 states, 3 reactions)"
    assert network.state_names == ("H(n=1)", "H(n=2)")
    assert network.state("H(n=2)") == X and network.state_index("H(n=2)") == 1
    assert network.reaction("dex").rate.forward == "exc"
    assert {r.name for r in network.reactions_of("H(n=2)")} == {"exc", "dex", "rad"}
    assert network.species == (H,)
    for lookup in (network.state, network.state_index, network.reaction):
        with pytest.raises(KeyError):
            lookup("nothing")
    manifest = network.manifest()
    assert manifest["counts"]["by_kind"]["excitation"] == 1
    assert manifest["assumptions"] == ["optically thin"]
    assert manifest["reactions"][1]["forward"] == "exc"
    conservation = network.conservation()
    assert conservation.names == ("H",) and conservation.matrix.tolist() == [[1.0, 1.0]]


def test_network_problems_are_all_reported() -> None:
    reactions = [
        Reaction("exc", (E, G), (E, X), "excitation", Constant(value=1.0)),
        Reaction("exc", (E, X), (E, G), "excitation", Constant(value=1.0)),
        Reaction(
            "dex", (E, X), (E, G), "deexcitation", DetailedBalance(forward="none")
        ),
        Reaction(
            "dex2", (E, X), (E, G), "deexcitation", DetailedBalance(forward="ion")
        ),
        Reaction("up", (G,), (X,), "spontaneous_emission", Constant(value=1.0)),
        Reaction("ion", (E, G), (E, E, ION), "ionization", Constant(value=1.0)),
    ]
    other = State(H, "n=1", 0.5, 2.0)
    lonely = State(H, "n=3", 12.0, 18.0)
    with pytest.raises(NetworkError) as error:
        ReactionNetwork([G, G, X, lonely, State(ELECTRON)], reactions)
    text = "\n".join(error.value.problems)
    for expected in (
        "defined 2 times",
        "'exc' is defined 2 times",
        "forward reaction 'none' is missing",
        "is not the inverse",
        "'H+' is not in the network",
        "takes part in no reaction",
        "electron is implicit",
        "must raise the energy",
        "must lower the energy",
    ):
        assert expected in text, expected
    network = ReactionNetwork(
        [other, X],
        [Reaction("exc", (E, G), (E, X), "excitation", Constant(value=1.0))],
        check=False,
    )
    assert any("differs" in p for p in network.problems())
    chain = [
        Reaction("exc", (E, G), (E, X), "excitation", Constant(value=1.0)),
        Reaction("dex", (E, X), (E, G), "deexcitation", DetailedBalance(forward="exc")),
        Reaction("again", (E, G), (E, X), "excitation", DetailedBalance(forward="dex")),
    ]
    with pytest.raises(NetworkError, match="itself an inverse"):
        ReactionNetwork([G, X], chain)


def test_subset_and_merge() -> None:
    network = two_level()
    only_ground = network.subset(["H(n=1)"])
    assert only_ground.reactions == () and only_ground.state_names == ("H(n=1)",)
    with pytest.raises(KeyError):
        network.subset(["H(n=9)"])
    ion = Reaction("ion", (E, G), (E, E, ION), "ionization", Arrhenius(a=1e-14, c=13.6))
    extra = ReactionNetwork([G, ION], [ion])
    merged = network.merge(extra, name="merged")
    assert (
        merged.state_names == ("H(n=1)", "H(n=2)", "H+") and len(merged.reactions) == 4
    )
    clash = ReactionNetwork(
        [State(H, "n=1", 1.0, 2.0), ION],
        [
            Reaction(
                "i",
                (E, State(H, "n=1", 1.0, 2.0)),
                (E, E, ION),
                "ionization",
                Constant(value=1.0),
            )
        ],
    )
    with pytest.raises(NetworkError):
        network.merge(clash)
    # dropping the forward reaction drops its inverse too
    reduced = merged.subset(["H(n=1)", "H(n=2)", "H+"])
    assert len(reduced.reactions) == 4


def test_provenance_round_trip() -> None:
    provenance = Provenance(
        Reference("A. Author, J. 1, 1 (2000)", doi="10.1/x", detail="table 1"),
        units="m^2",
        conversions=["cm^2 -> m^2"],
        validity=[Validity("energy", 1.0)],
        uncertainty=0.2,
        licence="published-fit",
    )
    assert provenance.redistributable
    assert str(provenance.reference) == "A. Author, J. 1, 1 (2000), table 1, doi:10.1/x"
    data = provenance.to_dict()
    assert data["validity"][0]["maximum"] is None
    assert Provenance.from_dict(data) == provenance
    assert provenance.validity[0].contains(5.0) and not provenance.validity[0].contains(
        0.5
    )
    with pytest.raises(ValueError):
        Validity("energy", 2.0, 1.0)
    with pytest.raises(ValueError):
        Provenance(Reference("x"), uncertainty=-1.0)
    assert not Provenance(Reference("x")).redistributable
