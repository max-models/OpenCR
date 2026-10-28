"""The opt-in scope-profiler regions: off unless the application enables the tag."""

from __future__ import annotations

import numpy as np
import pytest
import scope_profiler as sp

from opencr.core import ELECTRON_STATE, Conditions, Constant, Reaction, ReactionNetwork
from opencr.data.hydrogen import hydrogen_levels
from opencr.kinetics.system import PopulationSystem
from opencr.solvers.steady import solve_steady


def main() -> None:
    (g, x), _ = hydrogen_levels(2)
    network = ReactionNetwork(
        [g, x],
        [
            Reaction(
                "exc",
                (ELECTRON_STATE, g),
                (ELECTRON_STATE, x),
                "excitation",
                Constant(value=1e-15),
            ),
            Reaction("rad", (x,), (g,), "spontaneous_emission", Constant(value=1e8)),
        ],
    )
    conditions = Conditions(1.0, 1e18, densities={"H(n=1)": 1e19})
    solve_steady(
        PopulationSystem(network, np.array([1e-15, 1e8]), conditions, ["H(n=2)"])
    )


@pytest.fixture(autouse=True)
def _fresh_profiler(monkeypatch):
    """Keep the profiling state of these tests out of the others."""
    monkeypatch.delenv("SCOPE_PROFILER_ENABLE_TAGS", raising=False)
    sp.reset()
    yield
    sp.reset()


def _region_names(**setup_kwargs) -> set[str]:
    sp.setup(output=None, **setup_kwargs)
    # an application region, recorded either way
    with sp.region("user.run"):
        main()
    results = sp.finalize(verbose=False, return_results=True)
    return set(results.region_names)


def test_regions_are_off_by_default() -> None:
    assert _region_names() == {"user.run"}


@pytest.mark.parametrize("tags", [["opencr"], ["*"]])
def test_enabling_the_tag_records_the_regions(tags: list[str]) -> None:
    assert _region_names(enable_tags=tags) == {"opencr.solve_steady", "user.run"}


def test_the_opt_in_tag_has_an_entry_point():
    """``scope-profiler tags`` finds the tag through the package's entry point."""
    from importlib.metadata import entry_points

    from opencr._profiling import PROFILE_TAGS

    found = {
        entry_point.name: entry_point.value
        for entry_point in entry_points(group="scope_profiler.opt_in_tags")
    }
    assert found.get("opencr") == "opencr"
    assert "opencr" in PROFILE_TAGS
