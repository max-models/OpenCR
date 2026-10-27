"""The ``opencr`` command: list the predefined models and print their manifests.

``opencr list`` prints the models; ``opencr manifest <name> [--n-max N]``
prints a model's manifest (states, reactions, sources and assumptions) as
JSON; ``opencr effective <name> --te T --ne N`` prints the effective rate
coefficients between the drivers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import scope_profiler as sp

from opencr._profiling import PROFILE_TAGS
from opencr.core.conditions import Conditions
from opencr.models import available_models, load_model

__all__ = ["main"]


def _options(args: argparse.Namespace) -> dict:
    return {} if args.n_max is None else {"n_max": args.n_max}


@sp.profile("opencr.cli", tags=PROFILE_TAGS)
def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; return the exit status."""
    parser = argparse.ArgumentParser(
        prog="opencr",
        description="List the predefined models and print their manifests.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list the predefined models")
    manifest = commands.add_parser("manifest", help="print a model's manifest as JSON")
    manifest.add_argument("model")
    manifest.add_argument("--n-max", type=int, default=None)
    effective = commands.add_parser(
        "effective", help="print the effective rate coefficients"
    )
    effective.add_argument("model")
    effective.add_argument(
        "--te", type=float, required=True, help="electron temperature in eV"
    )
    effective.add_argument(
        "--ne", type=float, required=True, help="electron density in m^-3"
    )
    effective.add_argument("--n-max", type=int, default=None)
    args = parser.parse_args(argv)
    if args.command == "list":
        for name, description in available_models().items():
            print(f"{name}: {description}")
        return 0
    model = load_model(args.model, **_options(args))
    if args.command == "manifest":
        json.dump(model.manifest(), sys.stdout, indent=1)
        print()
        return 0
    conditions = Conditions(args.te, args.ne, densities={d: 1.0 for d in model.drivers})
    reduction = model.effective_rates(conditions)
    for source in reduction.drivers:
        for target in reduction.drivers:
            if source != target:
                rate = reduction.effective_rate(source, target)
                print(f"{source} -> {target}: {rate:.4e} m^3/s")
    return 0
