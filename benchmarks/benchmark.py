"""Benchmarks of OpenCR's reference path, the batched cunumpy path and the JAX backend.

Run from the repository::

    python benchmarks/benchmark.py            # the table (markdown)
    python benchmarks/benchmark.py --json     # as JSON
    python benchmarks/benchmark.py --repeat 20 --n-max 20 --points 2000

Every case is a scope-profiler region, repeated ``--repeat`` times after one
warm-up call, so the table separates the **first** call (which includes the
JIT compilation of the JAX cases and the import of their modules) from the
**median** of the repeats. The cases:

======================  ========================================================
construction            building the atomic hydrogen network and model
rates                   the rate coefficients of one set of conditions
jacobian                the sparse analytic Jacobian of the population equations
quasi-static            one quasi-static solve (sparse LU)
full quasi-neutral      a closed steady state with quasi-neutral n_e
transient               BDF from a neutral gas to 1e-2 s, 50 output times
scan (reference)        ``opencr.scan`` over ``--points`` (T_e, n_e) points
batched scan (cunumpy)  ``batched_scan`` over the same points on cunumpy's xp
jax rates               the jitted rates at the same temperatures
jax reduction           the vmapped quasi-static reduction of the same points
jax jacobian            the AD Jacobian of the population equations
======================  ========================================================

The GPU is not benchmarked here: cunumpy runs on CuPy when
``cunumpy.set_backend("cupy")`` succeeds, and JAX on its default device; a
speed-up is only claimed for a measured end-to-end gain, with the copies in
and out included (they are inside the regions).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import tempfile

import cunumpy as xp
import numpy as np
import scope_profiler as sp

import opencr
from opencr.analysis.scan import scan
from opencr.backends import device
from opencr.backends.batched import batched_scan


def cases(n_max: int, points: int, jax_cases: bool):
    """Return the benchmark cases as ``(name, setup(), call)`` pairs."""
    model = opencr.load_model("hydrogen/atomic/basic", n_max=n_max)
    drivers = opencr.Conditions(3.0, 1e18, densities={"H(n=1)": 1e19, "H+": 1e18})
    closed = opencr.Conditions(3.0, None, totals={"H": 1e19})
    system = model.system(drivers)
    n = np.full(system.size, 1e12)
    side = max(int(round(points**0.5)), 1)
    temperatures = np.geomspace(0.5, 100.0, side)
    densities = np.geomspace(1e16, 1e22, side)
    times = np.geomspace(1e-9, 1e-2, 50)
    out = [
        (
            "construction",
            lambda: opencr.load_model("hydrogen/atomic/basic", n_max=n_max),
        ),
        ("rates", lambda: model.evaluator.coefficients(drivers)),
        ("jacobian", lambda: system.jacobian(n)),
        ("quasi-static", lambda: model.solve_steady_state(drivers)),
        ("full quasi-neutral", lambda: model.solve_steady_state(closed, mode="full")),
        ("transient", lambda: model.solve_transient(closed, {"H(n=1)": 1e19}, times)),
        (
            f"scan (reference, {side * side} points)",
            lambda: scan(
                model,
                drivers,
                lines=False,
                electron_temperature=temperatures,
                electron_density=densities,
            ),
        ),
        (
            f"batched scan (cunumpy, {side * side} points)",
            lambda: batched_scan(model, drivers, temperatures, densities),
        ),
    ]
    if jax_cases:
        from opencr.backends.jax_backend import JaxModel

        grid_t, grid_n = np.meshgrid(temperatures, densities, indexing="ij")
        jm = JaxModel(model, drivers)
        out += [
            (
                f"jax rates ({side * side} temperatures)",
                lambda: jm.rates(grid_t.ravel()).block_until_ready(),
            ),
            (
                f"jax reduction ({side * side} points)",
                lambda: jm.reduction(grid_t, grid_n),
            ),
            ("jax jacobian", lambda: jm.jacobian(n, 3.0, 1e18).block_until_ready()),
        ]
    return out


def run(n_max: int, points: int, repeat: int, jax_cases: bool) -> list[dict]:
    """Run every case and return its first and median times in seconds."""
    rows = []
    path = os.path.join(tempfile.mkdtemp(), "benchmark.h5")
    with sp.session(output=path, return_results=True, verbose=False) as session:
        for name, call in cases(n_max, points, jax_cases):
            for _ in range(repeat + 1):
                with sp.region(name):
                    call()
    names = [name for name, _ in cases(n_max, points, jax_cases)]
    for region in session.results.summary():
        if region["name"] not in names:
            continue
        durations = list(session.results.get_region(region["name"]).durations)
        rows.append(
            {
                "case": region["name"],
                "first": durations[0],
                "median": statistics.median(durations[1:])
                if len(durations) > 1
                else durations[0],
                "calls": len(durations),
            }
        )
    return rows


def table(rows: list[dict], info: dict) -> str:
    """Return the rows as a markdown table with a header line on the machine."""
    lines = [
        f"OpenCR {info['opencr']} on {info['machine']}, cunumpy on the {info['device']}, "
        f"JAX {info['jax'] or 'not installed'}; {info['repeat']} repeats after the first call.",
        "",
        "| case | first (ms) | median (ms) |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {1e3 * row['first']:.2f} | {1e3 * row['median']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Run the benchmarks and print the table or JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--n-max", type=int, default=20, help="levels of the hydrogen model"
    )
    parser.add_argument("--points", type=int, default=1000, help="points of the scans")
    parser.add_argument(
        "--repeat", type=int, default=5, help="repeats after the first call"
    )
    parser.add_argument("--no-jax", action="store_true", help="skip the JAX cases")
    parser.add_argument(
        "--json", action="store_true", help="print JSON instead of a table"
    )
    args = parser.parse_args()
    jax_version = None
    if not args.no_jax:
        try:
            import jax

            jax_version = jax.__version__
        except ImportError:  # pragma: no cover
            pass
    rows = run(args.n_max, args.points, args.repeat, jax_version is not None)
    info = {
        "opencr": opencr.__version__,
        "machine": f"{platform.machine()} {platform.system()}",
        "device": device(),
        "cunumpy": xp.__version__,
        "jax": jax_version,
        "repeat": args.repeat,
        "n_max": args.n_max,
        "points": args.points,
    }
    if args.json:
        print(json.dumps({"info": info, "rows": rows}, indent=1))
    else:
        print(table(rows, info))


if __name__ == "__main__":
    main()
