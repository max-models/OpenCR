"""Datasets of CR solutions for surrogates, with provenance and reproducible splits.

:func:`generate_dataset` samples conditions in given ranges (log-uniform, by
a seeded Latin hypercube) and solves the model at each sample, storing the
inputs, the quantities (:mod:`opencr.analysis.quantities`) and the
provenance: the model's manifest digest, the OpenCR version, the sampling
and the seed. The split into training, validation and test sets is part of
the dataset (``split``), drawn from its own seed, so every surrogate trained
on the dataset sees the same split.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr

from opencr.analysis.quantities import check_quantity, quantities
from opencr.analysis.results import _opencr_version
from opencr.analysis.scan import _apply
from opencr.core.conditions import Conditions
from opencr.model import Model

__all__ = ["SPLITS", "generate_dataset", "manifest_digest", "split_indices"]

#: The names of the splits.
SPLITS = ("train", "validation", "test")


def manifest_digest(model: Model) -> str:
    """Return the SHA-256 of the model's manifest (states, reactions, data)."""
    text = json.dumps(model.manifest(), sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _latin_hypercube(rng: np.random.Generator, samples: int, dims: int) -> np.ndarray:
    cells = (np.arange(samples)[:, None] + rng.random((samples, dims))) / samples
    for k in range(dims):
        cells[:, k] = rng.permutation(cells[:, k])
    return cells


def split_indices(
    samples: int, fractions: Sequence[float] = (0.7, 0.15, 0.15), seed: int = 0
) -> np.ndarray:
    """Return the split label (0 train, 1 validation, 2 test) of each sample."""
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("give three fractions that add up to 1")
    order = np.random.default_rng(seed).permutation(samples)
    bounds = np.round(np.cumsum(fractions)[:2] * samples).astype(int)
    labels = np.zeros(samples, dtype=int)
    labels[order[bounds[0] : bounds[1]]] = 1
    labels[order[bounds[1] :]] = 2
    return labels


def generate_dataset(
    model: Model,
    conditions: Conditions,
    bounds: Mapping[str, tuple[float, float]],
    outputs: Sequence[str],
    samples: int = 200,
    seed: int = 0,
    mode: str = "quasi-static",
    fractions: Sequence[float] = (0.7, 0.15, 0.15),
) -> xr.Dataset:
    """Solve ``model`` at log-uniform samples of the condition fields in ``bounds``.

    Args:
        model: The model.
        conditions: The conditions the sampled fields are set in.
        bounds: The range of each sampled field (as for :func:`opencr.scan`:
            ``electron_temperature``, ``electron_density``, ``density:<state>``).
        outputs: The quantities to store.
        samples: The number of samples.
        seed: The seed of the sampling; the split uses ``seed + 1``.
        mode: The steady-state mode.
        fractions: The train, validation and test fractions.

    Returns:
        A dataset over ``sample`` with ``inputs`` (sample, parameter),
        ``outputs`` (sample, output), ``split`` (0, 1, 2 for train, validation,
        test) and the provenance in its attributes (with ``solve_time``, the
        solver's time per sample in s).
    """
    for name in outputs:
        check_quantity(model, name)
    names = list(bounds)
    low = np.log([bounds[n][0] for n in names])
    high = np.log([bounds[n][1] for n in names])
    if np.any(high <= low):
        raise ValueError("every range needs low < high")
    rng = np.random.default_rng(seed)
    x = np.exp(low + (high - low) * _latin_hypercube(rng, samples, len(names)))
    y = np.zeros((samples, len(outputs)))
    start = time.perf_counter()
    for i in range(samples):
        point = conditions
        for field, value in zip(names, x[i], strict=True):
            point = _apply(point, field, float(value))
        y[i] = quantities(model.solve_steady_state(point, mode), outputs)
    solve_time = (time.perf_counter() - start) / samples
    return xr.Dataset(
        {
            "inputs": (("sample", "parameter"), x),
            "outputs": (("sample", "output"), y),
            "split": ("sample", split_indices(samples, fractions, seed + 1)),
        },
        coords={
            "parameter": names,
            "output": list(outputs),
            "low": ("parameter", np.exp(low)),
            "high": ("parameter", np.exp(high)),
        },
        attrs={
            "model": model.name,
            "manifest_sha256": manifest_digest(model),
            "opencr_version": _opencr_version(),
            "sampling": "log-uniform Latin hypercube",
            "seed": seed,
            "solve": mode,
            "conditions": json.dumps(conditions.to_dict(), default=str),
            "fractions": list(fractions),
            "solve_time": solve_time,
        },
    )
