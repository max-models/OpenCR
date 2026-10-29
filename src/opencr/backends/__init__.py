"""The array layer: cunumpy, with explicit execution boundaries.

OpenCR's reference path is NumPy and SciPy (sparse matrices, sparse LU, stiff
integrators), one set of conditions at a time. The batched path
(:mod:`opencr.backends.batched`) evaluates the Maxwellian rate coefficients
and does the quasi-static reduction of many conditions at once as dense
arrays on cunumpy's ``xp``, which is NumPy or CuPy for the whole process
(``cunumpy.set_backend("cupy")`` moves it to the GPU when one is available).
The code is written once against ``xp``; nothing in OpenCR chooses between
the two.

The execution boundaries are explicit: inputs cross to ``xp`` with
``xp.asarray`` when a batch is assembled, outputs come back as NumPy arrays
with ``xp.to_numpy``, and a :class:`~opencr.core.rates.Function` rate always
runs on the host. :func:`device` says where the arrays live.
"""

from __future__ import annotations

import cunumpy as xp

__all__ = ["device"]


def device() -> str:
    """Return where cunumpy's arrays live: ``"gpu"`` on CuPy, else ``"cpu"``."""
    return "gpu" if xp.get_backend() == "cupy" else "cpu"
