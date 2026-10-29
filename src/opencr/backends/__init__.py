"""Array backends behind explicit execution boundaries.

The reference implementation of OpenCR is NumPy and SciPy (sparse matrices,
stiff integrators), and every result is validated there. The accelerated
path evaluates the Maxwellian rate coefficients and assembles and solves the
quasi-static rate matrices of many conditions at once as dense batched
arrays, on an array namespace chosen here: NumPy (:data:`NUMPY`) or cunumpy
(:data:`CUNUMPY`, which runs on CuPy when a GPU is available and on NumPy
otherwise). The execution boundaries are explicit: the inputs are copied to
the backend in :meth:`Backend.asarray`, the outputs come back as NumPy
arrays through :meth:`Backend.to_numpy`, and a
:class:`~opencr.core.rates.Function` rate always runs on the host.

:mod:`opencr.backends.batched` holds the batched quasi-static reduction and
:func:`~opencr.backends.batched.batched_scan`, the batched counterpart of
:func:`opencr.scan`. The models never see a backend: species, states and
reactions are plain Python objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any

import cunumpy
import numpy as np

__all__ = ["BACKENDS", "CUNUMPY", "NUMPY", "Backend", "get_backend"]


@dataclass(frozen=True)
class Backend:
    """An array namespace and its execution boundaries.

    Attributes:
        name: ``"numpy"`` or ``"cunumpy"``.
        xp: The array namespace.
    """

    name: str
    xp: ModuleType

    @property
    def device(self) -> str:
        """Where the arrays live: ``"cpu"``, or ``"gpu"`` for cunumpy on CuPy."""
        if self.xp is cunumpy and cunumpy.get_backend() == "cupy":
            return "gpu"  # pragma: no cover - needs a GPU
        return "cpu"

    def asarray(self, array: Any) -> Any:
        """Copy ``array`` to the backend (the entry boundary)."""
        return self.xp.asarray(array, dtype=float)

    def to_numpy(self, array: Any) -> np.ndarray:
        """Return ``array`` as a NumPy array on the host (the exit boundary)."""
        if self.xp is np:
            return np.asarray(array, dtype=float)
        return np.asarray(cunumpy.to_numpy(array), dtype=float)

    def __repr__(self) -> str:
        """The name and the device."""
        return f"Backend({self.name!r}, {self.device})"


#: The reference backend.
NUMPY = Backend("numpy", np)
#: The accelerated backend: cunumpy, on CuPy when a GPU is available.
CUNUMPY = Backend("cunumpy", cunumpy)
#: The backends by name.
BACKENDS = {"numpy": NUMPY, "cunumpy": CUNUMPY}


def get_backend(backend: str | Backend) -> Backend:
    """Return the backend named ``backend`` (or ``backend`` itself)."""
    if isinstance(backend, Backend):
        return backend
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {sorted(BACKENDS)}")
    return BACKENDS[backend]
