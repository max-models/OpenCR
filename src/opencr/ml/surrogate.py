"""A baseline surrogate of the CR solver with scikit-learn.

:class:`Surrogate` learns ``ln(outputs)`` from ``ln(inputs)`` of a dataset
(:func:`~opencr.ml.dataset.generate_dataset`), by default with a Gaussian
process (a neural network is the optional next step). The solver stays the reference: :meth:`Surrogate.evaluate`
reports the held-out error against it and the speed-up, and
:meth:`Surrogate.predict` flags inputs outside the training ranges (out of
domain), where the surrogate is not to be trusted. scikit-learn is imported
only here (the ``ml`` extra).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

__all__ = ["Prediction", "Surrogate"]


def _default_estimator(seed: int, dims: int) -> Any:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    kernel = ConstantKernel() * RBF(length_scale=[1.0] * dims) + WhiteKernel(
        1e-6, (1e-12, 1e-2)
    )
    return make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed
        ),
    )


@dataclass(frozen=True)
class Prediction:
    """Surrogate outputs and whether each input is inside the training domain.

    Attributes:
        values: The predicted outputs (sample, output).
        in_domain: True where every input is inside the training ranges.
    """

    values: np.ndarray
    in_domain: np.ndarray


class Surrogate:
    """A regression of the dataset's outputs on its inputs, in log space.

    Args:
        estimator: A scikit-learn regressor with multi-output ``fit`` and
            ``predict``; by default a Gaussian process with an
            anisotropic RBF kernel on standard-scaled inputs (smooth, few samples).
        seed: The seed of the default estimator.
    """

    def __init__(self, estimator: Any = None, seed: int = 0) -> None:
        """Store the estimator (built on first use)."""
        self.estimator = estimator
        self.seed = seed
        self.parameters: tuple[str, ...] = ()
        self.outputs: tuple[str, ...] = ()
        self.low = np.empty(0)
        self.high = np.empty(0)
        self.provenance: dict[str, Any] = {}

    def fit(self, dataset: xr.Dataset) -> Surrogate:
        """Train on the dataset's training split and return ``self``."""
        if self.estimator is None:
            self.estimator = _default_estimator(self.seed, dataset.sizes["parameter"])
        train = dataset.isel(sample=dataset["split"].values == 0)
        x = np.log(train["inputs"].values)
        y = np.log(train["outputs"].values)
        self.estimator.fit(x, y if y.shape[1] > 1 else y[:, 0])
        self.parameters = tuple(dataset["parameter"].values.tolist())
        self.outputs = tuple(dataset["output"].values.tolist())
        self.low = np.asarray(dataset["low"].values, dtype=float)
        self.high = np.asarray(dataset["high"].values, dtype=float)
        self.provenance = dict(dataset.attrs)
        return self

    def predict(self, inputs: np.ndarray) -> Prediction:
        """Return the outputs at ``inputs`` (sample, parameter) with the domain flag."""
        if not self.parameters:
            raise RuntimeError("fit the surrogate first")
        x = np.atleast_2d(np.asarray(inputs, dtype=float))
        in_domain = np.all((x >= self.low) & (x <= self.high), axis=1)
        y = np.asarray(self.estimator.predict(np.log(x)))
        y = y.reshape(len(x), len(self.outputs))
        return Prediction(np.exp(y), in_domain)

    def forward(self, theta: np.ndarray) -> np.ndarray:
        """One sample's outputs, as a forward model for :mod:`opencr.inference`."""
        return self.predict(np.asarray(theta)[None, :]).values[0]

    def evaluate(
        self, dataset: xr.Dataset, split: int = 2, solve_time: float | None = None
    ) -> xr.Dataset:
        """Return the held-out relative errors and the speed-up on one split.

        Args:
            dataset: The dataset the surrogate was trained on (or a new one
                from the same model).
            split: 1 for validation, 2 for test.
            solve_time: The solver's time per sample in s, for the speed-up;
                by default the dataset's ``solve_time``.

        Returns:
            Per output: the median, 95th percentile and maximum of
            ``|predicted / solved - 1|``, and in the attributes the surrogate's
            time per sample and the speed-up.
        """
        if solve_time is None:
            solve_time = dataset.attrs.get("solve_time")
        held = dataset.isel(sample=dataset["split"].values == split)
        start = time.perf_counter()
        predicted = self.predict(held["inputs"].values).values
        per_sample = (time.perf_counter() - start) / max(held.sizes["sample"], 1)
        error = np.abs(predicted / held["outputs"].values - 1.0)
        attrs: dict[str, Any] = {"split": split, "surrogate_time": per_sample}
        if solve_time is not None:
            attrs["solve_time"] = solve_time
            attrs["speed_up"] = solve_time / per_sample
        return xr.Dataset(
            {
                "median_error": ("output", np.median(error, axis=0)),
                "p95_error": ("output", np.percentile(error, 95, axis=0)),
                "max_error": ("output", error.max(axis=0)),
            },
            coords={"output": list(self.outputs)},
            attrs=attrs,
        )
