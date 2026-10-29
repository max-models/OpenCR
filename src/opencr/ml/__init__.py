"""Optional machine-learning surrogates of the CR solver. Imports scikit-learn only when used.

* :func:`generate_dataset`: solutions over sampled conditions, with
  provenance and a reproducible train/validation/test split;
* :class:`Surrogate`: a scikit-learn baseline with held-out errors, speed-up
  and an out-of-domain flag. The solver stays the reference.
"""

from opencr.ml.dataset import SPLITS, generate_dataset, manifest_digest, split_indices
from opencr.ml.surrogate import Prediction, Surrogate

__all__ = [
    "SPLITS",
    "Prediction",
    "Surrogate",
    "generate_dataset",
    "manifest_digest",
    "split_indices",
]
