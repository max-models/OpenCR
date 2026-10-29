"""Sensitivities, uncertainty propagation and inverse modelling on top of the solver.

* :func:`sensitivities` and :func:`propagate`
  (:mod:`opencr.inference.sensitivity`): how results depend on the rate data,
  and their uncertainty from the data's;
* :func:`posterior_grid` and :func:`fit` (:mod:`opencr.inference.inverse`):
  plasma parameters from measured quantities such as line ratios.
"""

from opencr.inference.inverse import (
    Fit,
    Observation,
    fit,
    forward_model,
    posterior_grid,
)
from opencr.inference.sensitivity import (
    Propagation,
    data_uncertainties,
    propagate,
    sensitivities,
)

__all__ = [
    "Fit",
    "Observation",
    "Propagation",
    "data_uncertainties",
    "fit",
    "forward_model",
    "posterior_grid",
    "propagate",
    "sensitivities",
]
