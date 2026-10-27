"""OpenCR: collisional-radiative models of low-temperature plasmas.

OpenCR builds collisional-radiative (CR) models from explicit states and
reactions with traceable data, and solves their population equations: steady
states, quasi-static excited states with effective rate coefficients, and time
evolution. The physics is defined independently of the numerical backend; the
reference implementation uses NumPy and SciPy, cunumpy and JAX are optional.
Cross sections and Maxwellian rates come from :mod:`plasmacoll`.
"""

from importlib.metadata import PackageNotFoundError, version

import scope_profiler as sp

from opencr._profiling import PROFILE_TAGS

try:
    __version__ = version("opencr")
except PackageNotFoundError:  # not installed, e.g. run from a source tree
    __version__ = "0.0.0+unknown"

from opencr.analysis.results import Result
from opencr.core.conditions import Conditions
from opencr.model import Model
from opencr.models import available_models, load_model

__all__ = ["Conditions", "Model", "Result", "available_models", "load_model"]

# The decorated functions pick the tag up, so registering after the imports is in time.
sp.register_opt_in_tag(*PROFILE_TAGS)
