"""Physical data with provenance.

References and provenance records (:mod:`~opencr.data.provenance`), the checks
every importer runs (:mod:`~opencr.data.checks`), OpenCR's native versioned
format (:mod:`~opencr.data.native`), importers of published formats
(:mod:`~opencr.data.importers`) and the bundled datasets (atomic hydrogen in
:mod:`~opencr.data.hydrogen`).
"""

import importlib
from typing import Any

#: The public names and their modules; imported on first use, because the core
#: layer imports :mod:`opencr.data.provenance` and the checks import the core.
_EXPORTS = {
    "DataError": "opencr.data.checks",
    "check_network_data": "opencr.data.checks",
    "load_network": "opencr.data.native",
    "save_network": "opencr.data.native",
    "Provenance": "opencr.data.provenance",
    "Reference": "opencr.data.provenance",
    "Validity": "opencr.data.provenance",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Import the public names lazily."""
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'opencr.data' has no attribute {name!r}")
