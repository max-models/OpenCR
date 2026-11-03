"""Physical data with provenance.

References and provenance records (`provenance`), the checks
every importer runs (`checks`), OpenCR's native versioned
format (`native`), importers of published formats
(`importers`) and the bundled datasets (atomic hydrogen in
`hydrogen`).
"""

import importlib
from typing import Any

#: The public names and their modules; imported on first use, because the core
#: layer imports `opencr.data.provenance` and the checks import the core.
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
