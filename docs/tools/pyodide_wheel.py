"""Build the opencr wheel the live demo installs in the browser (Pyodide).

The wheel goes to docs/public/pyodide/ with a wheel.json naming it, so the page installs the
code of the commit it was built from. Its dependencies come from PyPI through micropip.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "public" / "pyodide"


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(OUT), str(ROOT)], check=True
    )
    (wheel,) = OUT.glob("opencr-*-py3-none-any.whl")
    (OUT / "wheel.json").write_text(json.dumps({"wheel": wheel.name}) + "\n")
    print(f"wrote {wheel}")


if __name__ == "__main__":
    main()
