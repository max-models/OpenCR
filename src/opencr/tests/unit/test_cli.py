"""The opencr command."""

from __future__ import annotations

import json

from opencr.cli import main


def test_list(capsys) -> None:
    assert main(["list"]) == 0
    assert "hydrogen/atomic/basic" in capsys.readouterr().out


def test_manifest(capsys) -> None:
    assert main(["manifest", "hydrogen/atomic/basic", "--n-max", "3"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["counts"]["states"] == 4 and manifest["drivers"] == ["H(n=1)", "H+"]


def test_effective(capsys) -> None:
    assert (
        main(
            [
                "effective",
                "hydrogen/atomic/basic",
                "--te",
                "10",
                "--ne",
                "1e19",
                "--n-max",
                "8",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "H(n=1) -> H+" in out and "H+ -> H(n=1)" in out
