---
title: Installation
description: Install OpenCR and its optional extras.
---

OpenCR needs Python 3.10 or newer. With pip:

```bash
pip install opencr
```

or from a checkout, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/max-models/OpenCR
cd opencr
make install    # uv sync --extra dev, plus the pre-commit hooks
```

The core needs NumPy, SciPy, xarray and [plasmacoll](https://github.com/max-models/plasmacoll)
(the cross sections). Nothing else is imported unless you use it.

## Optional extras

| Extra     | Installs                                   | For                                         |
| --------- | ------------------------------------------ | ------------------------------------------- |
| `cunumpy` | cunumpy                                    | accelerated array backends (planned)        |
| `jax`     | JAX                                        | differentiable backends (planned)           |
| `ml`      | scikit-learn                               | surrogate models (planned)                  |
| `test`    | pytest and coverage                        | running the tests                           |
| `docs`    | the notebook runner, griffe, plasma-plots  | building these pages                        |
| `dev`     | linters and formatters, `test` and `docs`  | development                                 |

```bash
pip install "opencr[test]"
pytest --pyargs opencr
```

The comparison with ADAS runs only when `OPENCR_ADAS_DIR` points to a folder with your own
downloads of OPEN-ADAS `scd12_h.dat` and `acd12_h.dat` (their terms do not allow OpenCR to
ship them), see [Verification](/OpenCR/guides/verification/).
