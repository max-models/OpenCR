# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Molecular hydrogen: `hydrogen/molecular/basic` (H2(X, v=0..14), B, C, EF, a, d, H, H+, H2+;
  vibrational and electronic excitation, dissociation, ionization, radiative decays) and
  `hydrogen/molecular/ions` (dissociative attachment per level, H-, H3+ and their chemistry,
  optional wall losses).
- `argon/metastable`: Ar with lumped metastable and resonant 4s levels and Ar+ (the global-model
  set of Ashida, Lee and Lieberman), optionally a sputtered Cu or Ti with Lotz ionization and
  Penning ionization.
- Atomic hydrogen up to n = 60, with the Vriens-Smeets rates above n = 24.
- Named quantities of a steady state (`opencr.analysis.quantities`), with the electron cooling
  coefficient per driver.
- `opencr.inference`: sensitivities to the rate data, linear and Monte Carlo uncertainty
  propagation from the provenance, and inverse modelling (grid posterior and least-squares fit)
  from measured quantities such as line ratios.
- `opencr.ml`: datasets with provenance and reproducible splits, a scikit-learn Gaussian-process
  surrogate with held-out errors, speed-up and an out-of-domain flag.
- `opencr.tables`: effective-rate tables on a T_e x n_e grid with log interpolation and netCDF,
  for wellbalanced, beamtracer and freeofcharge.
- `Model.solve_steady_state(..., factors=...)` and `Model.scaled_rates`: rate coefficients scaled
  by reaction, detailed-balance inverses with their forward reaction.

- The OpenCR package layout: `core`, `data`, `kinetics`, `solvers`, `backends`, `analysis`,
  `models`, `ml` and `inference`, with the optional extras `cunumpy`, `jax` and `ml`; depends on
  plasmacoll, SciPy and xarray. The template's greeting, sine-wave example and example
  tutorial are gone.
- Core model objects (`opencr.core`): `Species`, `State`, rate models (`CrossSectionRate`,
  `Arrhenius`, `Tabulated`, `Constant`, `Function`, `DetailedBalance`), `Reaction` with charge
  and element checks, `ReactionNetwork` with validation, subsets, merging and a manifest, and
  `Conditions`.
- Data layer (`opencr.data`): `Provenance` on every reaction, importer checks, a versioned
  native JSON format (`save_network`, `load_network`), LXCat, rate-table and ADAS adf11
  importers, and atomic hydrogen data (levels, Einstein coefficients, radiative recombination).
- Rate coefficients (`opencr.kinetics.rates`): exact Maxwellian averages of piecewise-linear
  cross sections, any EEDF (`Maxwellian`, `Generalized`, `TabulatedEEDF`), detailed balance
  (Klein-Rosseland, Saha) and escape factors.
- Population equations with a sparse analytic Jacobian (`PopulationSystem`), the quasi-static
  reduction to population and effective rate coefficients (`quasi_static`), and the steady-state
  solver (sparse LU, conservation closure, quasi-neutral root, continuation and Newton) with a
  `SolverReport`.
- `Model`, `load_model` and the `hydrogen/atomic/basic` model (H(n=1..24) and H+, electron
  collisions from plasmacoll's `atomic_hydrogen` fits, recombination and emission).
- Analysis: fluxes, dominant processes, lines and photon emissivity coefficients, synthetic
  spectra, `scan`, xarray and NetCDF export; the `opencr` command (`list`, `manifest`,
  `effective`).
- Verification against Saha-Boltzmann, coronal and detailed-balance limits, and an opt-in
  comparison with ADAS; guides, four tutorials and a full example.

### Fixed

- `setup_project.sh` also renames the docs smoke paths (`api/opencr/...` in `docs.yml`), so the
  docs build of a new project no longer looks for the template's API pages.

### Added

- Opt-in scope-profiler regions, as in plasmacoll, wellbalanced and patchwork: `_profiling.py`
  with the package's tag (`opencr`, renamed by `setup_project.sh`), registered on
  import, and an example region on `main`. Off unless the application enables the tag (e.g.
  `scope_profiler.setup(enable_tags=["opencr"])`). Adds the `scope-profiler>=0.7.5`
  dependency.
- `PyPI dependencies` workflow and `scripts/check_pypi_dependencies.py`: on every pull request to
  `main`, fail if a dependency is a direct reference (`name @ git+https://...`), is not on PyPI, or
  has a version range that no release satisfies, instead of failing the release after the merge.

### Changed

- The quasi-static reduction accepts reactions with a prescribed partner (e.g. Penning
  ionization of a prescribed metal, H2+ + H2(v)): they are linear in the quasi-static reactant.
- `__version__` is read from the installed package metadata (`importlib.metadata`), not hardcoded,
  so it can no longer disagree with `pyproject.toml` (a hardcoded one went stale in releases).

## [0.1.0] - 2026-10-05

### Added

- Package layout with a console entry point and pytest tests.
- Astro + Starlight documentation site with executed tutorials and a generated API reference.
- GitHub Actions for tests, ruff, pyright, tutorials, documentation and PyPI publishing.
