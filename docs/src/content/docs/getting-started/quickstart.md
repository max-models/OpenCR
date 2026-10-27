---
title: Quickstart
description: Solve atomic hydrogen, read populations, effective rates and lines.
---

## A model and its conditions

```python
import opencr

model = opencr.load_model("hydrogen/atomic/basic", n_max=20)
print(model)  # Model('hydrogen/atomic/basic', 21 states, 630 reactions)
```

A solve takes the conditions: the electron temperature in eV, the electron density in m⁻³ and
the densities of the *drivers*, the states that are inputs rather than unknowns (for atomic
hydrogen the ground state and the proton, see [The CR model](/opencr/guides/model/)):

```python
conditions = opencr.Conditions(
    electron_temperature=3.0,
    electron_density=1e19,
    densities={"H(n=1)": 1e19, "H+": 1e18},
)
result = model.solve_steady_state(conditions)
print(result.report.summary())
result.populations["H(n=3)"]
```

## Effective coefficients and lines

```python
reduction = result.reduction
reduction.effective_rate("H(n=1)", "H+")  # S_CR, m^3/s
reduction.effective_rate("H+", "H(n=1)")  # alpha_CR, m^3/s

lines = result.emission()  # an xarray Dataset over the lines
lines.sel(line="A:3-2")  # H-alpha: wavelength, emissivity, power
result.photon_emissivity_coefficients().sel(
    line="A:3-2"
)  # excitation and recombination PECs
spectrum = result.emission_spectrum(fwhm=0.05)  # W m^-3 nm^-1 on 350-700 nm
```

## Ionization balance

Without drivers, a closed steady state needs the total density of each element:

```python
balance = model.solve_steady_state(
    opencr.Conditions(3.0, None, totals={"H": 1e19}),  # n_e from quasi-neutrality
    mode="full",
)
balance.populations["H+"], balance.electron_density
```

## Scans and export

```python
import numpy as np

data = opencr.scan(
    model,
    conditions,
    electron_temperature=np.geomspace(1, 100, 20),
    electron_density=[1e17, 1e19, 1e21],
)
data["effective_rate"].sel(target="H+", source="H(n=1)")  # S_CR on the grid
result.export("result.nc")  # NetCDF with the manifest and the conditions
```

## From the shell

```bash
opencr list
opencr manifest hydrogen/atomic/basic --n-max 5
opencr effective hydrogen/atomic/basic --te 10 --ne 1e19
```

## Profiling

OpenCR's coarse operations (`opencr.rate_coefficients`, `opencr.solve_steady`, `opencr.cli`)
are [scope-profiler](https://github.com/max-models/scope-profiler) regions with the opt-in tag
`opencr`, off unless enabled: `scope_profiler.setup(enable_tags=["opencr"])` or
`SCOPE_PROFILER_ENABLE_TAGS=opencr`.
