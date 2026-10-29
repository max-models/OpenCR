---
title: Sensitivities and inference
description: Sensitivities to the rate data, uncertainty propagation, inverse modelling, surrogates and effective-rate tables.
---

These tools work on named scalar quantities of a steady state
(`opencr.analysis.quantities`):

| Name | Quantity |
| --- | --- |
| `population:<state>` | a density (m⁻³) |
| `effective_rate:<source>><target>` | an effective rate coefficient (m³/s), quasi-static |
| `emissivity:<reaction>` | a line's photon emissivity (photons m⁻³ s⁻¹) |
| `electron_cooling:<driver>` | electron energy loss per electron and per driver particle (eV m³/s) |
| `ratio:<quantity>/<quantity>` | a ratio, e.g. `ratio:emissivity:A:3-2/emissivity:A:4-2` (Balmer α/β) |

## Sensitivities and uncertainties

`opencr.inference.sensitivities(model, conditions, outputs)` gives $\partial \ln q / \partial
\ln k_r$ by central differences; a detailed-balance inverse moves with its forward reaction
(`Model.solve_steady_state(..., factors={name: f})` scales rates the same way).
`propagate(..., method="linear" | "monte-carlo")` turns relative uncertainties of the data
(by default the `uncertainty` of each reaction's provenance) into the uncertainty of the
quantities: $\sigma^2(\ln q) = \sum_r S_{qr}^2 \sigma_r^2$ for independent data, or by sampling
log-normal factors; `Propagation.dominant` lists the reactions that matter most. This is the
uncertainty of the data, kept apart from the error of a surrogate.

## Inverse modelling

Measured quantities `Observation(name, value, uncertainty)` and parameters of the conditions
with log-uniform priors give the posterior
$p(\theta \mid y) \propto \exp(-\chi^2/2)$, $\chi^2 = \sum_i (\ln m_i(\theta) - \ln y_i)^2 /
s_i^2$. `posterior_grid` evaluates it on a grid (one to three parameters), `fit` finds the
maximum by least squares with a curvature estimate of the spread. The forward model is the CR
model (`forward_model(model, conditions, parameters, outputs)`) or a trained surrogate's
`forward`.

## Surrogates

`opencr.ml.generate_dataset` solves a model at a seeded Latin hypercube of log-uniform
conditions, with the provenance (the SHA-256 of the model's manifest, the OpenCR version, the
seed, the solver's time per sample) and a reproducible train/validation/test split.
`Surrogate().fit(dataset)` is a scikit-learn Gaussian process in log space (the `ml` extra);
`evaluate` reports held-out errors and the speed-up, `predict` flags inputs outside the training
ranges. The solver stays the reference.

## Effective-rate tables

`opencr.tables.effective_rate_table(model, conditions, outputs, temperatures, densities)`
tabulates quantities on a $T_e \times n_e$ grid; an `EffectiveRateTable` interpolates in
$\ln q$ against $\ln T_e$, $\ln n_e$ and writes netCDF. wellbalanced (`wellbalanced.cr`: the
hydrogen driver and the deuterium neutralizer), beamtracer (through wellbalanced) and
freeofcharge (`ElectronCRStopping`) take their rates from these tables.
