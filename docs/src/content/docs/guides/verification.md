---
title: Verification
description: The acceptance criteria of the atomic hydrogen model and how they are checked.
---

The tests in `src/opencr/tests/reference/` and `src/opencr/tests/integration/` check the model
against limits that are known exactly and against published effective coefficients. Each
criterion below is a test.

## Exact limits

| Check | Criterion |
| --- | --- |
| Detailed balance: radiation off, any $T_e$ and $n_e$ | populations in Saha–Boltzmann equilibrium with H⁺ to $10^{-8}$ |
| Klein–Rosseland detailed balance with a tabulated EEDF | Boltzmann populations to 0.5% |
| High density ($n_e = 10^{26}$ m⁻³, 1 eV) | Saha–Boltzmann to $10^{-3}$ for every level |
| Low density (corona, $10^{10}$ m⁻³) | excitation from the ground state plus radiative cascades to $10^{-4}$ |
| Two-level atom, a three-state ionization chain | analytic populations and effective rates to $10^{-12}$ |
| Maxwellian rates | plasmacoll's numerical quadrature to 0.2% |
| Oscillator strengths, Einstein coefficients | NIST ASD (Wiese and Fuhr 2009) to 1% |
| Radiative recombination at $10^4$ K | 1s and case-A totals (Osterbrock and Ferland) to 5–6% |
| Closed steady states | element totals to $10^{-10}$; non-negative populations over $0.2$–$200$ eV and $10^{14}$–$10^{24}$ m⁻³ |

## Effective coefficients against ADAS

The collisional-radiative ionization ($S_\mathrm{CR}$) and recombination ($\alpha_\mathrm{CR}$)
coefficients of `hydrogen/atomic/basic` ($n_\mathrm{max} = 20$) agree with ADAS (Summers et
al. 1996, OPEN-ADAS `scd12_h`, `acd12_h`) within 25% for $T_e$ = 2–100 eV and
$n_e = 10^{17}$–$10^{21}$ m⁻³, and within about 10% at $n_e \ge 10^{19}$ m⁻³ and
$T_e$ = 2–30 eV. The largest differences, up to 35%, are in $S_\mathrm{CR}$ at 1 eV and low
density, where ionization is dominated by the threshold region of the ground-state cross
section, and in $\alpha_\mathrm{CR}$ at 1 eV and $10^{21}$ m⁻³ (20%). Both are within the
uncertainty of the cross sections.

OPEN-ADAS data may not be redistributed, so the test runs on your own download:

```bash
export OPENCR_ADAS_DIR=/path/to/folder  # with scd12_h.dat and acd12_h.dat from open.adas.ac.uk
pytest src/opencr/tests/reference
```
