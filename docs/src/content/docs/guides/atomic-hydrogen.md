---
title: Atomic hydrogen
description: The hydrogen/atomic/basic model — states, processes, data, options and limits.
---

`opencr.load_model("hydrogen/atomic/basic", n_max=20, isotope="H")` builds atomic hydrogen
resolved in the principal quantum number $n$.

## States

H($n$) for $n = 1 \ldots n_\mathrm{max}$ with energies $E_n = I_H (1 - 1/n^2)$,
$I_H = 13.598$ eV, and weights $g_n = 2n^2$; the proton H⁺ at $I_H$ with $g = 1$. The drivers
of the quasi-static solve are H($n$=1) and H⁺. `isotope="D"` names the species D and D⁺ with
the same data (the reduced-mass shift of 0.03% is neglected).

## Processes

| Reaction names | Process | Data |
| --- | --- | --- |
| `exc:n-m` | e + H($n$) → e + H($m$) | Janev, Reiter and Samm (2003) eqs. (4)–(5) for 1→2…5 and 2→3; Johnson (1972) otherwise |
| `dex:m-n` | e + H($m$) → e + H($n$) | detailed balance (Klein–Rosseland) |
| `ion:n` | e + H($n$) → 2e + H⁺ | Janev et al. eq. (14) for $n \le 3$, Johnson eq. (15) above |
| `rec3:n` | 2e + H⁺ → e + H($n$) | detailed balance (Saha) |
| `rr:n` | e + H⁺ → H($n$) + hν | Janev et al. eqs. (21) and (22) |
| `A:m-n` | H($m$) → H($n$) + hν | Johnson's oscillator strengths, within 1% of the exact values |

The fits are implemented in plasmacoll (`plasmacoll.atomic_hydrogen`), which the Monte Carlo
collisions use too. `deexcitation=False`, `three_body_recombination=False` and
`radiative_recombination_=False` switch the processes off, for example to model an ionizing
plasma.

## Limits

- $n_\mathrm{max} \le 60$. Johnson's excitation formula turns negative for $n \to n+1$ above
  $n = 24$, so the excitations involving a level above 24 use the Maxwellian rate coefficients of
  Vriens and Smeets (1980) (`plasmacoll.atomic_hydrogen.vriens_smeets_excitation_rate`), within a
  factor 2 of Johnson's below 24; with a non-Maxwellian EEDF they are evaluated at its effective
  temperature. Effective recombination at low density depends on $n_\mathrm{max}$ (the levels
  above it are missing, not extrapolated).
- The $l$ sub-states are assumed statistically populated. This needs fast $l$-mixing, which
  at low density fails for $n = 2$ (the metastable 2s).
- Heavy-particle collisions with excited atoms (proton impact, which matters for high $n$ at
  high density) are not included. Molecular hydrogen is in [Molecular hydrogen](/OpenCR/guides/molecular-hydrogen/).
- The cross sections are uncertain by 10–20% from the ground state and 20–50% near threshold
  between excited states (the report's estimates); the effective coefficients inherit that.

`model.manifest()` lists every state, reaction, source and assumption.
