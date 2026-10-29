---
title: Molecular hydrogen
description: The hydrogen/molecular models — H2 in its vibrational levels, its electronic states, and the H-, H2+ and H3+ chemistry.
---

`opencr.load_model("hydrogen/molecular/basic")` builds H₂ resolved in the vibrational levels of
its ground state, with its excited electronic states, H, H⁺ and H₂⁺;
`"hydrogen/molecular/ions"` adds H⁻ and H₃⁺ and their chemistry. The drivers of the
quasi-static solve are H₂(X,v=0), H($n$=1), H⁺ and H₂⁺.

## States

| States | Energies and weights |
| --- | --- |
| `H2(X,v=0)` … `H2(X,v=14)` | Fantz and Wünderlich (2006), rotations summed ($g = 1$) |
| `H2(B)`, `H2(C)`, `H2(EF)`, `H2(a)`, `H2(d)` | the vertical thresholds of the excitation fits, electronic weights |
| `H(n=1)`, `H+` | as in [Atomic hydrogen](/OpenCR/guides/atomic-hydrogen/) |
| `H2+`, `H3+`, `H-` | $I(\mathrm{H_2}) - D_0$; below H₂ + H⁺ by the proton affinity 4.38 eV; −0.754 eV |

All energies share one scale with atomic hydrogen: zero at two H(1s) atoms per H₂, so
H₂(X,v=0) is at $-D_0 = -4.478$ eV. Detailed balance and photon energies follow from it.

## Processes

| Reaction names | Process | Data |
| --- | --- | --- |
| `eV:v-w`, `eV:w-v` | e + H₂(v) ↔ e + H₂(v+1, v+2) | Janev, Reiter and Samm (2003) eq. (74) for 0→1, 0→2, scaled as a harmonic oscillator; inverse by detailed balance |
| `exc:v-B` … `exc:v-d` | e + H₂(v) → e + H₂(B, C, EF, a, d) | eqs. (80), (81), (95), shifted to the threshold of level v |
| `diss:v-b`, `-c`, `-e` | e + H₂(v) → e + 2H through b, c and e | eq. (95) |
| `ion:v`, `dion:v` | ionization and dissociative ionization | eqs. (106), (107) |
| `A:B-X`, `A:C-X`, `A:EF-B`, `A:a-b`, `A:d-a` | radiative decays (Lyman, Werner, a→b continuum, Fulcher) | lifetimes of Fantz and Wünderlich (2006) |
| `da:v` | e + H₂(v) → H⁻ + H | eq. (124), table 27, per level |
| `det:e`, `mutual_neutralization`, `hminus_h_*` | the destruction of H⁻ by e, H⁺ and H | eqs. (58), (63), (71) |
| `mn:H2+`, `mn:H3+` | H⁻ + H₂⁺, H₃⁺ | the H⁺ value at room temperature, factor 2 |
| `conv:v` | H₂⁺ + H₂ → H₃⁺ + H | Theard and Huntress (1974) |
| `dr:H2+`, `dr:H3+:*` | dissociative recombination | magnitude of Janev et al.; McCall et al. (2003) |
| `wall:<state>` | losses to the walls, from `wall_loss={"H-": 1e4}` (s⁻¹) | given |

The rate of dissociative attachment over a Boltzmann vibrational population agrees with
plasmacoll's `dissociative_attachment` mixture to 3% (a test), and the vibrational ladder
under e–V collisions alone relaxes to a Boltzmann distribution at $T_e$ (to 10⁻⁶).

## Limits

- The excitation of v > 0 is scaled from v = 0; there is no V–V or V–T relaxation, no wall
  de-excitation and no recombinative desorption, so the high v are an upper estimate where
  those matter.
- B and C decay to X(v=0): the vibrational excitation through the radiative cascade (the EV
  process, a main source of high v) is not resolved.
- Atoms only in H($n$=1): merge the network with `hydrogen/atomic/basic`
  (`ReactionNetwork.merge`) for the atomic levels; dissociative excitation to H($n \ge 2$) is
  not included.
- With the ions, H₃⁺ + H⁻ is quadratic in two quasi-static states: the steady solve handles
  it (Newton), `Model.effective_rates` does not.
- Without wall losses the ions are lost only in the volume; give `wall_loss` for a source.
