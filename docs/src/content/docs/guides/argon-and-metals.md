---
title: Argon and metals
description: The argon/metastable model for magnetron discharges, with a sputtered metal.
---

`opencr.load_model("argon/metastable", metal="Cu", wall_loss={...})` builds the argon of a
global model with lumped 4s levels, and optionally a sputtered metal (Cu or Ti). It is the
first rung of the magnetron programme's CR needs (the ionization region model).

## States and processes

Ar, Ar(m) (the metastables 4s[3/2]₂ and 4s′[1/2]₀, $g = 6$), Ar(r) (the resonant 4s[3/2]₁ and
4s′[1/2]₁, $g = 6$) and Ar⁺, with the reaction set of Ashida, Lee and Lieberman (1995) as in
Lieberman and Lichtenberg (2005): excitation, de-excitation, ionization from the ground state
and stepwise from the 4s levels, metastable–resonant mixing by electrons (Arrhenius fits over
1–7 eV), metastable pooling (`pooling=False` removes it) and the resonance lines of Ar(r) (the
NIST A of 104.8 and 106.7 nm averaged), which
[radiation trapping](/OpenCR/tutorials/06-radiation-trapping/) can trap.

With a metal: M, M⁺ and M²⁺, electron-impact ionization of M and M⁺ from the Lotz formula
(`plasmacoll.metal_ionization_cross_section`, the outer subshell: a lower estimate) and Penning
ionization Ar(m) + M → Ar + M⁺ + e at `penning_rate` (5e-16 m³/s by default, an order of
magnitude: give a measured value). The ions need losses: give `wall_loss` (s⁻¹ by state).

## Limits

- The 4s levels are lumped and the 4p levels are not resolved; the Arrhenius fits assume a
  Maxwellian over 1–7 eV.
- Not included: Ar²⁺, the hot and warm argon groups and the hot secondary electrons of the
  ionization region model, the metal metastables (²D₅/₂, ²D₃/₂) and charge exchange between Ar⁺
  and the metal. These are the next rungs (the metal levels need cross sections, e.g. Freund et
  al. for Cu).
