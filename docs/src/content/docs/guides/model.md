---
title: The CR model
description: The population equations, what a solve prescribes and what it solves for, the closures, normalisation and units.
---

This page fixes the physics every OpenCR model shares: the equations, which densities are
inputs and which are unknowns, how a steady state is closed, and the units. The API follows
from it.

## Population equations

A model is a set of states $i$ (levels, ions, molecules) and reactions $r$. Reaction $r$
has reactant multiplicities $a_{ri}$, net stoichiometry $\nu_{ir}$ (products minus
reactants) and a rate coefficient $k_r$, so its rate per volume is

$$
w_r = k_r\, n_e^{a_{re}} \prod_i n_i^{a_{ri}},
$$

where $a_{re}$ counts the electrons among the reactants. The populations obey

$$
\frac{d n_i}{dt} = R_i(\mathbf n) = \sum_r \nu_{ir} w_r(\mathbf n).
$$

$k_r$ depends on the plasma conditions $\boldsymbol\theta$: the electron temperature or
energy distribution, the ion and gas temperatures, the escape factors of the lines. The
electron is never a state: $n_e$ is either given or follows from quasi-neutrality,
$n_e = \sum_i z_i n_i$.

Reactions are elementary: each has its own data and provenance. Effective rates (such as
the collisional-radiative ionization coefficient) are derived from them by eliminating
states (below); OpenCR never mixes the two in one network.

## Inputs and unknowns

A solve splits the states into

- **prescribed** states, whose densities are inputs (`Conditions.densities`): the
  *drivers* of a quasi-static solve (for atomic hydrogen the ground state $H(1)$ and the
  proton) and fixed *backgrounds* (H₂ in an atomic model);
- **evolving** states, the unknowns.

$T_e$ (or the energy distribution), $n_e$ or quasi-neutrality, $T_i$, $T_\mathrm{gas}$ and the
escape factors are always inputs. OpenCR does not solve an energy equation, and it does not
compute the energy distribution; plasmacoll's Boltzmann solver can (`EEDF.from_boltzmann`).

## The solves

**Quasi-static (CR) solve** — `Model.solve_steady_state(conditions)`, the default.
The excited states relax much faster than the drivers change (the usual CR assumption:
$A$ and electron-collision times of ns against µs–ms for transport and ionization), so they
are in steady state for given driver densities. With $P$ the drivers and $Q$ the rest, and
all reactions linear in the evolving states,

$$
0 = M_{QQ}\, n_Q + M_{QP}\, n_P \;\Rightarrow\; n_Q = -M_{QQ}^{-1} M_{QP}\, n_P .
$$

Each excited population is a sum of contributions from the drivers (for hydrogen an
*excitation* part $\propto n_e n_{H(1)}$ and a *recombination* part $\propto n_e n_{H^+}$).
Eliminating $Q$ leaves the effective rate matrix of the drivers,

$$
M^\mathrm{eff} = M_{PP} - M_{PQ} M_{QQ}^{-1} M_{QP},
$$

whose off-diagonal entries divided by $n_e$ are the effective (collisional-radiative)
coefficients: ionization $S_\mathrm{CR}$ from $H(1)$ to $H^+$, recombination
$\alpha_\mathrm{CR}$ back. Line emission is split the same way into photon emissivity
coefficients per driver. This is the generalised CR picture of ADAS (Summers et al. 2006).

**Closed steady state** — `Model.solve_steady_state(conditions, mode="full")`. Every
state evolves. The equations are then singular: $\sum_i c_{ei} R_i = 0$ for each conserved
element $e$ ($c_{ei}$ atoms of $e$ in state $i$). One equation per conserved element is replaced
by its total, $\sum_i c_{ei} n_i = N_e$, given in `Conditions.totals`. Without a total for a
conserved element the system is underdetermined and the solve stops with an error that says
which total is missing. The result is the ionization balance at the given $n_e$ (or
self-consistently with quasi-neutrality, which makes the system nonlinear).

**Time-dependent** — `Model.solve_transient(conditions, initial, times)`. Every evolving
state follows its equation from initial densities; the prescribed states stay prescribed
(constant, or following `conditions(t)` when a function of time is given). The system is stiff
(rates from $10^{-3}$ to $10^{9}\,\mathrm{s^{-1}}$), so the integrators are implicit (BDF by
default, or Radau) with the analytic sparse Jacobian. The report gives the drift of every element
total the evolving states conserve; $n_e$ is given or quasi-neutral at every moment, as in the
steady solves. Any output time can be read like a steady state (`history.at(index)`): fluxes,
dominant processes and emission at that moment, and the residual that says how far from steady
state the populations are.

Nonlinear reactions (two evolving reactants, such as H⁺ + H⁻ mutual neutralization, or
$n_e$ from quasi-neutrality) are allowed in the closed solve; the
quasi-static solve needs the excited states to enter linearly and says so otherwise.

## Normalisation and output

Populations are absolute densities in m⁻³. A quasi-static result also gives the reduced
populations $n_q/(n_e n_d)$ per driver $d$ (the population coefficients, in m³), and a
closed steady state the fractions $n_i/N_e$.

## Boundary conditions and what is left out

A model is zero-dimensional. Transport enters only as explicit reactions: a loss
`wall_loss` with a given rate (for example $1/\tau$ with a confinement time), or a
`pumping` source. Nothing is added behind the user's back. Radiation is optically thin
unless escape factors are set (below).

The atomic hydrogen model resolves the principal quantum number only (the $l$ sub-states
statistically populated, which needs fast $l$-mixing), truncates at a highest level
$n_\mathrm{max}$ without continuum lowering, and omits heavy-particle collisions with the
excited states; its manifest lists every assumption.

## Radiation trapping

An optically thick line is re-absorbed before it leaves the plasma, so only a fraction $g$ of
its spontaneous emission escapes: the model uses $g A_{ul}$ in place of $A_{ul}$ (and the
emissivities follow). OpenCR applies escape factors in two explicit ways, and never otherwise:

- `Conditions.escape`: a factor per line, given by the user (for example from a radiation
  transport calculation);
- `Conditions.trapping = RadiationTrapping(length, model)`: the factors are computed from the
  line-centre optical depth of each Doppler-broadened line over the absorption length $L$,

  $$
  \tau_0 = \frac{\lambda_0^3}{8\pi^{3/2}}\,\frac{g_u}{g_l}\,\frac{A_{ul}\, n_l}{v_D}\, L,
  \qquad v_D = \sqrt{2kT_\mathrm{gas}/m},
  $$

  with the approximation `model`: `"mewe"` (default), $g = (2 - e^{-\tau_0/1000})/(1 + \tau_0)$
  (Mewe 1967), or Holstein's asymptotic forms for a slab or a cylinder
  (`"holstein-slab"`, `"holstein-cylinder"`). The factors depend on the lower populations,
  which depend on the factors, so the solve iterates to self-consistency; the result's
  `conditions.escape` holds the final factors and its report the iterations. Factors given
  in `Conditions.escape` override the computed ones line by line.

$L$ is the distance the photons travel to the wall (a slab's half-thickness, a cylinder's
radius), a parameter of this zero-dimensional approximation and not something OpenCR can
know. The formulas are in `opencr.kinetics.radiation`.

## Units

SI throughout, except temperatures (as $kT$) and energies, in eV. Densities in m⁻³, rate
coefficients in s⁻¹ (one reactant), m³/s (two) and m⁶/s (three), wavelengths in nm in vacuum
(`air_wavelength` converts above 200 nm).
