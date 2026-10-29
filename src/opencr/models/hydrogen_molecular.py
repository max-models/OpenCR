"""Molecular hydrogen: H2 resolved in its vibrational levels, with H, H+ and H2+.

``hydrogen/molecular/basic`` (step 1 of the molecules): the states H2(X, v=0..v_max),
the excited electronic states B, C, EF, a and d, H(n=1), H+ and H2+, with

* electron-impact vibrational excitation ``v -> v+1, v+2`` and its inverse by
  detailed balance: the fits of ``0 -> 1`` and ``0 -> 2`` (Janev, Reiter and
  Samm 2003, eq. (74)) scaled to the other levels as for a harmonic
  oscillator, ``(v+1) sigma_01`` and ``(v+1)(v+2)/2 sigma_02``, shifted to
  the thresholds of the real ladder;
* electron-impact excitation of B, C, EF, a and d, dissociation through the
  triplets b, c and e, ionization and dissociative ionization (eqs. (80),
  (81), (95), (106), (107)) from every level, with the ``v = 0`` cross
  section shifted to the lower threshold of the excited level;
* the radiative decays B, C -> X(v=0), EF -> B, a -> b -> H + H (the
  continuum) and d -> a (the Fulcher band).

``hydrogen/molecular/ions`` adds the negative and molecular ions (step 2):
dissociative attachment ``e + H2(v) -> H- + H`` per level (eq. (124)), the
destruction of H- by electrons (eq. (58)), by H+ (eq. (63)), by H2+ and H3+
and by atoms (eq. (71)), the conversion ``H2+ + H2 -> H3+ + H`` and the
dissociative recombination of H2+ and H3+. Optional wall losses (``wall_loss``,
s^-1 by state) stand in for transport.

The drivers of the quasi-static solve are H2(X, v=0), H(n=1), H+ and H2+;
with the ions, H3+ + H- makes the system nonlinear, which the steady solver
handles (but not :meth:`~opencr.model.Model.effective_rates`). The model's
assumptions list what is simplified; its coverage is in the guide
"Molecular hydrogen".
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from plasmacoll import CrossSection
from plasmacoll.janev import (
    JANEV_FITS,
    HydrogenFit,
    dissociative_attachment_fit,
)

from opencr.core.network import ReactionNetwork
from opencr.core.rates import (
    Arrhenius,
    Constant,
    CrossSectionRate,
    DetailedBalance,
)
from opencr.core.reactions import Reaction
from opencr.core.species import ELECTRON_STATE, State
from opencr.data.hydrogen import JANEV_2003, hydrogen_levels
from opencr.data.molecular_hydrogen import (
    ELECTRONIC_STATES,
    LIFETIME_PROVENANCE,
    NUM_VIBRATIONAL_LEVELS,
    RADIATIVE_DECAYS,
    ion_states,
    molecular_states,
)
from opencr.data.provenance import Provenance, Reference

__all__ = [
    "ASSUMPTIONS",
    "ION_ASSUMPTIONS",
    "ION_CHEMISTRY_PROVENANCE",
    "molecular_hydrogen_network",
]

#: The assumptions of ``hydrogen/molecular/basic``.
ASSUMPTIONS = (
    "H2(X) resolved in v = 0..14 with rotations summed; the electronic states B, C, EF, "
    "a and d unresolved in v; the other singlets (B', B'', D, D', HH, GK, I) omitted "
    "(they decay back to X).",
    "Vibrational excitation of v > 0 scaled from 0 -> 1 and 0 -> 2 as for a harmonic "
    "oscillator; no V-V or V-T relaxation, no wall de-excitation or recombination.",
    "Electronic excitation, ionization and dissociation from v > 0 use the v = 0 cross "
    "section shifted to the lower threshold (the Franck-Condon dependence on v is "
    "neglected).",
    "B and C decay to X(v=0): the vibrational excitation through the radiative cascade "
    "(EV process) is not resolved, which underestimates the high v.",
    "Excitation of the triplets b, c and e dissociates (c and e cascade to b); a "
    "decays to the b continuum and dissociates.",
    "Atoms only in H(n=1): dissociative excitation to H(n >= 2) is omitted; merge with "
    "hydrogen/atomic/basic for the atomic levels.",
    "Optically thin; electron collisions only for H2 (no heavy-particle excitation).",
)
#: The further assumptions of ``hydrogen/molecular/ions``.
ION_ASSUMPTIONS = (
    "H-, H2+ and H3+ unresolved; mutual neutralization goes to ground-state atoms "
    "and molecules.",
    "Dissociative recombination of H2+ and H3+, and mutual neutralization of H- with "
    "H2+ and H3+, from representative literature rates (factor 2).",
    "H- + H associative detachment forms H2(X, v=0) (it forms hot v in reality).",
)

_JANEV = Provenance(
    Reference(JANEV_2003.citation, url=JANEV_2003.url),
    units="m^2 against eV",
    conversions=(
        "1e-16 cm^2 -> 1e-20 m^2",
        "tabulated by OpenCR from plasmacoll's fit",
    ),
    uncertainty=0.3,
    uncertainty_note="the report's fits; larger for the scaled and shifted levels",
    licence="published-fit",
)
#: The provenance of the ion-chemistry rates not from Janev, Reiter and Samm.
ION_CHEMISTRY_PROVENANCE = {
    "conversion": Provenance(
        Reference(
            "J. K. Theard and W. T. Huntress, Ion-molecule reactions and vibrational "
            "deactivation of H2+ ions in mixtures of hydrogen and helium, J. Chem. Phys. "
            "60, 2840 (1974)",
            doi="10.1063/1.1681453",
        ),
        units="m^3/s",
        conversions=("2.1e-9 cm^3/s -> 2.1e-15 m^3/s",),
        uncertainty=0.2,
        licence="published-fit",
    ),
    "h3_recombination": Provenance(
        Reference(
            "B. J. McCall et al., An enhanced cosmic-ray flux towards zeta Persei "
            "inferred from a laboratory study of the H3+ - e- recombination rate, "
            "Nature 422, 500 (2003)",
            doi="10.1038/nature01498",
        ),
        units="m^3/s against eV",
        conversions=("2.6e-7 (T/300 K)^-0.48 cm^3/s with T in eV",),
        uncertainty=0.3,
        notes="branching 3H : H2 + H = 0.75 : 0.25",
        licence="published-fit",
    ),
    "h2_recombination": Provenance(
        Reference(
            JANEV_2003.citation,
            url=JANEV_2003.url,
            detail="magnitude of the dissociative recombination of H2+ (not a fit)",
        ),
        units="m^3/s against eV",
        uncertainty=1.0,
        uncertainty_note="factor 2: depends on the vibrational state of H2+",
        licence="published-fit",
    ),
    "neutralization": Provenance(
        Reference(
            "E. G. Thorsteinsson and J. T. Gudmundsson, A global (volume averaged) model "
            "of a chlorine discharge, and J. T. Gudmundsson's hydrogen global models; "
            "the value of H+ + H- at room temperature",
            detail="2e-7 (T/300 K)^-0.5 cm^3/s",
        ),
        units="m^3/s against eV",
        uncertainty=1.0,
        uncertainty_note="factor 2, assumed equal for H2+ and H3+",
        licence="published-fit",
    ),
}

#: kT at 300 K in eV.
_ROOM = 0.025852


def _table(
    fit: HydrogenFit,
    threshold: float,
    scale: float = 1.0,
    max_energy: float = 1.0e4,
    num: int = 161,
) -> CrossSection:
    """Tabulate ``scale * fit(E - threshold + fit.threshold)`` from ``threshold``."""
    offset = np.geomspace(1.0e-6 * max(threshold, 1.0), max_energy - threshold, num)
    energy = np.concatenate([[threshold], threshold + offset])
    sigma = scale * fit(energy - threshold + fit.threshold)
    return CrossSection(
        energy=energy, sigma=sigma, threshold=threshold, label=fit.label
    )


def _detail(fit: HydrogenFit, note: str = "") -> Provenance:
    reference = Reference(
        _JANEV.reference.citation,
        url=_JANEV.reference.url,
        detail=fit.source + (f"; {note}" if note else ""),
    )
    return Provenance(
        reference,
        units=_JANEV.units,
        conversions=_JANEV.conversions,
        uncertainty=_JANEV.uncertainty,
        uncertainty_note=_JANEV.uncertainty_note,
        licence=_JANEV.licence,
    )


def _reduced(a: State, b: State) -> float:
    return a.species.mass * b.species.mass / (a.species.mass + b.species.mass)


def _molecular_reactions(
    x: list[State],
    excited: dict[str, State],
    atom: State,
    ions: dict[str, State],
    max_energy: float,
) -> list[Reaction]:
    e = ELECTRON_STATE
    reactions: list[Reaction] = []
    shift = [s.energy - x[0].energy for s in x]
    for step, key in ((1, "e_h2_vibrational_1"), (2, "e_h2_vibrational_2")):
        fit = JANEV_FITS[key]
        for v in range(len(x) - step):
            scale = math.comb(v + step, step)
            name = f"eV:{v}-{v + step}"
            reactions.append(
                Reaction(
                    name,
                    (e, x[v]),
                    (e, x[v + step]),
                    "excitation",
                    CrossSectionRate(
                        cross_section=_table(
                            fit, x[v + step].energy - x[v].energy, scale, max_energy
                        )
                    ),
                    _detail(fit, "harmonic-oscillator scaling" if v else ""),
                )
            )
            reactions.append(
                Reaction(
                    f"eV:{v + step}-{v}",
                    (e, x[v + step]),
                    (e, x[v]),
                    "deexcitation",
                    DetailedBalance(forward=name),
                )
            )
    triplets = ("e_h2_triplet_b", "e_h2_triplet_c", "e_h2_triplet_e")
    for v, level in enumerate(x):
        note = f"from v={v}, threshold shifted" if v else ""
        for label, state in excited.items():
            fit = JANEV_FITS[ELECTRONIC_STATES[label][0]]
            reactions.append(
                Reaction(
                    f"exc:{v}-{label}",
                    (e, level),
                    (e, state),
                    "excitation",
                    CrossSectionRate(
                        cross_section=_table(
                            fit, fit.threshold - shift[v], 1.0, max_energy
                        )
                    ),
                    _detail(fit, note),
                )
            )
        for key in triplets:
            fit = JANEV_FITS[key]
            reactions.append(
                Reaction(
                    f"diss:{v}-{key[-1]}",
                    (e, level),
                    (e, atom, atom),
                    "dissociation",
                    CrossSectionRate(
                        cross_section=_table(
                            fit, fit.threshold - shift[v], 1.0, max_energy
                        )
                    ),
                    _detail(fit, note),
                )
            )
        fit = JANEV_FITS["e_h2_ionization"]
        reactions.append(
            Reaction(
                f"ion:{v}",
                (e, level),
                (e, e, ions["H2+"]),
                "ionization",
                CrossSectionRate(
                    cross_section=_table(fit, fit.threshold - shift[v], 1.0, max_energy)
                ),
                _detail(fit, note),
            )
        )
        fit = JANEV_FITS["e_h2_dissociative_ionization"]
        reactions.append(
            Reaction(
                f"dion:{v}",
                (e, level),
                (e, e, ions["H+"], atom),
                "dissociative_ionization",
                CrossSectionRate(
                    cross_section=_table(fit, fit.threshold - shift[v], 1.0, max_energy)
                ),
                _detail(fit, note),
            )
        )
    for upper, lower, lifetime in RADIATIVE_DECAYS:
        if upper not in excited:
            continue
        products = (atom, atom) if lower is None else (
            (x[0],) if lower == "X" else (excited[lower],)
        )  # fmt: skip
        reactions.append(
            Reaction(
                f"A:{upper}-{lower or 'b'}",
                (excited[upper],),
                products,
                "spontaneous_emission",
                Constant(value=1.0 / lifetime),
                LIFETIME_PROVENANCE,
            )
        )
    return reactions


def _ion_reactions(
    x: list[State], atom: State, ions: dict[str, State], max_energy: float
) -> list[Reaction]:
    e = ELECTRON_STATE
    h_minus, h_plus, h2_plus, h3_plus = (ions[k] for k in ("H-", "H+", "H2+", "H3+"))
    reactions: list[Reaction] = []
    for v, level in enumerate(x):
        fit = dissociative_attachment_fit(v)
        reactions.append(
            Reaction(
                f"da:{v}",
                (e, level),
                (h_minus, atom),
                "dissociative_attachment",
                CrossSectionRate(
                    cross_section=_table(fit, fit.threshold, 1.0, min(max_energy, 50.0))
                ),
                _detail(fit),
            )
        )
        reactions.append(
            Reaction(
                f"conv:{v}",
                (h2_plus, level),
                (h3_plus, atom),
                "proton_transfer",
                Constant(value=2.1e-15),
                ION_CHEMISTRY_PROVENANCE["conversion"],
            )
        )
    fit = JANEV_FITS["e_hminus_detachment"]
    reactions.append(
        Reaction(
            "det:e",
            (e, h_minus),
            (e, e, atom),
            "detachment",
            CrossSectionRate(cross_section=fit.cross_section(max_energy)),
            _detail(fit),
        )
    )
    for key, products, kind in (
        ("mutual_neutralization", (atom, atom), "mutual_neutralization"),
        ("hminus_h_associative_detachment", (x[0], e), "associative_detachment"),
        ("hminus_h_detachment", (atom, atom, e), "detachment"),
    ):
        fit = JANEV_FITS[key]
        partner = h_plus if kind == "mutual_neutralization" else atom
        reactions.append(
            Reaction(
                f"{key}",
                (h_minus, partner),
                products,
                kind,
                CrossSectionRate(
                    cross_section=fit.cross_section(max_energy),
                    depends_on="ion" if partner is h_plus else "gas",
                    mass=_reduced(h_minus, partner),
                ),
                _detail(fit, "products in the ground state"),
            )
        )
    neutralization = Arrhenius(depends_on="ion", a=2.0e-13 * _ROOM**0.5, b=-0.5, c=0.0)
    reactions += [
        Reaction(
            "mn:H2+",
            (h_minus, h2_plus),
            (x[0], atom),
            "mutual_neutralization",
            neutralization,
            ION_CHEMISTRY_PROVENANCE["neutralization"],
        ),
        Reaction(
            "mn:H3+",
            (h_minus, h3_plus),
            (x[0], x[0]),
            "mutual_neutralization",
            neutralization,
            ION_CHEMISTRY_PROVENANCE["neutralization"],
        ),
        Reaction(
            "dr:H2+",
            (e, h2_plus),
            (atom, atom),
            "dissociative_recombination",
            Arrhenius(a=1.0e-14, b=-0.4, c=0.0),
            ION_CHEMISTRY_PROVENANCE["h2_recombination"],
        ),
        Reaction(
            "dr:H3+:3H",
            (e, h3_plus),
            (atom, atom, atom),
            "dissociative_recombination",
            Arrhenius(a=0.75 * 2.6e-13 * _ROOM**0.48, b=-0.48, c=0.0),
            ION_CHEMISTRY_PROVENANCE["h3_recombination"],
        ),
        Reaction(
            "dr:H3+:H2",
            (e, h3_plus),
            (x[0], atom),
            "dissociative_recombination",
            Arrhenius(a=0.25 * 2.6e-13 * _ROOM**0.48, b=-0.48, c=0.0),
            ION_CHEMISTRY_PROVENANCE["h3_recombination"],
        ),
    ]
    return reactions


def molecular_hydrogen_network(
    v_max: int = NUM_VIBRATIONAL_LEVELS - 1,
    electronic: bool = True,
    ions: bool = False,
    wall_loss: Mapping[str, float] | None = None,
    max_energy: float = 1.0e4,
) -> ReactionNetwork:
    """Return the network of ``hydrogen/molecular/basic`` (or ``.../ions`` with ``ions``).

    Args:
        v_max: The highest vibrational level of H2(X), at most 14.
        electronic: Resolve the excited electronic states B, C, EF, a and d.
        ions: Add H- and H3+ and their chemistry.
        wall_loss: Loss frequencies in s^-1 by state name (e.g. ``{"H-": 1e4}``),
            each a ``wall_loss`` reaction that removes the state.
        max_energy: The last tabulated cross-section energy in eV.
    """
    x = molecular_states(v_max, electronic=False)
    excited = (
        {s.label: s for s in molecular_states(0, electronic=True)[1:]}
        if electronic
        else {}
    )
    (atom,), _ = hydrogen_levels(1)
    charged = ion_states()
    states: list[State] = [*x, *excited.values(), atom, charged["H+"], charged["H2+"]]
    reactions = _molecular_reactions(x, excited, atom, charged, max_energy)
    if ions:
        states += [charged["H3+"], charged["H-"]]
        reactions += _ion_reactions(x, atom, charged, max_energy)
    names = {s.name: s for s in states}
    for name, frequency in (wall_loss or {}).items():
        if name not in names:
            raise KeyError(f"wall_loss: unknown state {name!r}")
        reactions.append(
            Reaction(
                f"wall:{name}",
                (names[name],),
                (),
                "wall_loss",
                Constant(value=frequency),
                Provenance(
                    Reference("given by the caller (wall_loss)"),
                    units="s^-1",
                    licence="public-domain",
                ),
            )
        )
    return ReactionNetwork(
        states,
        reactions,
        name="hydrogen/molecular/ions" if ions else "hydrogen/molecular/basic",
        assumptions=ASSUMPTIONS + (ION_ASSUMPTIONS if ions else ()),
    )
