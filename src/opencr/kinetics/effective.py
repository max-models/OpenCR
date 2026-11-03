r"""Elimination of quasi-static states: population coefficients and effective rates.

Split the states into the prescribed drivers ``P`` (the states with a density
in the conditions) and the rest ``Q``. With all reactions linear in the
states and ``n_e`` given, ``dn/dt = M n`` and the quasi-static ``Q`` follow
from ``M_QQ n_Q + M_QP n_P = 0``:

$$
\mathbf n_Q = C\,\mathbf n_P, \qquad C = -M_{QQ}^{-1} M_{QP},
$$

and the drivers then change at the effective rates

$$
\frac{d\mathbf n_P}{dt} = M^{\mathrm{eff}} \mathbf n_P, \qquad
M^{\mathrm{eff}} = M_{PP} + M_{PQ} C.
$$

``C / n_e`` are the population coefficients (m^3) and ``M_eff / n_e`` the
effective rate coefficients (m^3/s): for atomic hydrogen ``M_eff[H+, H(1)] / n_e``
is the collisional-radiative ionization coefficient and ``M_eff[H(1), H+] / n_e``
the recombination coefficient. The elementary reactions stay in the network;
the effective ones are a result, never mixed with them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from methodcard import Rule, method

from opencr.core.conditions import Conditions
from opencr.core.network import ReactionNetwork
from opencr.core.species import ELECTRON

__all__ = ["QuasiStatic", "SingularSystemError", "quasi_static"]


class SingularSystemError(ValueError):
    """The quasi-static system has no unique solution; the message says which states."""


@dataclass(frozen=True)
class QuasiStatic:
    """The quasi-static reduction of a network at fixed conditions.

    Attributes:
        drivers: The names of the prescribed states ``P``.
        excited: The names of the quasi-static states ``Q``.
        coefficients: ``C`` with shape ``(len(excited), len(drivers))``,
            dimensionless: ``n_Q = C n_P``.
        effective: ``M_eff`` with shape ``(len(drivers), len(drivers))`` in s^-1.
        electron_density: ``n_e`` in m^-3.
    """

    drivers: tuple[str, ...]
    excited: tuple[str, ...]
    coefficients: np.ndarray
    effective: np.ndarray
    electron_density: float

    @method(
        equations={
            "coefficient": r"\frac{n_q}{n_e\, n_d} = \frac{C_{qd}}{n_e} \quad [\mathrm{m^3}]",
        },
    )
    def population_coefficient(self, state: str, driver: str) -> float:
        """Return ``n_state / (n_e n_driver)`` in m^3."""
        c = self.coefficients[self.excited.index(state), self.drivers.index(driver)]
        return float(c / self.electron_density)

    @method(
        equations={
            "effective_rate": r"K_{\mathrm{source}\to\mathrm{target}} = \frac{M^{\mathrm{eff}}_{\mathrm{target},\mathrm{source}}}{n_e} \quad [\mathrm{m^3/s}]",
            "hydrogen": r"S_{\mathrm{CR}} = \frac{M^{\mathrm{eff}}_{\mathrm{H^+},\mathrm{H}(1)}}{n_e}, \qquad \alpha_{\mathrm{CR}} = \frac{M^{\mathrm{eff}}_{\mathrm{H}(1),\mathrm{H^+}}}{n_e}",
        },
        assumptions=(
            r"the diagonal ($\mathrm{source} = \mathrm{target}$) is minus the driver's total effective loss coefficient",
            r"three-body recombination enters $M_{\cdot,\mathrm{H^+}}$ with $k\,n_e^2$, so $\alpha_{\mathrm{CR}}$ depends on $n_e$",
            r"effective rates are results, never reactions of a network",
        ),
        references=(
            "[Summers et al., Plasma Phys. Control. Fusion 48, 263 (2006)](https://doi.org/10.1088/0741-3335/48/2/007)",
        ),
    )
    def effective_rate(self, source: str, target: str) -> float:
        """Return the effective rate coefficient from ``source`` to ``target`` in m^3/s.

        ``M_eff[target, source] / n_e``: the rate at which ``target`` is created
        per electron and per particle of ``source`` (``source != target``), or
        for ``source == target`` minus the total loss rate coefficient.
        """
        i, j = self.drivers.index(target), self.drivers.index(source)
        return float(self.effective[i, j] / self.electron_density)

    def populations(self, densities: dict[str, float]) -> np.ndarray:
        """Return ``n_Q`` for the driver densities ``densities``."""
        n_p = np.array([densities[d] for d in self.drivers])
        return self.coefficients @ n_p


def _solve(
    matrix: sps.csc_matrix, rhs: np.ndarray, names: tuple[str, ...]
) -> np.ndarray:
    """``matrix^-1 rhs`` by sparse LU, with a diagnosis when the matrix is singular."""
    diagonal = matrix.diagonal()
    lossless = [names[i] for i in np.flatnonzero(diagonal == 0.0)]
    if lossless:
        raise SingularSystemError(
            f"the states {lossless} have no loss process; they cannot be quasi-static. "
            "Prescribe their densities or add their losses."
        )
    try:
        lu = spla.splu(matrix)
    except RuntimeError as error:
        raise SingularSystemError(
            f"the quasi-static matrix is singular ({error}): a group of states has no path to "
            "the drivers"
        ) from error
    solution = lu.solve(np.asarray(rhs, dtype=float))
    # one step of iterative refinement
    residual = rhs - matrix @ solution
    return solution + lu.solve(residual)


def _rate_matrix(
    network: ReactionNetwork,
    rates: np.ndarray,
    conditions: Conditions,
    drivers: set[str],
) -> sps.csc_matrix:
    """Return ``M`` of ``dn/dt = M n``, linear in the quasi-static states.

    A reaction with one quasi-static heavy reactant is linear in it, with the
    densities of any driver reactants (and ``n_e``) in its coefficient; a
    reaction among drivers only is put on the column of its first heavy
    reactant. Two quasi-static reactants are not linear and are refused.
    """
    densities = conditions.densities
    electron_density = float(conditions.electron_density or 0.0)
    index = {name: i for i, name in enumerate(network.state_names)}
    rows, cols, values = [], [], []
    for r, reaction in enumerate(network.reactions):
        heavy = [s.name for s in reaction.reactants if s.species != ELECTRON]
        electrons = len(reaction.reactants) - len(heavy)
        coefficient = rates[r] * electron_density**electrons
        quasi = [h for h in heavy if h not in drivers]
        if len(quasi) > 1:
            raise ValueError(
                f"{reaction.name}: the quasi-static reduction needs at most one quasi-static "
                "reactant per reaction; prescribe one of them or solve the steady state"
            )
        column = quasi[0] if quasi else heavy[0]
        partners = list(heavy)
        partners.remove(column)
        for name in partners:
            coefficient *= densities[name]
        for name, change in reaction.net.items():
            if name in index:
                rows.append(index[name])
                cols.append(index[column])
                values.append(change * coefficient)
    size = len(network.state_names)
    return sps.csc_matrix((values, (rows, cols)), shape=(size, size))


def _quasi_static_rules() -> dict:
    """The SymPy rules of `quasi_static` (for the tests).

    A ground state ``G`` and a proton (drivers) and one excited state ``X``:
    excitation ``G -> X``, emission ``X -> G`` and ionization ``X -> H+``.
    """
    import sympy

    from opencr.core.rates import Constant
    from opencr.core.reactions import Reaction
    from opencr.core.species import ELECTRON_STATE as E
    from opencr.data.hydrogen import hydrogen_levels

    (ground, excited), ion = hydrogen_levels(2)

    def reduce(a: float, k12: float, kion: float, ne: float) -> QuasiStatic:
        network = ReactionNetwork(
            [ground, excited, ion],
            [
                Reaction(
                    "exc", (E, ground), (E, excited), "excitation", Constant(value=k12)
                ),
                Reaction(
                    "rad",
                    (excited,),
                    (ground,),
                    "spontaneous_emission",
                    Constant(value=a),
                ),
                Reaction(
                    "ion", (E, excited), (E, E, ion), "ionization", Constant(value=kion)
                ),
            ],
        )
        densities = {ground.name: 1.0, ion.name: 1.0}
        return quasi_static(
            network, np.array([k12, a, kion]), Conditions(1.0, ne, densities=densities)
        )

    def coefficient(a: float, k12: float, kion: float, ne: float) -> float:
        return float(reduce(a, k12, kion, ne).coefficients[0, 0])

    def ionization(a: float, k12: float, kion: float, ne: float) -> float:
        return reduce(a, k12, kion, ne).effective_rate(ground.name, ion.name)

    a, k12, kion, ne = sympy.symbols("a k12 kion ne", positive=True)
    ranges = {"a": (1e6, 1e9), "k12": (1e-16, 1e-14), "kion": (1e-15, 1e-13)}
    ranges["ne"] = (1e16, 1e21)
    return {
        "reduction": Rule(
            ne * k12 / (a + kion * ne), implementation=coefficient, ranges=ranges
        ),
        "effective": Rule(
            k12 * kion * ne / (a + kion * ne), implementation=ionization, ranges=ranges
        ),
    }


@method(
    equations={
        "matrix": r"M_{ij} = \sum_{r:\ j \text{ is the quasi-static reactant of } r} \nu_{ir}\, k_r\, n_e^{a_{re}} \prod_{p \in P,\, p \ne j} n_p^{a_{rp}}",
        "reduction": r"\frac{d\mathbf n_Q}{dt} = M_{QQ}\mathbf n_Q + M_{QP}\mathbf n_P = 0 \quad\Rightarrow\quad \mathbf n_Q = C\,\mathbf n_P, \qquad C = -M_{QQ}^{-1} M_{QP}",
        "effective": r"\frac{d\mathbf n_P}{dt} = M^{\mathrm{eff}} \mathbf n_P, \qquad M^{\mathrm{eff}} = M_{PP} + M_{PQ} C",
        "solve": r"M_{QQ} = LU, \qquad X_0 = U^{-1}L^{-1} M_{QP}, \qquad X = X_0 + U^{-1}L^{-1}\left(M_{QP} - M_{QQ} X_0\right)",
    },
    assumptions=(
        r"drivers $P$: the states with a density in the conditions; $Q$ the rest",
        r"every reaction linear in its one quasi-static reactant (two are refused); a reaction among drivers only goes on the column of its first heavy reactant",
        r"a given $n_e$ (a quasi-neutral one is refused)",
        r"sparse LU (`splu`) with one step of iterative refinement; a zero diagonal of $M_{QQ}$ (a state without losses) or a singular factorisation raises `SingularSystemError`",
    ),
    references=(
        "[Summers et al., Plasma Phys. Control. Fusion 48, 263 (2006)](https://doi.org/10.1088/0741-3335/48/2/007)",
        "[Virtanen et al., Nat. Methods 17, 261 (2020)](https://doi.org/10.1038/s41592-019-0686-2) (`splu`)",
    ),
    components=(QuasiStatic.population_coefficient, QuasiStatic.effective_rate),
    symbolic=_quasi_static_rules,
)
def quasi_static(
    network: ReactionNetwork, rates: np.ndarray, conditions: Conditions
) -> QuasiStatic:
    """Return the quasi-static reduction with the drivers ``conditions.densities``.

    Raises:
        ValueError: If ``n_e`` is not given or a reaction has two quasi-static
            reactants (reactions with driver partners are linear in the rest).
        SingularSystemError: If ``M_QQ`` is singular (a state without losses, or a
            group of states without a path out).
    """
    if conditions.quasi_neutral:
        raise ValueError("the quasi-static reduction needs a given electron density")
    drivers = tuple(n for n in network.state_names if n in conditions.densities)
    if not drivers:
        raise ValueError(
            "no drivers: give the prescribed densities in Conditions.densities"
        )
    matrix = _rate_matrix(network, rates, conditions, set(drivers))
    index = {name: i for i, name in enumerate(network.state_names)}
    p = [index[d] for d in drivers]
    excited = tuple(n for n in network.state_names if n not in drivers)
    q = [index[n] for n in excited]
    m_qq = matrix[q][:, q].tocsc()
    m_qp = matrix[q][:, p].toarray()
    m_pq = matrix[p][:, q]
    m_pp = matrix[p][:, p].toarray()
    if not q:
        coefficients = np.zeros((0, len(p)))
    else:
        coefficients = -_solve(m_qq, m_qp, excited)
    effective = m_pp + (m_pq @ coefficients if q else 0.0)
    return QuasiStatic(
        drivers,
        excited,
        coefficients,
        np.asarray(effective),
        float(conditions.electron_density or 0.0),
    )
