"""The population equations ``dn/dt = R(n)`` of a network, with their Jacobian.

:class:`PopulationSystem` compiles a network for one set of rate coefficients
and conditions: which states evolve, the densities of the prescribed ones,
and ``n_e`` (given, or quasi-neutral). Reaction ``r`` proceeds at

    w_r = k_r n_e^(a_re) prod_p n_p^(a_rp) prod_i n_i^(a_ri),

with ``p`` the prescribed and ``i`` the evolving reactants, and
``R = N w`` with the net stoichiometry ``N`` over the evolving states. The
Jacobian ``dR/dn = N dw/dn`` is assembled analytically in sparse form,
including the dependence of a quasi-neutral ``n_e`` on the charged states.

When every reaction has at most one evolving reactant and ``n_e`` is given, the
system is linear, ``R = M n + b`` (:meth:`PopulationSystem.linear_parts`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import scipy.sparse as sps

from opencr.core.conditions import Conditions
from opencr.core.network import ReactionNetwork
from opencr.core.species import ELECTRON

__all__ = ["MissingDensityError", "PopulationSystem"]


class MissingDensityError(ValueError):
    """A prescribed state has no density in the conditions."""


class PopulationSystem:
    """The population equations of the evolving states.

    Args:
        network: The reaction network.
        rates: ``k_r`` for every reaction (:class:`~opencr.kinetics.rates.RateEvaluator`).
        conditions: The conditions: ``n_e`` and the prescribed densities.
        evolving: The names of the evolving states; the others are prescribed
            and need a density in ``conditions.densities``.
    """

    def __init__(
        self,
        network: ReactionNetwork,
        rates: np.ndarray,
        conditions: Conditions,
        evolving: Sequence[str],
    ) -> None:
        """Compile the network."""
        self.network = network
        self.conditions = conditions
        self.evolving = tuple(evolving)
        unknown = set(self.evolving) - set(network.state_names)
        if unknown:
            raise KeyError(f"unknown evolving states {sorted(unknown)}")
        self.prescribed = tuple(
            n for n in network.state_names if n not in self.evolving
        )
        missing = [n for n in self.prescribed if n not in conditions.densities]
        if missing:
            raise MissingDensityError(
                f"no density given for the prescribed states {missing}; set them in "
                "Conditions.densities or let them evolve"
            )
        rates = np.asarray(rates, dtype=float)
        if rates.shape != (len(network.reactions),):
            raise ValueError("one rate coefficient per reaction is needed")
        self.rates = rates
        index = {name: i for i, name in enumerate(self.evolving)}
        self.index = index
        density = conditions.densities
        n_r = len(network.reactions)
        factor = rates.copy()
        electron_order = np.zeros(n_r)
        slots: list[list[int]] = []
        rows, cols, values = [], [], []
        for r, reaction in enumerate(network.reactions):
            own: list[int] = []
            for state in reaction.reactants:
                if state.species == ELECTRON:
                    electron_order[r] += 1
                elif state.name in index:
                    own.append(index[state.name])
                else:
                    factor[r] *= density[state.name]
            slots.append(own)
            for name, change in reaction.net.items():
                if name in index:
                    rows.append(index[name])
                    cols.append(r)
                    values.append(change)
        width = max((len(s) for s in slots), default=0)
        self._slots = np.full((n_r, max(width, 1)), -1, dtype=int)
        for r, own in enumerate(slots):
            self._slots[r, : len(own)] = own
        self._factor = factor
        self._electron_order = electron_order
        self.stoichiometry = sps.csr_matrix(
            (values, (rows, cols)), shape=(len(self.evolving), n_r)
        )
        self.charges = np.array(
            [network.state(n).charge for n in self.evolving], dtype=float
        )
        self._prescribed_charge = float(
            sum(network.state(n).charge * density[n] for n in self.prescribed)
        )
        self.max_order = width

    @property
    def size(self) -> int:
        """The number of evolving states."""
        return len(self.evolving)

    @property
    def is_linear(self) -> bool:
        """Whether ``R`` is linear in the evolving densities (given ``n_e``, one evolving reactant)."""
        return self.max_order <= 1 and not self.conditions.quasi_neutral

    def electron_density(self, n: np.ndarray) -> float:
        """Return ``n_e`` for the evolving densities ``n``."""
        if not self.conditions.quasi_neutral:
            return float(self.conditions.electron_density or 0.0)
        return float(
            self._prescribed_charge + self.charges @ np.asarray(n, dtype=float)
        )

    def reaction_rates(self, n: np.ndarray) -> np.ndarray:
        """Return ``w_r`` in m^-3 s^-1 for every reaction at the densities ``n``."""
        n = np.asarray(n, dtype=float)
        padded = np.append(n, 1.0)
        product = np.prod(padded[self._slots], axis=1)
        ne = max(self.electron_density(n), 0.0)
        return self._factor * ne**self._electron_order * product

    def rhs(self, n: np.ndarray) -> np.ndarray:
        """Return ``R(n) = dn/dt`` for the evolving states."""
        return self.stoichiometry @ self.reaction_rates(n)

    def jacobian(self, n: np.ndarray) -> sps.csr_matrix:
        """Return ``dR/dn`` as a sparse matrix."""
        n = np.asarray(n, dtype=float)
        padded = np.append(n, 1.0)
        values = padded[self._slots]
        ne = max(self.electron_density(n), 0.0)
        base = self._factor * ne**self._electron_order
        rows, cols, data = [], [], []
        for s in range(self._slots.shape[1]):
            active = self._slots[:, s] >= 0
            others = (
                np.prod(np.delete(values, s, axis=1), axis=1)
                if values.shape[1] > 1
                else 1.0
            )
            d = base * others
            r = np.flatnonzero(active)
            rows.append(r)
            cols.append(self._slots[r, s])
            data.append(np.broadcast_to(d, (len(self._slots),))[r])
        if self.conditions.quasi_neutral:
            charged = np.flatnonzero(self.charges)
            order = self._electron_order
            with np.errstate(divide="ignore", invalid="ignore"):
                dne = np.where(
                    order > 0,
                    self._factor
                    * order
                    * ne ** np.maximum(order - 1, 0)
                    * np.prod(values, axis=1),
                    0.0,
                )
            r_idx = np.flatnonzero(dne)
            for c in charged:
                rows.append(r_idx)
                cols.append(np.full(r_idx.size, c))
                data.append(dne[r_idx] * self.charges[c])
        derivative = sps.csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(len(self._slots), self.size),
        )
        return sps.csr_matrix(self.stoichiometry @ derivative)

    def linear_parts(self) -> tuple[sps.csr_matrix, np.ndarray]:
        """Return ``M`` and ``b`` of ``R = M n + b`` for a linear system."""
        if not self.is_linear:
            raise ValueError(
                "the system is not linear: a reaction has two evolving reactants or n_e is "
                "quasi-neutral"
            )
        zero = np.zeros(self.size)
        return self.jacobian(zero), self.rhs(zero)

    def source_by_prescribed(self) -> dict[str, np.ndarray]:
        """Return ``b`` split by prescribed state: the source with only that state present.

        The remainder (sources that need two prescribed states at once) is under
        ``"(mixed)"`` when it is not zero.
        """
        _, total = self.linear_parts()
        parts: dict[str, np.ndarray] = {}
        for name in self.prescribed:
            alone = {
                k: (v if k == name else 0.0)
                for k, v in self.conditions.densities.items()
            }
            system = PopulationSystem(
                self.network,
                self.rates,
                self.conditions.replace(densities=alone),
                self.evolving,
            )
            parts[name] = system.rhs(np.zeros(self.size))
        rest = total - sum(parts.values(), np.zeros(self.size))
        if np.any(np.abs(rest) > 1e-12 * max(np.abs(total).max(initial=0.0), 1e-300)):
            parts["(mixed)"] = rest
        return parts

    def conservation(self) -> tuple[tuple[str, ...], np.ndarray]:
        """Return the conserved elements and their matrix over the evolving states.

        An element is conserved by the evolving states when no reaction moves it
        to or from a prescribed state or out of the system (wall losses, pumping).
        """
        matrix = []
        names = []
        conservation = self.network.conservation()
        columns = [self.network.state_index(n) for n in self.evolving]
        for name, row in zip(conservation.names, conservation.matrix, strict=True):
            counts = row[columns]
            if not np.any(counts):
                continue
            change = counts @ self.stoichiometry
            if np.allclose(change, 0.0, atol=1e-12):
                matrix.append(counts)
                names.append(name)
        return tuple(names), np.asarray(matrix, dtype=float).reshape(
            len(names), self.size
        )

    def densities(self, n: np.ndarray) -> Mapping[str, float]:
        """Return every state's density by name: the evolving ``n`` and the prescribed ones."""
        out = {name: float(n[i]) for i, name in enumerate(self.evolving)}
        out.update({name: self.conditions.densities[name] for name in self.prescribed})
        return out
