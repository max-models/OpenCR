"""The optional JAX backend: jitted rates, vmapped scans, automatic-differentiation Jacobians.

Nothing in the core imports JAX; this module does, and it is only usable
with the ``jax`` extra installed. It reuses the reference's prepared rate
evaluators with ``jax.numpy`` as the array namespace and the structure of a
:class:`~opencr.kinetics.system.PopulationSystem`, so that

* :meth:`JaxModel.rates` evaluates every reaction's Maxwellian rate
  coefficient at one or many electron temperatures in one jitted call;
* :meth:`JaxModel.rhs` and :meth:`JaxModel.jacobian` give ``R(n)`` and
  ``dR/dn`` of the population equations by forward-mode automatic
  differentiation, for the general (nonlinear, quasi-neutral) equations;
* :meth:`JaxModel.reduction` is the quasi-static reduction of a batch of
  ``(T_e, n_e)`` with ``jax.vmap``, the counterpart of
  :mod:`opencr.backends.batched`;
* :meth:`JaxModel.effective_rate` is differentiable: ``jax.grad`` of it
  gives the sensitivity of an effective coefficient to ``T_e`` and ``n_e``.

Two things differ from the reference: a :class:`~opencr.core.rates.Function`
rate (Python code) is tabulated on its ``file_temperatures`` and interpolated
in log-log (the same as when it is written to a file), and a non-Maxwellian
EEDF is refused. The first call of each jitted function compiles it; the
benchmarks separate that from the warm time. Computations are in float64
(``jax_enable_x64``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cunumpy as xp
import jax
import jax.numpy as jnp
import numpy as np

from opencr.backends.batched import BatchedQuasiStatic, BatchedReduction
from opencr.core.conditions import Conditions
from opencr.core.rates import Function
from opencr.core.reactions import Reaction
from opencr.kinetics.effective import SingularSystemError
from opencr.kinetics.rates import _Prepared
from opencr.kinetics.system import PopulationSystem
from opencr.model import Model

jax.config.update("jax_enable_x64", True)

__all__ = ["JaxModel"]


class JaxModel:
    """A model on JAX, at fixed heavy-particle temperatures and escape factors.

    Args:
        model: The model.
        conditions: The conditions whose heavy temperatures, escape factors
            and prescribed densities are baked in; ``T_e`` and ``n_e`` are
            the arguments of the methods. A non-Maxwellian EEDF is refused.
    """

    def __init__(self, model: Model, conditions: Conditions) -> None:
        """Prepare the rate evaluators and the structure arrays."""
        if conditions.eedf is not None and not conditions.eedf.is_maxwellian:
            raise ValueError("the JAX backend is Maxwellian; it refuses an EEDF")
        self.model = model
        self.conditions = conditions
        evaluator = model.evaluator
        index = {r.name: i for i, r in enumerate(model.network.reactions)}
        self._prepared: list[_Prepared] = []
        for reaction, prepared in zip(
            model.network.reactions, evaluator._prepared, strict=True
        ):
            if isinstance(reaction.rate, Function):
                tabulated = Reaction(
                    reaction.name,
                    reaction.reactants,
                    reaction.products,
                    reaction.kind,
                    reaction.rate.tabulate(),
                    reaction.provenance,
                    reaction.escape,
                )
                prepared = evaluator._prepare(tabulated, index)
            self._prepared.append(prepared)
        self._evaluator = evaluator
        self._rates: Callable[[Any], Any] = jax.jit(self._rates_impl)
        # the batched structure (linear networks only; else no reduction)
        self._batched: BatchedQuasiStatic | None
        try:
            self._batched = BatchedQuasiStatic(model)
        except ValueError:
            self._batched = None
        if self._batched is not None:
            drivers, excited, p, q = self._batched.partition(conditions)
            self._drivers, self._excited = drivers, excited
            self._partition = (jnp.asarray(p), jnp.asarray(q))
            self._n_matrix = jnp.asarray(xp.to_numpy(self._batched._stoichiometry))
            self._selector = jnp.asarray(xp.to_numpy(self._batched._selector))
            self._matrix_order = jnp.asarray(xp.to_numpy(self._batched._electron_order))
        # the population equations of the evolving states, as PopulationSystem
        evolving = [
            n for n in model.network.state_names if n not in conditions.densities
        ]
        system = PopulationSystem(
            model.network, np.ones(len(model.network.reactions)), conditions, evolving
        )
        self.system = system
        self._slots = jnp.asarray(
            np.where(system._slots < 0, system.size, system._slots)
        )
        self._prescribed = jnp.asarray(system._factor)  # the rates were ones
        self._order = jnp.asarray(system._electron_order)
        self._stoichiometry = jnp.asarray(system.stoichiometry.toarray())
        self._charges = jnp.asarray(system.charges)
        self._prescribed_charge = float(system._prescribed_charge)
        self._rhs = jax.jit(self._rhs_impl)
        self._jacobian = jax.jit(jax.jacfwd(self._rhs_impl))
        self._reduce = jax.jit(jax.vmap(self._reduce_impl))
        self._effective = jax.jit(self._effective_impl, static_argnums=(0, 1))

    def _rates_impl(self, electron_temperature: Any) -> Any:
        te = jnp.atleast_1d(jnp.asarray(electron_temperature, dtype=float))
        return self._evaluator.evaluate(self._prepared, self.conditions, te, jnp)

    def rates(self, electron_temperature: Any) -> Any:
        """Return ``k`` of shape ``(..., n_reactions)`` for Maxwellian electrons at ``T_e`` (eV)."""
        k = self._rates(electron_temperature)
        return k[0] if jnp.ndim(electron_temperature) == 0 else k

    def _rhs_impl(
        self, n: Any, electron_temperature: Any, electron_density: Any
    ) -> Any:
        k = self._rates(electron_temperature)[0]
        padded = jnp.append(n, 1.0)
        product = jnp.prod(padded[self._slots], axis=1)
        ne = jnp.where(
            electron_density > 0.0,
            electron_density,
            self._prescribed_charge + self._charges @ n,
        )
        ne = jnp.maximum(ne, 0.0)
        w = k * self._prescribed * ne**self._order * product
        return self._stoichiometry @ w

    def rhs(
        self, n: Any, electron_temperature: float, electron_density: float | None = None
    ) -> Any:
        """Return ``R(n) = dn/dt`` of the evolving states (m^-3 s^-1).

        ``electron_density=None`` means quasi-neutrality, as in the conditions.
        """
        ne = -1.0 if electron_density is None else float(electron_density)
        return self._rhs(jnp.asarray(n, dtype=float), float(electron_temperature), ne)

    def jacobian(
        self, n: Any, electron_temperature: float, electron_density: float | None = None
    ) -> Any:
        """Return ``dR/dn`` as a dense array by forward-mode automatic differentiation."""
        ne = -1.0 if electron_density is None else float(electron_density)
        return self._jacobian(
            jnp.asarray(n, dtype=float), float(electron_temperature), ne
        )

    def _reduce_impl(self, electron_temperature: Any, electron_density: Any) -> Any:
        k = self._rates(electron_temperature)[0]
        weighted = k * electron_density**self._matrix_order
        matrix = jnp.einsum("ir,r,rj->ij", self._n_matrix, weighted, self._selector)
        p, q = self._partition
        m_qq = matrix[jnp.ix_(q, q)]
        coefficients = -jnp.linalg.solve(m_qq, matrix[jnp.ix_(q, p)])
        effective = matrix[jnp.ix_(p, p)] + matrix[jnp.ix_(p, q)] @ coefficients
        return k, coefficients, effective

    def reduction(
        self, electron_temperatures: Any, electron_densities: Any
    ) -> BatchedReduction:
        """Return the quasi-static reduction of every ``(T_e, n_e)`` pair (vmapped).

        The drivers are the prescribed states of the conditions, else the
        model's. The result holds NumPy arrays.
        """
        te, ne = np.broadcast_arrays(
            np.asarray(electron_temperatures, dtype=float),
            np.asarray(electron_densities, dtype=float),
        )
        te, ne = np.ravel(te), np.ravel(ne)
        batched = self._reducible()
        k, coefficients, effective = self._reduce(jnp.asarray(te), jnp.asarray(ne))
        c = np.asarray(coefficients)
        if not np.all(np.isfinite(c)):
            raise SingularSystemError(
                "the quasi-static matrix is singular at some point: a state without "
                "losses or a group of states with no path to the drivers"
            )
        return batched.assemble(
            te,
            ne,
            self._drivers,
            self._excited,
            c,
            np.asarray(effective),
            np.asarray(k),
        )

    def _reducible(self) -> BatchedQuasiStatic:
        if self._batched is None:
            raise ValueError(
                "the quasi-static reduction needs every reaction to have at most one "
                "heavy reactant"
            )
        if self._partition[1].size == 0:
            raise ValueError("every state is a driver: nothing to reduce")
        return self._batched

    def effective_rate(
        self, source: str, target: str, electron_temperature: Any, electron_density: Any
    ) -> Any:
        """Return ``M_eff[target, source] / n_e`` (m^3/s) as a differentiable JAX scalar.

        ``jax.grad(jax_model.effective_rate, argnums=2)(src, tgt, T_e, n_e)`` is
        the sensitivity to ``T_e``.
        """
        self._reducible()
        i, j = self._drivers.index(target), self._drivers.index(source)
        return self._effective(
            i,
            j,
            jnp.asarray(electron_temperature, dtype=float),
            jnp.asarray(electron_density, dtype=float),
        )

    def _effective_impl(self, i: int, j: int, te: Any, ne: Any) -> Any:
        _, _, effective = self._reduce_impl(te, ne)
        return effective[i, j] / ne
