from typing import Optional, NamedTuple

from jax import numpy as jnp

from .types import ArrayLike

__all__ = ["functional_inputs", "FunctionalInputs"]


class FunctionalInputs(NamedTuple):
    rho: ArrayLike
    gamma: Optional[ArrayLike] = None
    tau: Optional[ArrayLike] = None


def functional_inputs_spinless(rho: ArrayLike):

    if rho.ndim == 1:
        return (rho,)

    if rho.ndim != 2 or rho.shape[0] not in [4, 6]:
        raise ValueError(f"Invalid input shape: {rho.shape}")

    rho0 = rho[0]
    gamma = jnp.sum(rho[1:4] ** 2, axis=0)
    tau = rho[5] if rho.shape[0] == 6 else None

    return rho0, gamma, tau


def functional_inputs_spin(rho: ArrayLike):

    if rho.ndim == 2:
        return (rho,)

    if rho.ndim != 3 or rho.shape[1] not in [4, 6]:
        raise ValueError(f"Invalid input shape: {rho.shape}")

    rho0 = rho[:, 0, :].transpose()

    grads_a, grads_b = rho[:, 1:4, :]
    gamma_aa = jnp.sum(grads_a**2, axis=0)
    gamma_bb = jnp.sum(grads_b**2, axis=0)
    gamma_ab = jnp.sum(grads_a * grads_b, axis=0)
    gamma = jnp.stack([gamma_aa, gamma_ab, gamma_bb], axis=1)

    tau = rho[:, 5, :].T if rho.shape[1] == 6 else None

    return rho0, gamma, tau


def functional_inputs(
    rho: ArrayLike, omega: Optional[ArrayLike] = None, spin: int = 0
) -> FunctionalInputs:

    if omega is not None:
        raise NotImplementedError("Non-local exact exchange is not supported yet.")

    if spin == 0:
        args = functional_inputs_spinless(rho)
    else:
        args = functional_inputs_spin(rho)

    return FunctionalInputs(*args)
