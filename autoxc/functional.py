from typing import Optional, NamedTuple, Callable
from functools import partial, wraps
from warnings import warn

import jax
from jax import numpy as jnp

from pyscf.dft import libxc

from .derivatives import libxc_derivatives
from .utils import as_numpy_arrays
from .types import Array, SCFLike, Functional

__all__ = ["set_custom_functional"]


class FunctionalInputs(NamedTuple):
    rho: Array
    gamma: Optional[Array] = None
    tau: Optional[Array] = None
    omega: Optional[Array] = None


def functional_inputs_spinless(rho: Array):

    if rho.ndim == 1:
        return (rho,)

    if rho.ndim != 2 or rho.shape[0] not in [4, 6]:
        raise ValueError(f"Invalid input shape: {rho.shape}")

    rho0 = rho[0]
    gamma = jnp.sum(rho[1:4] ** 2, axis=0)
    tau = rho[5] if rho.shape[0] == 6 else None

    return rho0, gamma, tau


def functional_inputs_spin(rho: Array):

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


def functional_inputs(rho: Array, omega: Optional[Array] = None, spin: int = 0) -> FunctionalInputs:

    if omega is not None:
        raise NotImplementedError("Non-local exact exchange is not supported yet.")

    if spin == 0:
        args = functional_inputs_spinless(rho, omega)
    else:
        args = functional_inputs_spin(rho, omega)

    return FunctionalInputs(*args)


####################################################################################################

WrappedFunctional = Callable[[FunctionalInputs], Array]


def wrap_functional(f: Functional, *args, **kwargs) -> WrappedFunctional:
    @wraps(f)
    def wrapped(inputs: FunctionalInputs):
        return f(*inputs, *args, **kwargs)

    return wrapped


def _maybe_jit(f, jittable, *args, **kwargs):
    return jax.jit(f, *args, **kwargs) if jittable else f


def make_eval_xc(functional: WrappedFunctional, jittable: bool = True):
    @partial(_maybe_jit, jittable=jittable, static_argnames=["spin", "deriv"])
    def _eval_xc_aux(rho, omega, spin, deriv):
        inputs = functional_inputs(rho, omega, spin)
        deriv_fn = libxc_derivatives(functional, spin, deriv)
        return deriv_fn(inputs)

    @as_numpy_arrays
    def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
        del xc_code, relativity, verbose
        return _eval_xc_aux(rho, omega, spin=spin, deriv=deriv)

    return eval_xc


def set_custom_functional(
    ks: SCFLike, f: Functional, xctype: str = "LDA", jittable: bool = True, *args, **kwargs
) -> SCFLike:

    if not jax.config.read("jax_enable_x64"):
        warn(
            "Double precision is disabled in JAX."
            "Enabling it for SCF calculations is recommended."
        )

    wrapped_func = wrap_functional(f, *args, **kwargs)
    eval_xc = make_eval_xc(wrapped_func, jittable)

    ks.xc = ""
    libxc.define_xc_(ks._numint, eval_xc, xctype=xctype)

    return ks
