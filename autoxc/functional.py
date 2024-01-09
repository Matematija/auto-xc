from typing import Callable

from functools import partial, wraps
from warnings import warn

import jax

from pyscf.dft import libxc

from .derivatives import libxc_derivatives
from .utils import as_numpy_arrays
from .inputs import functional_inputs, FunctionalInputs
from .types import SCFLike, Functional, ArrayLike

__all__ = ["custom_functional"]

WrappedFunctional = Callable[[FunctionalInputs], ArrayLike]


def wrap_functional(f: Functional, **kwargs) -> WrappedFunctional:
    @wraps(f)
    def wrapped(inputs: FunctionalInputs, *args):
        density_args = tuple(x for x in inputs if x is not None)
        return f(*density_args, *args, **kwargs)

    return wrapped


def _maybe_jit(f, jittable, *args, **kwargs):
    return jax.jit(f, *args, **kwargs) if jittable else f


def make_eval_xc(functional: WrappedFunctional, jittable: bool = True):
    @partial(_maybe_jit, jittable=jittable, static_argnames=["spin", "deriv"])
    def _eval_xc_aux(rho, omega, deriv, spin):
        inputs = functional_inputs(rho, omega, spin)
        deriv_fn = libxc_derivatives(functional, spin, deriv)
        return deriv_fn(inputs)

    @as_numpy_arrays
    def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
        del xc_code, relativity, verbose
        return _eval_xc_aux(rho, omega, deriv, spin)

    return eval_xc


def custom_functional(
    ks: SCFLike, f: Functional, xctype: str, jittable: bool = True, **kwargs
) -> SCFLike:

    if not jax.config.read("jax_enable_x64"):
        warn(
            "Double precision is disabled in JAX."
            "Enabling it for SCF calculations is recommended."
        )

    xctype = xctype.strip().upper()

    if xctype not in ["LDA", "GGA", "MGGA"]:
        raise NotImplementedError(f"`xctype` must be one of 'LDA', 'GGA', 'MGGA'. Got {xctype}.")

    wrapped_func = wrap_functional(f, **kwargs)
    eval_xc = make_eval_xc(wrapped_func, jittable)

    ks.xc = ""
    libxc.define_xc_(ks._numint, eval_xc, xctype=xctype)

    return ks
