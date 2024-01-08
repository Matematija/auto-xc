import jax
from jax import numpy as jnp

from .functional import FunctionalInputs, WrappedFunctional

__all__ = ["libxc_derivatives"]


# * vxc = (vrho, vsigma, vlapl, vtau) for restricted case

# * vxc for unrestricted case
# | vrho[:,2]   = (u, d)
# | vsigma[:,3] = (uu, ud, dd)
# | vlapl[:,2]  = (u, d)
# | vtau[:,2]   = (u, d)


def process_grads(grads):
    vrho, vgamma, vtau = grads
    return vrho, vgamma, None, vtau


########################################################################

# * fxc for restricted case:
#     (v2rho2, v2rhosigma, v2sigma2,
#      v2lapl2, vtau2, v2rholapl,
#      v2rhotau, v2lapltau, v2sigmalapl,
#      v2sigmatau)

# * fxc for unrestricted case:
#     | v2rho2[:,3]     = (u_u, u_d, d_d)
#     | v2rhosigma[:,6] = (u_uu, u_ud, u_dd, d_uu, d_ud, d_dd)
#     | v2sigma2[:,6]   = (uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd)
#     | v2lapl2[:,3]
#     | v2tau2[:,3]     = (u_u, u_d, d_d)
#     | v2rholapl[:,4]
#     | v2rhotau[:,4]   = (u_u, u_d, d_u, d_d)
#     | v2lapltau[:,4]
#     | v2sigmalapl[:,6]
#     | v2sigmatau[:,6] = (uu_u, uu_d, ud_u, ud_d, dd_u, dd_d)


def process_lda_hessian(hessian, spin):

    if spin == 0:
        return hessian.rho.rho
    else:
        i, j = jnp.triu_indices(2)
        return hessian.rho.rho[:, i, j]


def process_gga_hessian(hessian, spin):

    if spin == 0:
        v2rhosigma = hessian.rho.gamma
        v2sigma2 = hessian.gamma.gamma
    else:
        i, j = jnp.triu_indices(3)
        v2rhosigma = hessian.rho.gamma.reshape(-1, 6)
        v2sigma2 = hessian.gamma.gamma[:, i, j]

    return v2rhosigma, v2sigma2


def process_mgga_hessian(hessian, spin):

    if spin == 0:
        v2rhotau = hessian.rho.tau
        v2sigmatau = hessian.gamma.tau
        v2tau2 = hessian.tau.tau

    else:
        i, j = jnp.triu_indices(2)

        v2rhotau = hessian.rho.tau.reshape(-1, 4)
        v2sigmatau = hessian.gamma.tau.reshape(-1, 6)
        v2tau2 = hessian.tau.tau[:, i, j]

    return v2rhotau, v2sigmatau, v2tau2


def process_hessian(hessian, spin):

    v2rho2 = process_lda_hessian(hessian, spin)

    v2rhosigma, v2sigma2, v2tau2, v2rhotau, v2sigmatau = (None,) * 5
    v2lapl2, v2rholapl, v2lapltau, v2sigmalapl = (None,) * 4

    if hessian.gamma is not None:
        v2rhosigma, v2sigma2 = process_gga_hessian(hessian, spin)

    if hessian.tau is not None:
        assert hessian.gamma is not None, "Cannot have a MGGA without gradients!"
        v2rhotau, v2sigmatau, v2tau2 = process_mgga_hessian(hessian, spin)

    return (
        v2rho2,
        v2rhosigma,
        v2sigma2,
        v2lapl2,
        v2tau2,
        v2rholapl,
        v2rhotau,
        v2lapltau,
        v2sigmalapl,
        v2sigmatau,
    )


########################################################################


def libxc_derivatives(functional: WrappedFunctional, spin: int = 0, deriv: int = 1):

    assert deriv in [0, 1, 2], f"Unknown derivative order: {deriv}"

    if deriv == 0:
        return lambda *args: (functional(*args), None, None, None)

    def eval_exc_aux(inputs: FunctionalInputs):
        exc = functional(inputs)
        return inputs.rho * exc, exc

    if deriv >= 2:
        hess_fn = jax.vmap(jax.hessian(eval_exc_aux, has_aux=True))

    def derivatives(inputs: FunctionalInputs):

        _, back, exc = jax.vjp(eval_exc_aux, inputs, has_aux=True)
        (grads,) = back(jnp.ones_like(exc))

        vxc = process_grads(grads, spin)
        fxc, kxc = None, None

        if deriv >= 2:
            hess, _ = hess_fn(inputs)
            fxc = process_hessian(hess, spin)

        return exc, vxc, fxc, kxc

    return derivatives
