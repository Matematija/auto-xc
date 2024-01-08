# <p style="text-align: center;">Auto XC</p>

Automatic differentiation tools for custom density functionals with libXC in quantum chemistry. Auto XC uses the [JAX](https://github.com/google/jax) automatic differentiation engine and [PySCF](https://github.com/pyscf/pyscf) to interface with [libXC](https://gitlab.com/libxc/libxc).

## Example usage:

A simple PySCF example - methane.

```python
from pyscf import gto, dft

atom = """
    C  0.00000000, 0.00000000, 0.00000000;
    H  0.62911800, 0.62911800, 0.62911800;
    H  -0.62911800, -0.62911800, 0.62911800;
    H  0.62911800, -0.62911800, -0.62911800;
    H  -0.62911800, 0.62911800, -0.62911800;
"""

mol = gto.M(atom=atom, basis='ccpvdz')
```

A simple PBE calculation:

```python
mf = dft.RKS(mol)
mf.xc = 'pbe,' # PBE exchange only, libXC version
mf.kernel()
```
```
converged SCF energy = -40.1385994117519
```

Now define the PBE exchange analytically and auto-generate the required derivatives:

```python
from math import pi
import autoxc

def pbe_x(rho, gamma):

    kappa, mu = 0.804, 0.2195

    lda = -(3 / 4) * (3 * rho / pi) ** (1 / 3)
    scale = 4 * (3 * pi**2) ** (2 / 3) * rho ** (8 / 3)

    x = mu * gamma / scale
    F = 1 + x / (1 + x / kappa)

    return lda * F

mf = dft.RKS(mol) # Spin-restricted calculation
autoxc.custom_functional(mf, pbe_x, xctype='gga')
mf.kernel()
```
```
converged SCF energy = -40.13856531146
```

## Conventions and restrictions

Any density functional can be experimented with this way by passing the function $\varepsilon _{xc}$ to `autoxc.custom_functional` as long as it is JAX-differentiable. We use the following convention:

$$
E_{xc} \left[ \rho \right] = \int \text{d} ^3 \mathbf{r} \; \rho(\mathbf{r}) \, \varepsilon_{xc}\left( \rho(\mathbf{r}), \gamma (\mathbf{r}), \tau (\mathbf{r}) \right) 
$$

where $\gamma (\mathbf{r}) = \left| \nabla \rho (\mathbf{r}) \right| ^2$ and $\tau (\mathbf{r}) = \frac{1}{2} \sum _a \left| \nabla \varphi _a (\mathbf{r}) \right| ^2$ ($\varphi _a$ are Kohn-Sham orbitals). At the moment, we support:
* Local density approximations (LDA): $\varepsilon _{xc} \longrightarrow \varepsilon _{xc} \left( \rho (\mathbf{r}) \right)$
* Generalized gradient approximations (GGA): $\varepsilon _{xc} \longrightarrow \varepsilon _{xc} \left( \rho (\mathbf{r}), \gamma (\mathbf{r}) \right)$
* Meta-GGAs: $\varepsilon _{xc} \longrightarrow \varepsilon _{xc} \left( \rho (\mathbf{r}), \gamma (\mathbf{r}), \tau (\mathbf{r}) \right)$

First and second derivatives (time-dependent DFT) can be auto-generated for all of the above. Third derivatives are not supported yet. Both spin-restricted and unrestricted calculations are supported in all cases.

If there are any missing features you would like to see or bugs to fix, please open an issue or a pull request.