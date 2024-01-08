from typing import Union, Any, Callable
from pyscf import scf
from jax.typing import ArrayLike

SCFLike = Union[scf.rhf.RHF, scf.uhf.UHF]
PyTree = Any
Functional = Callable[..., ArrayLike]
