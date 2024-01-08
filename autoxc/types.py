from typing import Union, Any, Callable

import numpy as np
import jax
from pyscf import scf

Array = Union[np.ndarray, jax.Array]
SCFLike = Union[scf.rhf.RHF, scf.uhf.UHF]
PyTree = Any
Functional = Callable[..., Array]
