import numpy as np

from jax.tree_util import tree_map


def _as_numpy_array(x):
    if isinstance(x, np.ndarray) or x is None:
        return x
    else:
        return np.asarray(x)


def as_numpy_arrays(f):
    def wrapped(*args, **kwargs):
        out = f(*args, **kwargs)
        return tree_map(_as_numpy_array, out)

    return wrapped
