import jax
import numpy as np
from flax.struct import dataclass


@dataclass
class Batch:
    inputs: jax.Array
    labels: jax.Array

    def __getitem__(self, key):
        """Supports slicing and element access in batch."""
        vals = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (np.ndarray, jax.Array)):
                vals[k] = v[key]
            else:
                vals[k] = v
        return self.__class__(**vals)
