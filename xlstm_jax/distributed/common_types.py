from typing import Any

import flax.linen as nn
import jax
from flax.struct import dataclass
from flax.training import train_state

from xlstm_jax import dataset

PyTree = Any
Metrics = dict[str, tuple[jax.Array, ...]]
Parameter = jax.Array | nn.Partitioned
PRNGKeyArray = jax.Array
Batch = dataset.Batch


class TrainState(train_state.TrainState):
    rng: jax.Array
