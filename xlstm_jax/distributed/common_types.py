from typing import Any

import flax.linen as nn
import jax
from flax.training import train_state

PyTree = Any
Metrics = dict[str, tuple[jax.Array, ...]]
Parameter = jax.Array | nn.Partitioned
PRNGKeyArray = jax.Array


class TrainState(train_state.TrainState):
    rng: jax.Array
