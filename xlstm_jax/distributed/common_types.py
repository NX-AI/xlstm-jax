from typing import Any

import flax.linen as nn
import jax
from flax.struct import dataclass
from flax.training import train_state

PyTree = Any
Metrics = dict[str, tuple[jax.Array, ...]]
Parameter = jax.Array | nn.Partitioned


class TrainState(train_state.TrainState):
    rng: jax.Array


@dataclass
class Batch:
    inputs: jax.Array
    labels: jax.Array
