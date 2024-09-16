from typing import Any

from xlstm_jax.distributed.common_types import PRNGKeyArray

import jax
from flax.training import train_state


class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include mutable variables
    # like batch statistics. If a model has no mutable vars, it is None.
    mutable_variables: Any = None
    # RNG kept for init, dropout, etc.
    rng: PRNGKeyArray | None = None
