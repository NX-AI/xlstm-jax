# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Poeppel
from abc import abstractmethod
from typing import Any

import jax
from flax import linen as nn


class mLSTMBackend(nn.Module):
    config: Any

    @nn.compact
    @abstractmethod
    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        i: jax.Array,
        f: jax.Array,
    ) -> jax.Array:
        pass
