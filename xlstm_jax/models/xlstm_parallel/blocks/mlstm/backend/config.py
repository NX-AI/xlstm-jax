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
        """
        Forward pass of the mLSTM cell.

        Args:
            q: Query tensor of shape (B, NH, S, DH).
            k: Key tensor of shape (B, NH, S, DH).
            v: Value tensor of shape (B, NH, S, DH).
            i: Input gate tensor of shape (B, NH, S, 1).
            f: Forget gate tensor of shape (B, NH, S, 1).

        Returns:
            Output tensor of shape (B, NH, S, DH).
        """

    @property
    def can_vmap_over_heads(self) -> bool:
        """
        Whether the backend can be vmaped over the heads dimension.

        Should be False if the backend requires manual transposition of the input tensors to work over heads.
        """
        raise NotImplementedError
