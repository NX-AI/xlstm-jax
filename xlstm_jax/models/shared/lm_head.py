from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

from xlstm_jax.distributed.array_utils import split_array_over_mesh
from xlstm_jax.distributed.data_parallel import shard_module_params

from ..configs import ParallelConfig
from .utils import soft_cap_logits


class TPLMHead(nn.Module):
    """
    Language model head with Tensor Parallelism.

    Args:
        parallel: Configuration for parallelism.
        vocab_size: Size of the vocabulary.
        kernel_init: Initializer for the output layer.
        norm_fn: Normalization function to apply before the output layer. If None, no normalization is applied.
        lm_head_dtype: Data type for the output layer.
        logits_soft_cap: Soft cap for the logits. If not None, the logits will be clipped to this value.
    """

    parallel: ParallelConfig
    vocab_size: int
    kernel_init: nn.initializers.Initializer
    norm_fn: Callable[..., nn.Module] | None
    lm_head_dtype: jnp.dtype = jnp.float32
    logits_soft_cap: float | None = None

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array:
        """
        Apply the language model head to the input features.

        Args:
            x: Input features, shape (..., hidden size).

        Returns:
            The logits, shape (..., vocab_size).
        """
        # Gather outputs over feature dimension and split over sequence length.
        x = jax.lax.all_gather(x, axis_name=self.parallel.model_axis_name, axis=-1, tiled=True)
        x = split_array_over_mesh(x, axis_name=self.parallel.model_axis_name, split_axis=1)
        # Apply norm - Shard parameters over model axis.
        if self.norm_fn is not None:
            norm_fn = shard_module_params(
                self.norm_fn,
                axis_name=self.parallel.model_axis_name,
                min_weight_size=self.parallel.fsdp_min_weight_size,
            )
            x = norm_fn()(x)
        # Apply output layer - Shard parameters over model axis.
        dense_fn = shard_module_params(
            nn.Dense,
            axis_name=self.parallel.model_axis_name,
            min_weight_size=self.parallel.fsdp_min_weight_size,
        )
        x = dense_fn(
            features=self.vocab_size,
            kernel_init=self.kernel_init,
            use_bias=False,
            dtype=self.lm_head_dtype,
            name="out_dense",
        )(x)
        # Output will be enforced to be float32.
        x = x.astype(jnp.float32)
        # Apply soft cap to the logits.
        x = soft_cap_logits(x, self.logits_soft_cap)
        return x
