import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass


@dataclass
class Batch:
    inputs: jax.Array
    targets: jax.Array

    def __getitem__(self, key):
        """Supports slicing and element access in batch."""
        vals = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (np.ndarray, jax.Array)):
                vals[k] = v[key]
            else:
                vals[k] = v
        return self.__class__(**vals)


@dataclass
class LLMBatch(Batch):
    """Batch for LLM training.

    Contains inputs and targets along with their respective positions and segmentations.

    Attributes:
        inputs (jax.Array): The input data.
        targets (jax.Array): The target data.
        inputs_position (jax.Array): Positions of the input tokens.
        inputs_segmentation (jax.Array): Segmentation of the input tokens. 0 to indicate padding.
        targets_position (jax.Array): Positions of the target tokens.
        targets_segmentation (jax.Array): Segmentation of the target tokens. 0 to indicate padding.
    """

    inputs_position: jax.Array
    inputs_segmentation: jax.Array
    targets_position: jax.Array
    targets_segmentation: jax.Array

    @staticmethod
    def from_inputs(inputs: jax.Array, targets: jax.Array | None = None) -> "LLMBatch":
        """Create LLMBatch from inputs.

        Helper function for quickly creating a default LLM Batch.

        Args:
            inputs (jax.Array): The input data.
            targets (jax.Array, optional): The target data. If not provided, the inputs are used as targets and the
                inputs are shifted right by one.
        """
        if targets is None:
            targets = inputs
            # Shift inputs to the right by padding and slicing.
            inputs = np.pad(inputs, ((0, 0), (1, 0)), mode="constant", constant_values=inputs.dtype.type(0))[:, :-1]
        position = jnp.arange(inputs.shape[1], dtype=jnp.int32)
        position = jnp.broadcast_to(position, inputs.shape)
        segmentation = jnp.ones(inputs.shape, dtype=jnp.int32)
        return LLMBatch(
            inputs=inputs,
            targets=targets,
            inputs_position=position,
            inputs_segmentation=segmentation,
            targets_position=position,
            targets_segmentation=segmentation,
        )

    @staticmethod
    def get_dtype_struct(batch_size: int, max_length: int) -> "LLMBatch":
        """Get the shape and dtype structure for LLMBatch.

        Args:
            batch_size (int): The size of the batch.
            max_length (int): The maximum length of the sequences.
        """
        return LLMBatch(
            inputs=jax.ShapeDtypeStruct((batch_size, max_length), jnp.int32),
            targets=jax.ShapeDtypeStruct((batch_size, max_length), jnp.int32),
            inputs_position=jax.ShapeDtypeStruct((batch_size, max_length), jnp.int32),
            inputs_segmentation=jax.ShapeDtypeStruct((batch_size, max_length), jnp.int32),
            targets_position=jax.ShapeDtypeStruct((batch_size, max_length), jnp.int32),
            targets_segmentation=jax.ShapeDtypeStruct((batch_size, max_length), jnp.int32),
        )
