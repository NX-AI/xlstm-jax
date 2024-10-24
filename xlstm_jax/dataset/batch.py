import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass


@dataclass
class Batch:
    """Batch of training data."""

    inputs: jax.Array
    """The input data."""
    targets: jax.Array
    """The target data."""

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
    """
    Batch for LLM training.

    Contains inputs and targets along with their respective positions and segmentations.
    """

    inputs_position: jax.Array
    """Positions of the input tokens."""
    inputs_segmentation: jax.Array
    """Segmentation of the input tokens. 0 to indicate padding."""
    targets_position: jax.Array
    """Positions of the target tokens."""
    targets_segmentation: jax.Array
    """Segmentation of the target tokens. 0 to indicate padding."""

    def get_document_borders(self) -> jax.Array:
        """Get the document borders for the input data.

        A token represents a document border if its previous input token has a different input segmentation.
        For instance, if the input segmentation is [1, 1, 2, 2, 2, 3], the document borders are [1, 0, 1, 0, 0, 1].
        This mask can be useful for processing documents separately in a recurrent model, i.e. when to reset the
        hidden state.

        Returns:
            A boolean array indicating the document borders.
        """
        return jnp.pad(
            self.inputs_segmentation[:, :-1] != self.inputs_segmentation[:, 1:],
            ((0, 0), (1, 0)),
            mode="constant",
            constant_values=True,
        )

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


@dataclass
class LLMIndexedBatch(LLMBatch):
    """
    Batch for LLM data with document indices and sequence indices for correct ordering.
    """

    document_idx: jax.Array
    """Document indices for batch sequences"""
    sequence_idx: jax.Array
    """Sequence indices within documents for batch sequences"""

    @staticmethod
    def from_inputs(
        inputs: jax.Array, document_idx: jax.Array, sequence_idx: jax.Array, targets: jax.Array | None = None
    ) -> "LLMIndexedBatch":
        batch = LLMBatch.from_inputs(inputs=inputs, targets=targets)

        return LLMIndexedBatch(
            inputs=batch.inputs,
            targets=batch.targets,
            inputs_position=batch.inputs_position,
            inputs_segmentation=batch.inputs_segmentation,
            targets_position=batch.targets_position,
            targets_segmentation=batch.targets_segmentation,
            document_idx=document_idx,
            sequence_idx=sequence_idx,
        )

    @staticmethod
    def get_dtype_struct(batch_size: int, max_length: int) -> "LLMIndexedBatch":
        """Get the shape and dtype structure for LLMIndexedBatch.

        Args:
            batch_size (int): The size of the batch.
            max_length (int): The maximum length of the sequences.
        """
        return LLMIndexedBatch(
            inputs=jax.ShapeDtypeStruct((batch_size, max_length), jnp.int32),
            targets=jax.ShapeDtypeStruct((batch_size, max_length), jnp.int32),
            inputs_position=jax.ShapeDtypeStruct((batch_size, max_length), jnp.int32),
            inputs_segmentation=jax.ShapeDtypeStruct((batch_size, max_length), jnp.int32),
            targets_position=jax.ShapeDtypeStruct((batch_size, max_length), jnp.int32),
            targets_segmentation=jax.ShapeDtypeStruct((batch_size, max_length), jnp.int32),
            document_idx=jax.ShapeDtypeStruct((batch_size,), jnp.int32),
            sequence_idx=jax.ShapeDtypeStruct((batch_size,), jnp.int32),
        )
