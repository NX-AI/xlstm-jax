#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xlstm_jax.dataset.batch import LLMBatch, LLMIndexedBatch


@pytest.mark.parametrize("batch_size", [4, 16])
@pytest.mark.parametrize("context_length", [32, 128])
def test_llm_batch_document_borders(batch_size: int, context_length: int):
    """
    Test the document borders function from the LLMBatch class.

    Args:
        batch_size: The size of the batch.
        context_length: The length of the context.
    """
    rng = jax.random.PRNGKey(batch_size * context_length)
    eod_token_id = 0
    inputs = jax.random.randint(rng, (batch_size, context_length), 0, 10)
    eod_mask = inputs == eod_token_id
    inputs_segmentation = 1 + jnp.cumsum(eod_mask, axis=1)
    inputs_position = jnp.broadcast_to(jnp.arange(context_length), inputs.shape)
    # TODO: this test does not shift inputs or targets. Therefore, the document borders might be not computed correctly.
    batch = LLMBatch(
        inputs=inputs,
        targets=inputs,
        inputs_position=inputs_position,
        targets_position=inputs_position,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=inputs_segmentation,
    )
    document_borders = batch.get_document_borders()
    assert document_borders.shape == (
        batch_size,
        context_length,
    ), "The document borders should have the same shape as the input."
    assert jnp.all(document_borders[:, 0]), "The first token should always be a document border."
    assert jnp.all(document_borders[:, 1:] == eod_mask[:, 1:]), "The document borders should match the EOD mask."


@pytest.mark.parametrize("batch_class", [LLMBatch, LLMIndexedBatch])
def test_llm_batch_sample(batch_class: type):
    """
    Test an LLMBatch for sampling
    """
    batch_size, context_length = 4, 8
    sample = batch_class.get_sample(batch_size=batch_size, max_length=context_length)
    sample_shapes = batch_class.get_dtype_struct(batch_size=batch_size, max_length=context_length)

    for sample_leaf, sample_shape_leaf in zip(jax.tree.leaves(sample), jax.tree.leaves(sample_shapes)):
        assert isinstance(sample_leaf, jax.Array)
        assert isinstance(sample_shape_leaf, jax.ShapeDtypeStruct)
        assert sample_leaf.shape == sample_shape_leaf.shape


@pytest.mark.parametrize("context_length", [16, 2048])
@pytest.mark.parametrize("use_indexed_batch", [False, True])
def test_llm_batch_slice(context_length: int, use_indexed_batch: bool):
    """
    Test slicing an LLMBatch and LLMIndexedBatch.
    """
    batch_size = 32
    rngs = jax.random.split(jax.random.PRNGKey(0), 5)
    # Simulate multiple documents.
    doc_borders = jax.random.bernoulli(rngs[0], 0.10, (batch_size, context_length))
    inputs_segmentation = jnp.cumsum(doc_borders, axis=1) + 1
    inputs_positions = np.zeros_like(inputs_segmentation)
    for i in range(1, context_length):
        inputs_positions[:, i] = np.where(doc_borders[:, i], 0, inputs_positions[:, i - 1] + 1)
    inputs_positions = jnp.array(inputs_positions)
    inputs = jax.random.randint(rngs[1], (batch_size, context_length), 0, 10)
    targets = jnp.roll(inputs, shift=1, axis=1)
    # Masking inputs to simulate padding.
    input_lengths = jax.random.randint(rngs[2], (batch_size, 1), 1, context_length)
    inputs_mask = inputs_positions < input_lengths
    inputs = jnp.where(inputs_mask, inputs, 0)
    targets = jnp.where(inputs_mask, targets, 0)
    inputs_segmentation = jnp.where(inputs_mask, inputs_segmentation, 0)
    inputs_positions = jnp.where(inputs_mask, inputs_positions, 0)
    # Create the batch.
    if use_indexed_batch:
        document_idx = jax.random.randint(rngs[3], (batch_size,), 0, 10_000)
        sequence_idx = jax.random.randint(rngs[4], (batch_size,), 0, 10_000)
        batch = LLMIndexedBatch(
            inputs=inputs,
            targets=targets,
            inputs_position=inputs_positions,
            inputs_segmentation=inputs_segmentation,
            targets_position=inputs_positions,
            targets_segmentation=inputs_segmentation,
            document_idx=document_idx,
            sequence_idx=sequence_idx,
        )
    else:
        batch = LLMBatch(
            inputs=inputs,
            targets=targets,
            inputs_position=inputs_positions,
            inputs_segmentation=inputs_segmentation,
            targets_position=inputs_positions,
            targets_segmentation=inputs_segmentation,
        )

    def check_slice(original_batch, sliced_batch, slice_start, slice_end, slice_size):
        # Check the slice.
        prefix = (
            f"[Slice {slice_start}:{slice_end}, Slice size {slice_size}, Context length {context_length}, "
            f"Batch size {batch_size}, Use indexed batch {use_indexed_batch}]"
        )
        attributes = [
            "inputs",
            "targets",
            "inputs_position",
            "inputs_segmentation",
            "targets_position",
            "targets_segmentation",
        ]
        if use_indexed_batch:
            attributes.extend(["document_idx", "sequence_idx"])
        for attr in attributes:
            assert hasattr(sliced_batch, attr), f"{prefix} Missing attribute {attr}."
            sliced_val = getattr(sliced_batch, attr)
            original_val = getattr(original_batch, attr)
            original_val = original_val[:, slice_start:slice_end] if original_val.ndim > 1 else original_val
            assert sliced_val.shape == original_val.shape, f"{prefix} {attr} has incorrect shape."
            assert sliced_val.dtype == original_val.dtype, f"{prefix} {attr} has incorrect dtype."
            # Check the values.
            np.testing.assert_array_equal(sliced_val, original_val, err_msg=f"{prefix} {attr} incorrectly sliced.")
        # Check document borders.
        sliced_doc_borders = sliced_batch.get_document_borders()
        original_doc_borders = original_batch.get_document_borders()
        np.testing.assert_array_equal(
            sliced_doc_borders,
            original_doc_borders[:, slice_start:slice_end],
            err_msg=f"{prefix} Document borders incorrectly sliced.",
        )

    # Slice the batch.
    for slice_size in [context_length // 16, context_length // 5, context_length]:
        for slice_start in range(0, context_length, slice_size):
            # Check the slice.
            slice_end = min(slice_start + slice_size, context_length)
            sliced_batch = batch[:, slice_start:slice_end]
            check_slice(batch, sliced_batch, slice_start, slice_end, slice_size)
            # Check the slice of the slice.
            new_slice_start = slice_size // 4
            new_slice_end = slice_size // 4 * 3
            check_slice(
                sliced_batch,
                sliced_batch[:, new_slice_start:new_slice_end],
                new_slice_start,
                new_slice_end,
                new_slice_end - new_slice_start,
            )
