import jax
import jax.numpy as jnp
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


def test_llm_batch_sample():
    """
    Test an LLMBatch for sampling
    """
    batch_size, context_length = 4, 8
    sample = LLMBatch.get_sample(batch_size=batch_size, max_length=context_length)
    sample_shapes = LLMBatch.get_dtype_struct(batch_size=batch_size, max_length=context_length)

    for sample_leaf, sample_shape_leaf in zip(jax.tree.leaves(sample), jax.tree.leaves(sample_shapes)):
        assert isinstance(sample_leaf, jax.Array)
        assert isinstance(sample_shape_leaf, jax.ShapeDtypeStruct)
        assert sample_leaf.shape == sample_shape_leaf.shape


def test_llm_indexed_batch_sample():
    """
    Test an LLMBatch for sampling
    """
    batch_size, context_length = 4, 8
    sample = LLMIndexedBatch.get_sample(batch_size=batch_size, max_length=context_length)
    sample_shapes = LLMIndexedBatch.get_dtype_struct(batch_size=batch_size, max_length=context_length)

    for sample_leaf, sample_shape_leaf in zip(jax.tree.leaves(sample), jax.tree.leaves(sample_shapes)):
        assert isinstance(sample_leaf, jax.Array)
        assert isinstance(sample_shape_leaf, jax.ShapeDtypeStruct)
        assert sample_leaf.shape == sample_shape_leaf.shape
