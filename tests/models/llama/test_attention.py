import jax.numpy as jnp
import numpy as np
import pytest

from xlstm_jax.models.llama.attention import create_causal_mask


@pytest.mark.parametrize("seqlen", [8, 16])
def test_create_causal_mask(seqlen: int):
    """
    Tests creating causal masks.
    """
    mask = create_causal_mask(seqlen, dtype=jnp.bool_)
    assert mask.shape == (1, 1, seqlen, seqlen)
    mask = mask[0, 0]
    for i in range(seqlen):
        assert np.all(mask[i, : i + 1]), f"Masks was not all True for earlier tokens at i={i}."
        assert np.all(~mask[i, i + 1 :]), f"Masks was not all False for later tokens at i={i}."
