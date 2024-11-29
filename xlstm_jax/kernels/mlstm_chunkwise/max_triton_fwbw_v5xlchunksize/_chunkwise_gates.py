"""In this file we compute the chunkwise or cumulative gates (i.e. vecA and vecB)
for the forward and backward pass of the mLSTM.
We use the stable formulations, i.e. we avoid subtraction of forget gates.
"""

import jax
import jax.numpy as jnp

# TODO write a triton kernel for this


def compute_chunkwise_log_gates_vecB_vecA(
    vecI: jax.Array,  # (B, NH, S)
    vecF: jax.Array,  # (B, NH, S)
    chunk_size: int,
    return_vecB_only: bool = False,
):
    B, NH, S = vecI.shape
    assert S % chunk_size == 0, f"S={S} is not divisible by chunk_size={chunk_size}"
    NC = S // chunk_size
    L = chunk_size

    # compute vecB
    vecF_logsig: jax.Array = jax.nn.log_sigmoid(vecF.astype(jnp.float32))
    vecF_logsig_chunked = vecF_logsig.reshape(B, NH, NC, L)
    vecB = vecF_logsig_chunked.cumsum(axis=-1)

    if return_vecB_only:
        return vecB

    # compute vecA
    vecI_chunked = vecI.reshape(B, NH, NC, L)
    # unstable vecA computation:
    # vecA = (vecB[..., -1, None] - vecB) + vecI  # (B, NH, NC, L)
    # stable vecA computation:
    vecF_cumsum = jnp.flip(jnp.flip(vecF_logsig_chunked[..., 1:], axis=-1).cumsum(-1), axis=-1)
    vecA = (
        jnp.concat(
            [
                vecF_cumsum,
                jnp.zeros((B, NH, NC, 1), dtype=jnp.float32),
            ],
            axis=-1,
        )
        + vecI_chunked
    )  # (B, NH, NC, L)
    return vecB, vecA
