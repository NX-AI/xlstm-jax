from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from xlstm_jax.common_types import PyTree, TrainState
from xlstm_jax.distributed import fold_rng_over_axis
from xlstm_jax.distributed.data_parallel import gather_params


def temperature_sampling(logits: jax.Array, rng: jax.Array, temperature: float = 1.0) -> jax.Array:
    """
    Temperature sampling for sampling tokens from a logit distribution.

    Args:
        logits: The unnormalized log probabilities, shape (..., vocab_size).
        rng: The PRNG key for sampling.
        temperature: The temperature for the sampling. A higher temperature will result in more uniform sampling.
            Defaults to 1.0.

    Returns:
        The sampled tokens, shape (...).
    """
    if temperature == 0.0:
        # Temperature 0 is equivalent to greedy sampling.
        return greedy_sampling(logits, rng)
    # Apply temperature
    logits /= temperature
    # Sample from the logits
    return jax.random.categorical(rng, logits, axis=-1)


def greedy_sampling(logits: jax.Array, rng: jax.Array | None = None) -> jax.Array:
    """
    Greedy sampling for sampling tokens from a logit distribution.

    Args:
        logits: The unnormalized log probabilities, shape (..., vocab_size).
        rng: The PRNG key. This is not used, but is included for compatibility with other sampling functions.

    Returns:
        The sampled tokens, shape (...).
    """
    del rng
    return jax.numpy.argmax(logits, axis=-1)


def generate_tokens(
    state: TrainState,
    rng: jax.Array,
    prefix_tokens: jax.Array,
    prefix_mask: jax.Array | None,
    cache: PyTree | None = None,
    max_length: int = 2048,
    eod_token_id: int = -1,
    token_sample_fn: Callable[[jax.Array, jax.Array], jax.Array] = temperature_sampling,
    gather_params_once: bool = False,
    data_axis_name: str = "dp",
    fsdp_axis_name: str = "fsdp",
    param_dtype: jnp.dtype | None = None,
) -> tuple[jax.Array, jax.Array]:
    """
    Generate tokens from an LLM.

    Args:
        state: The training state, including the parameters and the apply function.
        rng: The PRNG key for the random number generator.
        prefix_tokens: The prefix tokens for the generation, shape (batch_size, seq len). The sequence length
            must be at most max_length. The prefix tokens are the tokens that are already known and should not
            be generated.
        prefix_mask: The mask for the prefix tokens. If None, all tokens are considered valid. Defaults to None.
        cache: The cache for the model to start from (e.g. recurrent states, kv cache). If None, the cache
            will be initialized.
        max_length: The maximum length of the generated text. Defaults to 2048.
        eod_token_id: The end-of-document token id. If all sequences hit this token, generation will stop. Defaults
            to -1, in which case generation will continue until max_length is reached. Note that if this is set to
            -1, we perform the generation in a for loop, and otherwise in a while loop, which stops if all sequences
            have hit the EOD token or the max length is reached.
        token_sample_fn: The token sampler to use for sampling tokens. Defaults to temperature sampling with
            temperature 1.0.
        gather_params_once: Whether to gather fsdp-sharded parameters once before generating. This reduces
            communication overhead between devices, but requires the model to fit on a single device (up to TP
            parallelism). Defaults to false.
        data_axis_name: The data axis name. Defaults to "dp".
        fsdp_axis_name: The fsdp axis name. Defaults to "fsdp".
        param_dtype: The dtype that the parameters should be converted to before applying the model. For instance,
            if all operations happen in bfloat16, setting this to bfloat16 converts all parameters once into bfloat16
            before generating. Defaults to None, in which case the parameters are not converted.

    Returns:
        The sampled tokens and a mask for valid tokens (true if valid, false otherwise).
    """
    batch_size = prefix_tokens.shape[0]

    # Prepare parameters.
    params = state.params
    if param_dtype is not None:
        params = jax.tree_map(lambda x: x.astype(param_dtype), params)

    # Prepare prefix tokens and mask.
    if prefix_tokens.ndim == 1:
        prefix_tokens = prefix_tokens[:, None]
    if prefix_mask is None:
        prefix_mask = jnp.ones_like(prefix_tokens, dtype=bool)
    else:
        # Mask out invalid tokens.
        prefix_tokens = prefix_tokens * prefix_mask

    # Different RNG over data axes, same over model axes.
    rng = fold_rng_over_axis(rng, (data_axis_name, fsdp_axis_name))

    # Gather params if needed.
    if gather_params_once:
        params = gather_params(params, axis_name=fsdp_axis_name)

    # Single apply of model with support for caching.
    @jax.named_scope("apply_fn")
    def _apply_fn(inputs: jax.Array, cache: PyTree | None) -> tuple[jax.Array, PyTree]:
        variables = {"params": params}
        if cache is not None:
            variables["cache"] = cache
        logits, mutable_variables = state.apply_fn(
            variables,
            inputs,
            document_borders=None,
            train=False,
            mutable="cache",
        )
        if "cache" in mutable_variables:
            cache = mutable_variables["cache"]
        return logits, cache

    if cache is None:
        # Initialize cache.
        _, cache = _apply_fn(prefix_tokens, cache)

    @jax.named_scope("generate_step")
    def _generate_one_step(loop_state: dict[str, Any]) -> dict[str, Any]:
        # Function for performing a single generation step, ie generate one token per sequence.
        # We use a loop state to keep track of the current state of the generation. The loop state
        # is a dictionary with the following keys:
        # - idx: The current index in the generation process. We start counting at 0, i.e. the first token we generate
        #        is at index 0, and will be stored at index 1 in the tokens buffer (behind first input token).
        idx = loop_state["idx"]
        # - rng: The PRNG key, we split it to get a new key for the next step.
        step_rng, loop_state["rng"] = jax.random.split(loop_state["rng"])
        # - cache: The cache for the model.
        cache = loop_state["cache"]
        # - last_is_valid: A boolean mask indicating whether the last token was valid. This is used to stop a sequence
        #                  if it has hit the EOD token. This could also be solved over the is_valid and token mask, but
        #                  this is more efficient, offers easy access in the loop condition, and catches easier corner
        #                  cases (e.g. first token, etc).
        last_is_valid = loop_state["last_is_valid"]
        # - tokens: The generated tokens so far. Is a buffer for the full sequence, and contains the prefix if set.
        #           We load the last token to generate the next token.
        last_token = jax.lax.dynamic_index_in_dim(loop_state["tokens"], idx, axis=1)
        # - is_valid: A boolean mask indicating whether a token is valid. By default, all future tokens are invalid
        #             to indicate that their location is a buffer. If a future token has been set by the prefix, it
        #             is set to True. We run the sampling function on all tokens, but ignore the generated tokens
        #             for the prefix tokens.
        is_prefix = jax.lax.dynamic_index_in_dim(loop_state["is_valid"], idx + 1, axis=1)
        prefix_token = jax.lax.dynamic_index_in_dim(loop_state["tokens"], idx + 1, axis=1)

        # Sample next token.
        logits, cache = _apply_fn(last_token, cache)
        with jax.named_scope("token_sample_fn"):
            next_token = token_sample_fn(logits, step_rng)

        # Overwrite prefix tokens.
        next_token = jnp.where(is_prefix, prefix_token, next_token)

        # Update loop state.
        # - idx: Increment the index.
        loop_state["idx"] = idx + 1
        # - cache: Update the cache.
        loop_state["cache"] = cache
        # - last_is_valid: Update the last_is_valid flag. If the next token is the EOD token and has been not a prefix
        #                  token, the sequence is stopped and set to invalid.
        seq_reached_eod = jnp.logical_and((next_token == eod_token_id), ~is_prefix).squeeze(-1)
        next_is_valid = jnp.logical_and(~seq_reached_eod, last_is_valid)
        loop_state["last_is_valid"] = next_is_valid
        # - tokens: Update the tokens buffer with the newest token.
        loop_state["tokens"] = jax.lax.dynamic_update_index_in_dim(loop_state["tokens"], next_token, idx + 1, axis=1)
        # - is_valid: Update the is_valid mask. If the last token was valid, the token in this generated step is valid.
        #             If we generated an EOD token in this step, we already set the next_is_valid to False, but still
        #             need to set the current token to valid. If the last token was invalid, the current token is
        #             invalid.
        loop_state["is_valid"] = jax.lax.dynamic_update_index_in_dim(
            loop_state["is_valid"], last_is_valid, idx + 1, axis=1
        )

        return loop_state

    def _loop_cond(loop_state: dict[str, Any]) -> jax.Array:
        # Stop if all sequences have hit the EOD token or the max length is reached.
        return jnp.logical_and(jnp.any(loop_state["last_is_valid"]), loop_state["idx"] < max_length - 1)

    # Initialize loop state.
    tokens = jnp.zeros((batch_size, max_length), dtype=prefix_tokens.dtype)
    tokens = jax.lax.dynamic_update_slice_in_dim(tokens, prefix_tokens, 0, axis=1)
    start_is_valid = prefix_mask[:, 0]  # If first token is invalid, the whole sequence is invalid.
    is_valid = jnp.zeros((batch_size, max_length), dtype=bool)
    is_valid = jax.lax.dynamic_update_index_in_dim(is_valid, prefix_mask, 0, axis=1)
    loop_state = {
        "idx": 0,
        "rng": rng,
        "cache": cache,
        "last_is_valid": start_is_valid,
        "tokens": tokens,
        "is_valid": is_valid,
    }

    # Run loop.
    # If we have given an EOD token, we perform early stopping by running a while loop to generate tokens.
    # Otherwise, we perform a for loop to generate tokens.
    if eod_token_id >= 0:
        loop_state = jax.lax.while_loop(_loop_cond, _generate_one_step, loop_state)
    else:
        loop_state = jax.lax.fori_loop(0, max_length - 1, lambda i, val: _generate_one_step(val), loop_state)

    # Final outputs.
    tokens = loop_state["tokens"]
    is_valid = loop_state["is_valid"]

    # Mask out invalid tokens.
    tokens = tokens * is_valid

    return tokens, is_valid
