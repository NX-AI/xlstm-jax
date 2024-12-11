xlstm_jax.trainer.llm.sampling
==============================

.. py:module:: xlstm_jax.trainer.llm.sampling


Functions
---------

.. autoapisummary::

   xlstm_jax.trainer.llm.sampling.temperature_sampling
   xlstm_jax.trainer.llm.sampling.greedy_sampling
   xlstm_jax.trainer.llm.sampling.generate_tokens


Module Contents
---------------

.. py:function:: temperature_sampling(logits, rng, temperature = 1.0)

   Temperature sampling for sampling tokens from a logit distribution.

   :param logits: The unnormalized log probabilities, shape (..., vocab_size).
   :param rng: The PRNG key for sampling.
   :param temperature: The temperature for the sampling. A higher temperature will result in more uniform sampling.
                       Defaults to 1.0.

   :returns: The sampled tokens, shape (...).


.. py:function:: greedy_sampling(logits, rng = None)

   Greedy sampling for sampling tokens from a logit distribution.

   :param logits: The unnormalized log probabilities, shape (..., vocab_size).
   :param rng: The PRNG key. This is not used, but is included for compatibility with other sampling functions.

   :returns: The sampled tokens, shape (...).


.. py:function:: generate_tokens(state, rng, prefix_tokens, prefix_mask, cache = None, max_length = 2048, eod_token_id = -1, token_sample_fn = temperature_sampling, gather_params_once = False, data_axis_name = 'dp', fsdp_axis_name = 'fsdp', param_dtype = None)

   Generate tokens from an LLM.

   :param state: The training state, including the parameters and the apply function.
   :param rng: The PRNG key for the random number generator.
   :param prefix_tokens: The prefix tokens for the generation, shape (batch_size, seq len). The sequence length
                         must be at most max_length. The prefix tokens are the tokens that are already known and should not
                         be generated.
   :param prefix_mask: The mask for the prefix tokens. If None, all tokens are considered valid. Defaults to None.
   :param cache: The cache for the model to start from (e.g. recurrent states, kv cache). If None, the cache
                 will be initialized.
   :param max_length: The maximum length of the generated text. Defaults to 2048.
   :param eod_token_id: The end-of-document token id. If all sequences hit this token, generation will stop. Defaults
                        to -1, in which case generation will continue until max_length is reached. Note that if this is set to
                        -1, we perform the generation in a for loop, and otherwise in a while loop, which stops if all sequences
                        have hit the EOD token or the max length is reached.
   :param token_sample_fn: The token sampler to use for sampling tokens. Defaults to temperature sampling with
                           temperature 1.0.
   :param gather_params_once: Whether to gather fsdp-sharded parameters once before generating. This reduces
                              communication overhead between devices, but requires the model to fit on a single device (up to TP
                              parallelism). Defaults to false.
   :param data_axis_name: The data axis name. Defaults to "dp".
   :param fsdp_axis_name: The fsdp axis name. Defaults to "fsdp".
   :param param_dtype: The dtype that the parameters should be converted to before applying the model. For instance,
                       if all operations happen in bfloat16, setting this to bfloat16 converts all parameters once into bfloat16
                       before generating. Defaults to None, in which case the parameters are not converted.

   :returns: The sampled tokens and a mask for valid tokens (true if valid, false otherwise).


