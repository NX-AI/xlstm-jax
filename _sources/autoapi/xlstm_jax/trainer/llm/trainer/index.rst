xlstm_jax.trainer.llm.trainer
=============================

.. py:module:: xlstm_jax.trainer.llm.trainer


Classes
-------

.. autoapisummary::

   xlstm_jax.trainer.llm.trainer.LLMTrainerConfig
   xlstm_jax.trainer.llm.trainer.LLMTrainer


Module Contents
---------------

.. py:class:: LLMTrainerConfig

   Bases: :py:obj:`xlstm_jax.trainer.base.trainer.TrainerConfig`


   Configuration for the LLM trainer.

   See TrainerConfig for inherited attributes.

   .. attribute:: log_logit_stats

      Whether to log statistics of the logits during training. Note that for large vocabularies, this
      can be expensive.


   .. py:attribute:: log_logit_stats
      :type:  bool
      :value: False



.. py:class:: LLMTrainer

   Bases: :py:obj:`xlstm_jax.trainer.base.trainer.TrainerModule`


   Trainer for Autoregressive Language Models.


   .. py:method:: loss_function(params, apply_fn, batch, rng, train = True, mutable_variables = None)

      Calculate the loss function for the model.

      :param params: The model parameters.
      :param apply_fn: The model apply function.
      :param batch: The input batch.
      :param rng: The random number generator.
      :param train: Whether the model is in training mode. Defaults to True.
      :param mutable_variables: Additional mutable variables which are passed to the model apply function and set to
                                mutable. Defaults to None.

      :returns: The loss and a tuple of metrics and mutable variables.



   .. py:method:: get_metric_postprocess_fn()

      Get function to post-process metrics with on host.

      Will be passed to logger. Adds perplexity to the metrics.

      :returns: The postprocess metric function.



   .. py:method:: get_generate_fn(max_length = 2048, eod_token_id = -1, token_sample_fn = temperature_sampling, gather_params_once = False, param_dtype = None)

      Create a function to generate text from the model.

      :param max_length: The maximum length of the generated text. Defaults to 2048.

      eod_token_id: The end-of-document token id. If all sequences hit this token, generation will stop. Defaults
          to -1, in which case generation will continue until max_length is reached. Note that if this is set to
          -1, we perform the generation in a for loop, and otherwise in a while loop.
          token_sample_fn: The token sampler to use for sampling tokens. Defaults to temperature sampling with
              temperature 1.0.
          gather_params_once: Whether to gather fsdp-sharded parameters once before generating. This reduces
              communication overhead between devices, but requires the model to fit on a single device (up to TP
              parallelism). Defaults to false.
          param_dtype: The dtype that the parameters should be converted to before applying the model. For instance,
              if all operations happen in bfloat16, setting this to bfloat16 converts all parameters once into
              bfloat16 before generating. Defaults to None, in which case the parameters are not converted.

      :returns: The generate function. Takes as input the state, an RNG key, prefix tokens, prefix mask, and an optional
                cache. It returns the generated tokens and a mask for valid tokens, including the prefix given.



   .. py:method:: get_cache_shape_dtype_struct(exmp_batch = None)

      Get the shape, dtype, and structure of the cache.

      :param exmp_batch: An example batch to determine the shape of the cache. Defaults to None, in which case the
                         example batch from the trainer is used.

      :returns: A PyTree with leafs being ShapeDtypeStruct of the cache elements.



   .. py:method:: create_recurrent_evaluation_step_function(chunk_size, exmp_batch = None, cache_init_fn = None)

      Create and return a function for the evaluation step.

      Compared to the `create_evaluation_step_function`, this evaluation supports much longer sequences by chunking
      the input and running the model recurrently over the chunks. This is useful for evaluation on long documents.
      This is enabled by keeping a cache, which is forwarded between evaluation steps.

      Note: do *not* jit this function if you want to support arbitrary input shapes. This function jit's the
      recurrent function for a single chunk, and adds a python loop around it to handle arbitrary length sequences.
      Thus, no outer jit is needed.

      Note: this function is explicitly meant for recurrent models like xLSTM. Using this function on a non-recurrent
      model will lead to unexpected, incorrect results.

      :param chunk_size: Size of the chunks to split the input into. The slices are performed over the sequence length.
      :param exmp_batch: An example batch to determine the shape of the cache. Defaults to None, in which case the
                         example batch from the trainer is used.
      :param cache_init_fn: A function to initialize the cache. If not provided, the cache is initialized with zeros.
                            The function should take the shape dtype struct of the cache as input and return the initialized cache.

      :returns: The evaluation step function with support for arbitrary length sequences.



