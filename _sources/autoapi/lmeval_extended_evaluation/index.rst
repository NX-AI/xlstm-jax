lmeval_extended_evaluation
==========================

.. py:module:: lmeval_extended_evaluation


Attributes
----------

.. autoapisummary::

   lmeval_extended_evaluation.LOGGER


Classes
-------

.. autoapisummary::

   lmeval_extended_evaluation.LMEvalEvaluationConfig
   lmeval_extended_evaluation.LMEvalEvaluation


Functions
---------

.. autoapisummary::

   lmeval_extended_evaluation.log_info
   lmeval_extended_evaluation.fuse_document_results


Module Contents
---------------

.. py:data:: LOGGER

.. py:function:: log_info(msg)

   Logs an info message on the host device.

   :param msg: Message to be logged.


.. py:function:: fuse_document_results(results_dict)

    Fuse log-likelihood results of capped sequences (max_length) to document log-likelihoods.
   Aggregate results from (potentially) many batches of (potentially) many different tasks.
   All results must have matching document indices and aggregate log-likelihoods over multiple
   sequence indices weighted by their counts.

   If the exact sequence is the result of a greedy decoding (i.e. if all single token accuracies of non-masked parts
   are 1), also aggregate "greedy" accuracies.
   :param results_dict: Dictionary of all results in concatenated form.

   :returns: Log-likelihoods and greedy (boolean) Accuracy for documents ordered by index.


.. py:class:: LMEvalEvaluationConfig

   Bases: :py:obj:`xlstm_jax.trainer.callbacks.extended_evaluation.ExtendedEvaluationConfig`


   .. py:attribute:: tokenizer_path
      :type:  str

      Tokenizer path


   .. py:attribute:: evaluation_tasks
      :type:  list[str]

      List of evaluation task from LM Evaluation Harness


   .. py:attribute:: cache_requests
      :type:  bool
      :value: True


      Whether to cache requests


   .. py:attribute:: limit_requests
      :type:  int | None
      :value: None


      Whether to limit requests to a smaller number for debugging purposes


   .. py:attribute:: write_out
      :type:  bool
      :value: False


      Whether to write out results


   .. py:attribute:: use_infinite_eval
      :type:  bool
      :value: True


      Whether to use the infinite eval


   .. py:attribute:: infinite_eval_chunksize
      :type:  int
      :value: 64


      The chunk size for using the infinite eval


   .. py:attribute:: context_length
      :type:  int | None
      :value: None


      Override context_length of the model


   .. py:attribute:: batch_size
      :type:  int | None
      :value: None


      Override batch_size of the trainer


   .. py:attribute:: worker_buffer_size
      :type:  int
      :value: 1


      Worker buffer size for the grain loader


   .. py:attribute:: worker_count
      :type:  int
      :value: 0


      Number of workers for the grain loading


   .. py:attribute:: debug
      :type:  bool
      :value: False


      Scale ouputs such that metrics can be computed for a random testing model


   .. py:attribute:: system_instruction
      :type:  str | None
      :value: None


      Additional system instruction


   .. py:attribute:: num_fewshot
      :type:  int | None
      :value: None


      Define number of in-context samples for few-shot training.


   .. py:attribute:: bootstrap_iters
      :type:  int
      :value: 1000


      Bootstrap iterations for calculating stderrs on metrics - LMEval standard is 100000, limit here for speed


   .. py:attribute:: apply_chat_template
      :type:  bool | str
      :value: False


      Apply the LMEval chat template, or a custom template


   .. py:method:: create(trainer, data_module = None)

      :param trainer: Trainer
      :param data_module: DataloaderModule containing train/val/test - not used here

      :returns: LMEvalEvaluation object



.. py:class:: LMEvalEvaluation(config, trainer, data_module = None)

   Bases: :py:obj:`xlstm_jax.trainer.callbacks.extended_evaluation.ExtendedEvaluation`


   LMEvalEvaluation Callback


   .. py:attribute:: context_length


   .. py:attribute:: batch_size


   .. py:attribute:: lm


   .. py:method:: create_modified_exemplary_batch(exmp_batch)

      Create an LLMIndexedBatch from a LLMBatch (for compilation purposes as example).

      :param exmp_batch:

      :returns: LLMIndexedBatch



   .. py:method:: run_evaluate()

      Runs the evaluation in LM Eval Harness. Does use external datasets.
      Might be called from callback functions to get metrics during training.

      :returns: Results from LMEval evaluation.



   .. py:method:: get_metric_postprocess_fn()
      :staticmethod:


      Get function to post-process metrics with on host.

      Will be passed to logger. Adds perplexity to the metrics.

      :returns: The postprocess metric function.



   .. py:method:: create_jitted_functions()

      Create jitted version of the evaluation function.



   .. py:method:: init_eval_metrics(batch = None)

      Override parent init_eval_metrics potentially for infinite eval.
      Then metrics are partly aggregated one level below (along the sequence) and aggregated fully
      (across batches) within `eval_model`.

      :param batch: An input to the model with which the shapes are inferred. If None, the :attr:`exmp_batch` is used.

      :returns: A dictionary of metrics with the same shape as the eval metrics.



   .. py:method:: aggregate_metrics(aggregated_metrics, eval_metrics)

      Aggregate metrics over multiple batches. This is an adaption of the parent class that ignores the
      passed "step_metrics" in the metrics dictionary. The "step_metrics" are single recurrent step
      metrics to be donated for a future evaluation step.

      :param aggregated_metrics: Old aggregated metrics
      :param eval_metrics: Single batch metrics

      :returns: aggregated_metrics including the new batch



   .. py:method:: create_recurrent_evaluation_step_function(chunk_size, exmp_batch = None, cache_init_fn = None)

      Create and return a recurrent function for the evaluation step. (see also llm/trainer.py).

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



   .. py:method:: finalize_metrics(aggregated_metrics)

      Calculate final metrics from aggregated_metrics. (i,e, mean=sum/count)

      :param aggregated_metrics: Aggregated metrics over the whole epoch

      :returns: Final metrics that are to be reported / logged.



   .. py:method:: eval_function(params, apply_fn, batch, rng = None, mutable_variables = None)

      Function that passes the batch through the model and generates some extended metrics.

      :param params: Model parameters.
      :param apply_fn: Model functions.
      :param batch: LLMIndexedBatch that is passed through the model.
      :param rng: RNG for potential dropout.
      :param mutable_variables: Mutable variables for the evaluation step function, e.g. the cache (recurrent state).

      :returns: Tuple with Metrics and MutableVariables.



   .. py:method:: on_filtered_validation_epoch_start(epoch_idx, step_idx)

      Runs evaluation on filtered validation epochs / steps.

      :param epoch_idx: Epoch Index
      :param step_idx: Step Index



   .. py:method:: on_test_epoch_start(epoch_idx)

      Runs evaluation on test_epoch.

      :param epoch_idx: Epoch index



