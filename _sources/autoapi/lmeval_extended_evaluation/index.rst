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


   Configuration for additional Evaluations callback.



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



   .. py:attribute:: every_n_epochs
      :type:  int
      :value: 1


      If the callback implements functions on a per-epoch basis (e.g. `on_training_epoch_start`,
      `on_training_epoch_end`, `on_validation_epoch_start`), this parameter specifies the frequency of calling these
      functions.


   .. py:attribute:: every_n_steps
      :type:  int
      :value: -1


      If the callback implements functions on a per-step basis (e.g. `on_training_step`), this parameter specifies the
      frequency of calling these functions.


   .. py:attribute:: main_process_only
      :type:  bool
      :value: False


      Whether to call the callback only in the main process.


   .. py:method:: get(key, default=None)


   .. py:method:: to_dict()

      Converts the config to a dictionary.

      Helpful for saving to disk or logging.



   .. py:method:: from_dict(config_class, data, strict_classname_parsing = False, ignore_extensive_attributes = True, none_to_zero_for_ints = False)
      :staticmethod:


      Utility for parsing dictionaries back into a nested dataclass structure, including arbitrary classes and types.

      Currently, this is tailored towards the current logging system with the "hardly" invertible to_dict.

      :param config_class: Typically a dataclass, but can be any other type as well
                           If it is another type, the parser tries to create an object via
                           config_class(**data) if data is a dictionary or config_class(data) else.
      :param data: Typically a dictionary that contains attributes of the dataclass.
                   Can be any other kind of data.
      :param strict_classname_parsing: Parse class names strictly.
      :param ignore_extensive_attributes: Ignore attributes that are not defined in the dataclass.
      :param none_to_zero_for_ints: Convert None to 0 for integer types.

      :returns: An object of type `config_class` that contains the data as attributes.



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



   .. py:attribute:: config


   .. py:attribute:: trainer


   .. py:attribute:: exmp_batch


   .. py:attribute:: eval_step
      :value: None



   .. py:attribute:: _eval_metric_shapes
      :value: None



   .. py:method:: create_evaluation_step_function()

      Create and return a function for the extended evaluation step.

      The function takes as input the training state and a batch from the val/test loader.
      The function is expected to return a dictionary of logging metrics and a new train state.

      :returns: Step function calculating metrics for one batch.



   .. py:method:: eval_model(data_loader, mode = 'test', epoch_idx = 0)

      Evaluate the model on a dataset.

      :param data_loader: Data loader of the dataset to evaluate on.
      :param mode: Whether 'val' or 'test'
      :param epoch_idx: Current epoch index.

      :returns: A dictionary of the evaluation metrics, averaged over data points in the dataset.



   .. py:method:: on_extended_evaluation_start()

      Callback for extended evaluation start.



   .. py:method:: on_extended_evaluation_end(final_metrics)

      Callback for extended evaluation end with final_metrics attached.



   .. py:attribute:: data_module
      :value: None



   .. py:attribute:: _every_n_epochs


   .. py:attribute:: _every_n_steps


   .. py:attribute:: _main_process_only


   .. py:attribute:: _active_on_epochs


   .. py:attribute:: _active_on_steps


   .. py:method:: on_training_start()

      Called at the beginning of training.



   .. py:method:: on_training_end()

      Called at the end of training.



   .. py:method:: on_training_epoch_start(epoch_idx)

      Called at the beginning of each training epoch.

      :param epoch_idx: Index of the current epoch.



   .. py:method:: on_filtered_training_epoch_start(epoch_idx)

      Called at the beginning of each `every_n_epochs` training epoch. To be implemented by subclasses.

      :param epoch_idx: Index of the current epoch.



   .. py:method:: on_training_epoch_end(train_metrics, epoch_idx)

      Called at the end of each training epoch.

      :param train_metrics: Dictionary of training metrics of the current epoch.
      :param epoch_idx: Index of the current epoch.



   .. py:method:: on_filtered_training_epoch_end(train_metrics, epoch_idx)

      Called at the end of each `every_n_epochs` training epoch. To be implemented by subclasses.

      :param train_metrics: Dictionary of training metrics of the current epoch.
      :param epoch_idx: Index of the current epoch.



   .. py:method:: on_training_step(step_metrics, epoch_idx, step_idx)

      Called at the end of each training step.

      :param step_metrics: Dictionary of training metrics of the current step.
      :param epoch_idx: Index of the current epoch.
      :param step_idx: Index of the current step.



   .. py:method:: on_filtered_training_step(step_metrics, epoch_idx, step_idx)

      Called at the end of each `every_n_steps` training step. To be implemented by subclasses.

      :param step_metrics: Dictionary of training metrics of the current step.
      :param epoch_idx: Index of the current epo.
      :param step_idx: Index of the current step.



   .. py:method:: on_validation_epoch_start(epoch_idx, step_idx)

      Called at the beginning of validation.

      :param epoch_idx: Index of the current training epoch.
      :param step_idx: Index of the current training step.



   .. py:method:: on_validation_epoch_end(eval_metrics, epoch_idx, step_idx)

      Called at the end of each validation epoch.

      :param eval_metrics: Dictionary of evaluation metrics of the current epoch.
      :param epoch_idx: Index of the current training epoch.
      :param step_idx: Index of the current training step.



   .. py:method:: on_filtered_validation_epoch_end(eval_metrics, epoch_idx, step_idx)

      Called at the end of each `every_n_epochs` validation epoch. To be implemented by subclasses.

      :param eval_metrics: Dictionary of evaluation metrics of the current epoch.
      :param epoch_idx: Index of the current training epoch.
      :param step_idx: Index of the current training step.



   .. py:method:: on_test_epoch_end(test_metrics, epoch_idx)

      Called at the end of each test epoch. To be implemented by subclasses.

      :param test_metrics: Dictionary of test metrics of the current epoch.
      :param epoch_idx: Index of the current epoch.



   .. py:method:: set_dataset(data_module)

      Sets the data module.

      :param data_module: Data module object.



   .. py:method:: finalize(status = None)

      Called at the end of the whole training process.

      To be implemented by subclasses.

      :param status: Status of the training process.



