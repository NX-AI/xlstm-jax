xlstm_jax.trainer.callbacks.extended_evaluation
===============================================

.. py:module:: xlstm_jax.trainer.callbacks.extended_evaluation


Attributes
----------

.. autoapisummary::

   xlstm_jax.trainer.callbacks.extended_evaluation.LOGGER


Classes
-------

.. autoapisummary::

   xlstm_jax.trainer.callbacks.extended_evaluation.EvalState
   xlstm_jax.trainer.callbacks.extended_evaluation.ExtendedEvaluationConfig
   xlstm_jax.trainer.callbacks.extended_evaluation.ExtendedEvaluation


Functions
---------

.. autoapisummary::

   xlstm_jax.trainer.callbacks.extended_evaluation.device_metrics_aggregation


Module Contents
---------------

.. py:data:: LOGGER

.. py:class:: EvalState

   Bases: :py:obj:`flax.struct.PyTreeNode`


   EvalState with additional mutable variables and RNG.

   :param step: Counter starts at 0 and is incremented by every call to ``.apply_gradients()``.
   :param apply_fn: Usually set to ``model.apply()``. Kept in this dataclass for convenience to have a shorter params
                    list for the ``train_step()`` function in your training loop.
   :param params: The parameters to be updated by ``tx`` and used by ``apply_fn``.


   .. py:attribute:: step
      :type:  int | jax.Array


   .. py:attribute:: rng
      :type:  jax.Array


   .. py:attribute:: apply_fn
      :type:  collections.abc.Callable


   .. py:attribute:: params
      :type:  flax.core.FrozenDict[str, Any]


   .. py:attribute:: mutable_variables
      :type:  Any
      :value: None



   .. py:method:: from_train_state(*, train_state)
      :classmethod:



   .. py:method:: create(*, apply_fn, params, **kwargs)
      :classmethod:



.. py:class:: ExtendedEvaluationConfig

   Bases: :py:obj:`xlstm_jax.trainer.callbacks.callback.CallbackConfig`


   Configuration for additional Evaluations callback.



   .. py:method:: create(trainer, data_module = None)

      Creates an Evaluation callback.

      :param trainer: Trainer object.
      :param data_module: Data module object.



.. py:function:: device_metrics_aggregation(trainer, metrics)

   Aggregates metrics beyond a single scalar value and a count.

   Also include `single_noreduce` metrics by concatenation.

   :param trainer: Trainer (for aggregation axes)
   :param metrics: the sharded metrics

   :returns: The reduced/gathered metrics.


.. py:class:: ExtendedEvaluation(config, trainer, data_module = None)

   Bases: :py:obj:`xlstm_jax.trainer.callbacks.callback.Callback`


   Callback that runs additional evaluations.

   :param config: The configuration for the Evaluation callback.
   :param trainer: Trainer
   :param data_module: :class:`DataloaderModule`, containing train/val/test data loaders.


   .. py:attribute:: config


   .. py:attribute:: trainer


   .. py:attribute:: exmp_batch


   .. py:attribute:: eval_step
      :value: None



   .. py:attribute:: _eval_metric_shapes
      :value: None



   .. py:method:: create_modified_exemplary_batch(exmp_batch)

      Create a modified exemplary batch for evaluation. Is useful for passing additional
      information / metadata to the batch for post-processing.

      :param exmp_batch: "Original" training exemplary batch

      :returns: Modified exemplary batch for evaluation, might be the unmodified original.



   .. py:method:: eval_function(params, apply_fn, batch, rng)

      The extended evaluation function calculating metrics.

      This function needs to be overwritten by a subclass.

      :param params: The model parameters.
      :param apply_fn: The apply function of the state.
      :param batch: The current batch.
      :param rng: The random number generator.

      :returns: A tuple of metrics and mutable variables.



   .. py:method:: create_jitted_functions()

      Create jitted version of the evaluation function.



   .. py:method:: create_evaluation_step_function()

      Create and return a function for the extended evaluation step.

      The function takes as input the training state and a batch from the val/test loader.
      The function is expected to return a dictionary of logging metrics and a new train state.

      :returns: Step function calculating metrics for one batch.



   .. py:method:: init_eval_metrics(batch = None, alternative_eval_step = None)

      Initialize the evaluation metrics with zeros.

      We infer the evaluation metric shape from the eval_step function. This is done to prevent a
      double-compilation of the eval_step function, where the first step has to be done with metrics None,
      and the next one with the metrics shape.

      :param batch: An input to the model with which the shapes are inferred. If None, the :attr:`exmp_batch` is used.
      :param alternative_eval_step: An optional alternative eval step (not self.eval_step). This is needed if
                                    for a more complex eval step, that internally computes multiple steps (i.e. infinite eval in
                                    `lmeval_extended_evaluation.py`)

      :returns: A dictionary of metrics with the same shape as the eval metrics.



   .. py:method:: aggregate_metrics(aggregated_metrics, eval_metrics)

      Aggregate metrics over multiple batches.

      This is needed for "expensive" metrics that go beyond a scalar value and an accumulation count. These are then
      aggregated in CPU memory. The individual batch metrics might already be an actual aggregate for scalar values.


      :param aggregated_metrics: Old aggregated metrics
      :param eval_metrics: Single batch metrics

      :returns: aggregated_metrics including the new batch



   .. py:method:: eval_model(data_loader, mode = 'test', epoch_idx = 0)

      Evaluate the model on a dataset.

      :param data_loader: Data loader of the dataset to evaluate on.
      :param mode: Whether 'val' or 'test'
      :param epoch_idx: Current epoch index.

      :returns: A dictionary of the evaluation metrics, averaged over data points in the dataset.



   .. py:method:: finalize_metrics(aggregated_metrics)

      Calculate final metrics from aggregated_metrics. (i,e, mean=sum/count)

      :param aggregated_metrics: Aggregated metrics over the whole epoch

      :returns: Final metrics that are to be reported / logged.



   .. py:method:: on_extended_evaluation_start()

      Callback for extended evaluation start.



   .. py:method:: on_extended_evaluation_end(final_metrics)

      Callback for extended evaluation end with final_metrics attached.



