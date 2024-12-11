xlstm_jax.trainer.metrics
=========================

.. py:module:: xlstm_jax.trainer.metrics


Attributes
----------

.. autoapisummary::

   xlstm_jax.trainer.metrics.LOGGER


Functions
---------

.. autoapisummary::

   xlstm_jax.trainer.metrics.update_metrics
   xlstm_jax.trainer.metrics.aggregate_metrics
   xlstm_jax.trainer.metrics._empty_val
   xlstm_jax.trainer.metrics._update_single_metric
   xlstm_jax.trainer.metrics.get_metrics


Module Contents
---------------

.. py:data:: LOGGER

.. py:function:: update_metrics(global_metrics, step_metrics, default_log_modes = None)

   Update metrics with new values.

   :param global_metrics: Global metrics to update. If `None`, a new dictionary is created.
   :param step_metrics: Metrics to update with.
   :param default_log_modes: The default log mode for the metrics. If `None`, only the mean will be logged. Otherwise, we
                             log each of the modes specified. The metric key will be appended with the log mode.

   :returns: Updated global metrics.


.. py:function:: aggregate_metrics(aggregated_metrics, batch_metrics)

   This function aggregates multiple metrics for the `single_noreduce` and `single_noreduce_wcount` case.
   For `single_noreduce` batches of single values are concatenated. The count is the number of samples.
   For `single_noreduce_wcount` batches of values are concatenated, as well as counts for each sample.
   This is needed for e.g. the loglikelihood per sequence.
   Concatenation happens in CPU memory after a conversion.

   The function returns batch_metrics in all other cases and moves them to CPU memory.

   :param aggregated_metrics: Previously aggregated metrics to append to potentially.
   :param batch_metrics: Metrics from a batch

   :returns: Newly aggregated metrics.


.. py:function:: _empty_val(value)

.. py:function:: _update_single_metric(global_metrics, key, value, count, log_modes = None)

   Update a single metric.

   :param global_metrics: Global metrics to update.
   :param key: Key of the metric to update.
   :param value: Value of the metric to update.
   :param count: Count of the metric to update.
   :param log_modes: The log modes for the metric.


.. py:function:: get_metrics(global_metrics, reset_metrics = True)

   Calculates metrics to log from global metrics.

   Supports resetting the global metrics after logging. For example, if the global metrics are logged every epoch, the
   global metrics can be reset after obtaining the metrics to log such that the next epoch starts with empty metrics.

   :param global_metrics: Global metrics to log.
   :param reset_metrics: Whether to reset the metrics after logging.

   :returns: The updated global metrics if reset_metrics is `True`, otherwise the original global metrics.
             Additionally, the metrics to log on the host device are returned.


