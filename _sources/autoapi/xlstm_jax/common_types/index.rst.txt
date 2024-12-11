xlstm_jax.common_types
======================

.. py:module:: xlstm_jax.common_types


Attributes
----------

.. autoapisummary::

   xlstm_jax.common_types.PyTree
   xlstm_jax.common_types.Parameter
   xlstm_jax.common_types.PRNGKeyArray
   xlstm_jax.common_types.LogMode
   xlstm_jax.common_types.ImmutableMetricElement
   xlstm_jax.common_types.ImmutableMetrics
   xlstm_jax.common_types.MutableMetricElement
   xlstm_jax.common_types.MutableMetrics
   xlstm_jax.common_types.StepMetricsElement
   xlstm_jax.common_types.StepMetrics
   xlstm_jax.common_types.MetricElement
   xlstm_jax.common_types.Metrics
   xlstm_jax.common_types.HostMetricElement
   xlstm_jax.common_types.HostMetrics


Classes
-------

.. autoapisummary::

   xlstm_jax.common_types.TrainState


Module Contents
---------------

.. py:data:: PyTree

.. py:data:: Parameter

.. py:data:: PRNGKeyArray

.. py:data:: LogMode

   Mode for logging. Describes how to aggregate metrics over steps.

   - `mean`: Mean of the metric.
   - `mean_nopostfix`: Mean of the metric without adding a mean postfix to the key.
   - `single`: Single value of the metric, i.e. only tracks the last value.
   - `max`: Maximum value of the metric.
   - `std`: Standard deviation of the metric.
   - `single_noreduce`: Concatenate the metrics of multiple values.
   - `single_noreduce_wcount`: Concatenate the metrics and counts of multiple values.

.. py:data:: ImmutableMetricElement

.. py:data:: ImmutableMetrics

.. py:data:: MutableMetricElement

.. py:data:: MutableMetrics

.. py:data:: StepMetricsElement

.. py:data:: StepMetrics

.. py:data:: MetricElement

.. py:data:: Metrics

.. py:data:: HostMetricElement

.. py:data:: HostMetrics

.. py:class:: TrainState

   Bases: :py:obj:`flax.training.train_state.TrainState`


   TrainState with additional mutable variables and RNG.


   .. py:attribute:: mutable_variables
      :type:  Any
      :value: None



   .. py:attribute:: rng
      :type:  PRNGKeyArray | None
      :value: None



