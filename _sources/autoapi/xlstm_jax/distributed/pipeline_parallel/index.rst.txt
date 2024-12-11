xlstm_jax.distributed.pipeline_parallel
=======================================

.. py:module:: xlstm_jax.distributed.pipeline_parallel


Classes
-------

.. autoapisummary::

   xlstm_jax.distributed.pipeline_parallel.PipelineModule


Functions
---------

.. autoapisummary::

   xlstm_jax.distributed.pipeline_parallel.execute_pipeline_step
   xlstm_jax.distributed.pipeline_parallel.execute_pipeline


Module Contents
---------------

.. py:function:: execute_pipeline_step(module, state, input, *args, model_axis_name, **kwargs)

   Single micro-batch pipeline step.

   :param module: Flax module representing the stage to execute.
   :param state: Last communicated features between stages. Used as input to the module for all stages except the first.
   :param input: Original micro-batch input to the pipeline stage. Used as input to the module for the first stage.
   :param \*args: Additional arguments to the module.
   :param model_axis_name: Name of the model axis in the mesh/shard_map.
   :param \*\*kwargs: Additional keyword arguments to the module.

   :returns: Tuple of the new state (after communication) and the output of the module.


.. py:function:: execute_pipeline(module, x, *args, num_microbatches, model_axis_name, **kwargs)

   Execute a pipeline of stages on a batch of data.

   Uses the principle of GPipe in splitting the batch into micro-batches
   and running the pipeline stages in parallel.

   :param module: Flax module representing the pipeline stage to execute.
   :param x: Batch of input data, only needed on device of the first stage. Data will be split into micro-batches.
   :param \*args: Additional arguments to the module.
   :param num_microbatches: Number of micro-batches to split the batch into.
   :param model_axis_name: Name of the model axis in the mesh/shard_map.
   :param \*\*kwargs: Additional keyword arguments to the module.

   :returns: Output of the last stage of the pipeline. For devices that are not
             the last stage, the output is zeros.


.. py:class:: PipelineModule

   Bases: :py:obj:`flax.linen.Module`


   Module wrapper for executing a pipeline of stages.

   This module is used to wrap a stage of a pipeline to execute in pipeline parallelism.

   :param model_axis_name: Name of the model axis in the mesh/shard_map.
   :param num_microbatches: Number of micro-batches to split the batch into.
   :param module_fn: Function that returns the module to execute in the pipeline.


   .. py:attribute:: model_axis_name
      :type:  str


   .. py:attribute:: num_microbatches
      :type:  int


   .. py:attribute:: module_fn
      :type:  collections.abc.Callable[Ellipsis, flax.linen.Module]


