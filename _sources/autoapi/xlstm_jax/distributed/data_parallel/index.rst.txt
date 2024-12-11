xlstm_jax.distributed.data_parallel
===================================

.. py:module:: xlstm_jax.distributed.data_parallel


Functions
---------

.. autoapisummary::

   xlstm_jax.distributed.data_parallel.shard_params
   xlstm_jax.distributed.data_parallel.gather_array_with_mean_grads
   xlstm_jax.distributed.data_parallel.gather_params
   xlstm_jax.distributed.data_parallel.shard_module_params
   xlstm_jax.distributed.data_parallel.sync_gradients


Module Contents
---------------

.. py:function:: shard_params(params, axis_name, min_weight_size = 2**18)

   Shard parameters across the given mesh axis.

   :param params: The parameters to shard.
   :param axis_name: The axis to shard parameters across.
   :param min_weight_size: The minimum size of a parameter to shard. Parameters with fewer values will not be sharded.

   :returns: PyTree of same structure as params, but with leaves sharded over new axis if possible.


.. py:function:: gather_array_with_mean_grads(x, axis, axis_name, gather_dtype = None, grad_scatter_dtype = None)

   Gathering with averaging gradients across replicas.

   :param x: The array to gather.
   :param axis: The axis of the array to gather across.
   :param axis_name: The axis name of the mesh to gather across.
   :param gather_dtype: The dtype to cast the array to before gathering. If None, no casting is performed.
   :param grad_scatter_dtype: The dtype to cast the gradients to before scattering. If None, the dtype of x is used.

   :returns: The gathered array with a gradient function that averages across replicas.


.. py:function:: gather_params(params, axis_name, gather_dtype = None, grad_scatter_dtype = None)

   Gather parameters from all replicas across the given axis.

   :param params: The parameters to gather.
   :param axis_name: The axis to gather parameters across.
   :param gather_dtype: The dtype to cast the parameters to before gathering. If None, no casting is performed.
   :param grad_scatter_dtype: The dtype to cast the gradients to before scattering. If None, the dtype of the parameters
                              is used.

   :returns: PyTree of same structure as params, but with leaves gathered if they were a nn.Partitioned object.


.. py:function:: shard_module_params(target, axis_name, min_weight_size = 2**18, gather_dtype = None, grad_scatter_dtype = None)

   Shard parameters of a module across replicas.

   :param target: The module to shard.
   :param axis_name: The axis name to shard parameters across.
   :param min_weight_size: The minimum size of a parameter to shard. Parameters with fewer values will not be sharded.
   :param gather_dtype: The dtype to cast the parameters to before gathering. If None, no casting is performed.
   :param grad_scatter_dtype: The dtype to cast the gradients to before scattering. If None, the dtype of the parameters
                              is used.

   :returns: The module with sharded parameters.


.. py:function:: sync_gradients(grads, axis_names)

   Synchronize gradients across devices.

   Gradients for parameters that are replicated over a given axis are averaged across devices.
   Parameters that are partitioned over a given axis are considered to already have a mean of
   the gradients on each device, and hence do not need to be altered.

   :param grads: The gradients to synchronize.
   :param axis_names: The axis names to synchronize gradients across.

   :returns: The gradients averaged over the specified axes if they are replicated.


