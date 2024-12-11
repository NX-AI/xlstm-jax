xlstm_jax.distributed.array_utils
=================================

.. py:module:: xlstm_jax.distributed.array_utils


Functions
---------

.. autoapisummary::

   xlstm_jax.distributed.array_utils.fold_rng_over_axis
   xlstm_jax.distributed.array_utils.split_array_over_mesh
   xlstm_jax.distributed.array_utils.stack_params
   xlstm_jax.distributed.array_utils.unstack_params


Module Contents
---------------

.. py:function:: fold_rng_over_axis(rng, axis_name)

   Folds the random number generator over the given axis.

   This is useful for generating a different random number for each device
   across a certain axis (e.g. the model axis).

   :param rng: The random number generator.
   :param axis_name: The axis name to fold the random number generator over.

   :returns: A new random number generator, different for each device index along the axis.


.. py:function:: split_array_over_mesh(x, axis_name, split_axis)

   Split an array over the given mesh axis.

   :param x: The array to split.
   :param axis_name: The axis name of the mesh to split over.
   :param split_axis: The axis of the array to split.

   :returns: The slice of the array for the current device along the given axis.


.. py:function:: stack_params(params, axis_name, axis = 0, mask_except = None)

   Stacks sharded parameters along a given axis name.

   :param params: PyTree of parameters.
   :param axis_name: Name of the axis to stack along.
   :param axis: Index of the axis to stack along.
   :param mask_except: If not None, only the `mask_except`-th shard will be non-zero.

   :returns: PyTree of parameters with the same structure as `params`, but with the leaf
             nodes replaced by `nn.Partitioned` objects with sharding over axis name added
             to `axis`-th axis of parameters.


.. py:function:: unstack_params(params, axis_name)

   Unstacks parameters along a given axis name.

   Inverse operation to `stack_params`.

   :param params: PyTree of parameters.
   :param axis_name: Name of the axis to unstack along.

   :returns: PyTree of parameters with the same structure as `params`, but
             with the leaf nodes having the sharding over the axis name removed.


