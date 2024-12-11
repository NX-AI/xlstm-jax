xlstm_jax.models.xlstm_parallel.components.normalization
========================================================

.. py:module:: xlstm_jax.models.xlstm_parallel.components.normalization


Attributes
----------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.components.normalization.LOGGER
   xlstm_jax.models.xlstm_parallel.components.normalization.NormType


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.components.normalization.NormLayer
   xlstm_jax.models.xlstm_parallel.components.normalization.MultiHeadNormLayer
   xlstm_jax.models.xlstm_parallel.components.normalization.resolve_norm


Module Contents
---------------

.. py:data:: LOGGER

.. py:data:: NormType

.. py:function:: NormLayer(weight = True, bias = False, eps = 1e-05, dtype = jnp.float32, norm_type = 'layernorm', model_axis_name = None, **kwargs)

   Create a norm layer.

   :param weight: Whether to use a learnable scaling weight or not.
   :param bias: Whether to use a learnable bias or not.
   :param eps: Epsilon value for numerical stability.
   :param dtype: Data type of the norm. Note that the statistic reductions in the norms are forced to be float32.
   :param norm_type: Type of the norm layer. Currently supported types are "layernorm" and "rmsnorm".
   :param model_axis_name: Name of the model axis to shard over. If None, no sharding is performed.
   :param \*\*kwargs: Additional keyword arguments for the norm layer.

   :returns: Norm layer.


.. py:function:: MultiHeadNormLayer(weight = True, bias = False, eps = 1e-05, dtype = jnp.float32, axis = 1, norm_type = 'layernorm', model_axis_name = None, **kwargs)

   Create a multi-head norm layer.

   Effectively vmaps a norm layer over the specified axis.

   :param weight: Whether to use a learnable scaling weight or not.
   :param bias: Whether to use a learnable bias or not.
   :param eps: Epsilon value for numerical stability.
   :param dtype: Data type of the norm. Note that the statistic reductions in the norms are forced to be float32.
   :param axis: Axis to vmap the norm layer over, i.e. the head axis. The normalization is always performed over
                the last axis.
   :param norm_type: Type of the norm layer. Currently supported types are "layernorm" and "rmsnorm".
   :param model_axis_name: Name of the model axis to shard over. If None, no sharding is performed.
   :param \*\*kwargs: Additional keyword arguments for the norm layer.

   :returns: Multi-head norm layer.


.. py:function:: resolve_norm(norm_type, weight = True, bias = False, eps = 1e-05, dtype = jnp.float32, **kwargs)

   Resolve the norm layer based on the norm type.

   :param norm_type: Type of the norm layer. Currently supported types are "layernorm" and "rmsnorm".
   :param weight: Whether to use a learnable scaling weight or not.
   :param bias: Whether to use a learnable bias or not.
   :param eps: Epsilon value for numerical stability.
   :param dtype: Data type of the norm. Note that the statistic reductions in the norms are forced to be float32.
   :param \*\*kwargs: Additional keyword arguments.

   :returns: Tuple of the norm class and the keyword arguments.


