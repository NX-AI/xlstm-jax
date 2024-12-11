xlstm_jax.utils.model_param_handling.store
==========================================

.. py:module:: xlstm_jax.utils.model_param_handling.store


Attributes
----------

.. autoapisummary::

   xlstm_jax.utils.model_param_handling.store.LOGGER


Functions
---------

.. autoapisummary::

   xlstm_jax.utils.model_param_handling.store.store_checkpoint_sharded


Module Contents
---------------

.. py:data:: LOGGER

.. py:function:: store_checkpoint_sharded(state_dict, checkpoint_path, max_shard_size = 1 << 30, metadata = None)

   Save model parameters in sharded fashion into multiple safetensors files.

   :param state_dict: Model state dict.
   :type state_dict: dict[str, torch.Tensor]
   :param checkpoint_path: Checkpoint Path for the model to be stored in.
   :type checkpoint_path: Path
   :param max_shard_size: Maximal shard size in bytes. Defaults to 1<<30.
   :type max_shard_size: int, optional
   :param metadata: Additional metadata for the checkpoint. Defaults to {}.
   :type metadata: dict[str, Any], optional


