xlstm_jax.utils.model_param_handling.handle_mlstm_simple
========================================================

.. py:module:: xlstm_jax.utils.model_param_handling.handle_mlstm_simple


Attributes
----------

.. autoapisummary::

   xlstm_jax.utils.model_param_handling.handle_mlstm_simple.LOGGER


Functions
---------

.. autoapisummary::

   xlstm_jax.utils.model_param_handling.handle_mlstm_simple.create_mlstm_simple_config_from_jax_config
   xlstm_jax.utils.model_param_handling.handle_mlstm_simple.apply_mlstm_param_reshapes
   xlstm_jax.utils.model_param_handling.handle_mlstm_simple.move_mlstm_jax_state_dict_into_torch_state_dict
   xlstm_jax.utils.model_param_handling.handle_mlstm_simple.pipeline_convert_mlstm_checkpoint_jax_to_torch_simple
   xlstm_jax.utils.model_param_handling.handle_mlstm_simple.store_mlstm_simple_to_checkpoint
   xlstm_jax.utils.model_param_handling.handle_mlstm_simple.convert_mlstm_checkpoint_jax_to_torch_simple


Module Contents
---------------

.. py:data:: LOGGER

.. py:function:: create_mlstm_simple_config_from_jax_config(model_config_jax, overrides = None)

.. py:function:: apply_mlstm_param_reshapes(state_dict)

.. py:function:: move_mlstm_jax_state_dict_into_torch_state_dict(model_state_dict_torch, model_state_dict_jax_path = None, model_state_dict_jax = None)

   Move the mLSTM jax model state dict into the torch model.

   Either loads the jax model state dict from the model_state_dict_jax_path or uses the provided model_state_dict_jax.

   :param model_torch: The torch model.
   :type model_torch: dict[str, Any]
   :param model_state_dict_jax_path: The path to the jax model state dict. Defaults to None.
   :type model_state_dict_jax_path: Path
   :param model_state_dict_jax: The jax model state dict. Defaults to None.
   :type model_state_dict_jax: dict[str, Any]

   :returns: The torch model with the jax model state dict loaded.
   :rtype: mLSTM


.. py:function:: pipeline_convert_mlstm_checkpoint_jax_to_torch_simple(jax_orbax_model_checkpoint, jax_model_config, torch_model_config_overrides = None)

.. py:function:: store_mlstm_simple_to_checkpoint(mlstm_model, store_torch_model_checkpoint_path, checkpoint_type = 'plain', max_shard_size = 0)

   Stores a mLSTM simple model into a checkpoint directory, using either the
   `huggingface` or `plain` format.

   :param mlstm_model: The mLSTM simple model.
   :type mlstm_model: mLSTM
   :param store_torch_model_checkpoint_path; Torch checkpoint path to store into.:
   :param checkpoint_type: Type of model checkpoint, either 'plain' or 'huggingface'.
   :param max_shard_size: Largest size of a checkpoint model shard. Zero means no sharding.


.. py:function:: convert_mlstm_checkpoint_jax_to_torch_simple(load_jax_model_checkpoint_path, store_torch_model_checkpoint_path, checkpoint_type = 'plain', max_shard_size = 0)

   Convert a jax mLSTM checkpoint to a torch mLSTM checkpoint.

   Loads the jax mLSTM checkpoint, creates a torch mLSTM model, and moves the jax checkpoint parameters into the
   torch model.

   The checkpoint for the torch model is then saved to the store_torch_model_checkpoint_path.

   The torch checkpoint is a directory containing the model params as .safetensors file(s) and a config.yaml file.

   :param load_jax_model_checkpoint_path: Orbax checkpoint path.
   :param store_torch_model_checkpoint_path; Torch checkpoint path to store into.:
   :param checkpoint_type: Type of model checkpoint, either 'plain' or 'huggingface'.
   :param max_shard_size: Largest size of a checkpoint model shard. Zero means no sharding


