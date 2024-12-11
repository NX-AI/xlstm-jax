xlstm_jax.utils.model_param_handling.load
=========================================

.. py:module:: xlstm_jax.utils.model_param_handling.load


Attributes
----------

.. autoapisummary::

   xlstm_jax.utils.model_param_handling.load.LOGGER


Functions
---------

.. autoapisummary::

   xlstm_jax.utils.model_param_handling.load.load_model_params_and_config_from_checkpoint


Module Contents
---------------

.. py:data:: LOGGER

.. py:function:: load_model_params_and_config_from_checkpoint(checkpoint_path, return_config_as_dataclass = False)

   Load model parameters and config from a jax checkpoint.

   :param checkpoint_path: The path to the checkpoint file.
   :type checkpoint_path: str | Path

   :returns: The model parameters and the model config.
   :rtype: tuple[dict[str, Any], dict[str, Any]]


