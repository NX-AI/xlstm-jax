xlstm_jax.utils.model_param_handling.convert_state_dict
=======================================================

.. py:module:: xlstm_jax.utils.model_param_handling.convert_state_dict


Attributes
----------

.. autoapisummary::

   xlstm_jax.utils.model_param_handling.convert_state_dict.LOGGER


Functions
---------

.. autoapisummary::

   xlstm_jax.utils.model_param_handling.convert_state_dict.find_parameter_match_key
   xlstm_jax.utils.model_param_handling.convert_state_dict.create_full_state_dict_key_mapping
   xlstm_jax.utils.model_param_handling.convert_state_dict.move_state_dict_params_
   xlstm_jax.utils.model_param_handling.convert_state_dict.move_safetensors_state_dict_params_
   xlstm_jax.utils.model_param_handling.convert_state_dict.convert_state_dict_keys_
   xlstm_jax.utils.model_param_handling.convert_state_dict.apply_weight_transforms_


Module Contents
---------------

.. py:data:: LOGGER

.. py:function:: find_parameter_match_key(from_key, to_keys, match_dict)

   Finds the matching parameter key for the target state dict.

   :param from_key: The key of the source state dict.
   :type from_key: str
   :param to_keys: The keys of the target state dict.
   :type to_keys: list[str]
   :param match_dict: The dict that maps the source state dict keys to the target state dict keys.
                      Should contain unique substrings of the source state dict keys as keys and the
                      corresponding target state dict keys substrings as values.
   :type match_dict: dict[str, str]

   :returns: The target state dict key that matches the source state dict key.


.. py:function:: create_full_state_dict_key_mapping(from_state_dict, to_state_dict, match_dict)

   Creates a full state dict key mapping from the source state dict to the target state dict.

   :param from_state_dict: The source state dict.
   :type from_state_dict: dict
   :param to_state_dict: The target state dict.
   :type to_state_dict: dict
   :param match_dict: The dict that maps the source state dict keys to the target state dict keys.
   :type match_dict: dict

   :returns: The full state dict key mapping from the source state dict to the target state dict.
   :rtype: dict


.. py:function:: move_state_dict_params_(from_state_dict, to_state_dict, match_dict)

   Move the params of a model from one state dict to another state dict.
   Modifies the to_state_dict in place.

   :param from_state_dict: The source state dict.
   :type from_state_dict: dict[str, Any]
   :param to_state_dict: The target state dict.
   :type to_state_dict: dict[str, Any]
   :param match_dict: The dict that maps the source state dict keys to the target state dict keys.
                      Should contain unique substrings of the source state dict keys as keys and the
                      corresponding target state dict keys substrings as values.
   :type match_dict: dict[str, str]

   :returns: The target (modified to_state_dict) state dict with the converted parameters.
   :rtype: dict[str, Any]


.. py:function:: move_safetensors_state_dict_params_(from_state_dict_path, to_state_dict, match_dict)

   Move the params of a model from one state dict to another state dict.
   It loads the from_state_dict from a file on-the-fly. This means only the to_state_dict is in memory.
   Modifies the to_state_dict in place.

   :param from_state_dict: The path to the source state dict.
   :type from_state_dict: Path
   :param to_state_dict: The target state dict.
   :type to_state_dict: dict[str, Any]
   :param match_dict: The dict that maps the source state dict keys to the target state dict keys.
                      Should contain unique substrings of the source state dict keys as keys and the
                      corresponding target state dict keys substrings as values.
   :type match_dict: dict[str, str]

   :returns: The target (modified to_state_dict) state dict with the converted parameters.
   :rtype: dict[str, Any]


.. py:function:: convert_state_dict_keys_(state_dict, full_key_mapping)

   Converts the keys of the state dict according to the key mapping.

   :param state_dict: The state dict to convert.
   :type state_dict: dict[str, Any]
   :param full_key_mapping: The key mapping that maps the old keys to the new keys.
   :type full_key_mapping: dict[str, str]

   :returns: The converted state dict with the new keys.
   :rtype: dict[str, Any]


.. py:function:: apply_weight_transforms_(state_dict, apply_transforms_to_keys)

   Applies weight transforms to the weights of the state dict.

   There are currently these transforms supported:
       - "transpose": Transposes the weight tensor. Accepts only 2D tensors.
       - "squeeze-XXX": Squeezes the XXX dimension of the weight tensor.
          If XXX is not given, squeezes all dimensions of size 1.
       - "flatten": Flattens the weight tensor.

   If possible the transforms are applied in-place on tensors.
   Also the state_dict is modified in-place.

   :param state_dict: The state dict with the weights.
   :type state_dict: dict[str, torch.Tensor]
   :param apply_transforms_to_keys: The dict that maps the transform
                                    to the keys of the state dict.
   :type apply_transforms_to_keys: dict[str, list[str]]

   :returns: The state dict with the transformed weights
   :rtype: dict[str, torch.Tensor]


