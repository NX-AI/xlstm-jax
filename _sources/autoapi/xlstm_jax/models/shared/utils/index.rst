xlstm_jax.models.shared.utils
=============================

.. py:module:: xlstm_jax.models.shared.utils


Functions
---------

.. autoapisummary::

   xlstm_jax.models.shared.utils.prepare_module
   xlstm_jax.models.shared.utils.soft_cap_logits


Module Contents
---------------

.. py:function:: prepare_module(layer, layer_name, config)

   Remats and shards layer if needed.

   This function wraps the layer function in a remat and/or sharding function if its layer name is present in the
   remat and fsdp configuration, respectively.

   :param layer: The layer to prepare.
   :param layer_name: The name of the layer.
   :param config: The configuration to use.

   :returns: The layer with remat and sharding applied if needed.


.. py:function:: soft_cap_logits(logits, cap_value)

   Soft caps logits to a value.

   Performs a tanh operation on the logits and scales the result to the cap value. Common technique in attention
   and output language heads to prevent large logits from dominating the softmax. See for example Gemma2:
   https://arxiv.org/abs/2408.00118

   :param logits: The logits to cap.
   :param cap_value: The value to cap logits to. If None, no cap is applied.

   :returns: The capped logits.


