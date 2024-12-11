xlstm_jax.utils.model_param_handling.convert_checkpoint
=======================================================

.. py:module:: xlstm_jax.utils.model_param_handling.convert_checkpoint


Functions
---------

.. autoapisummary::

   xlstm_jax.utils.model_param_handling.convert_checkpoint.convert_orbax_checkpoint_to_torch_state_dict


Module Contents
---------------

.. py:function:: convert_orbax_checkpoint_to_torch_state_dict(orbax_pytree, split_blocks = True, blocks_layer_name = 'blocks')

   Convert orbax pytree params to a (flat) torch state dict.

   :param orbax_pytree: The orbax pytree params.
   :type orbax_pytree: dict[str, Any]
   :param split_blocks: Whether to split the parameters of the blocks into individual tensors. Defaults to True.
                        Jax stores the weight tensors/arrays of the all blocks in
                        a single tensor/array with the first dimension being the number of blocks.
                        PyTorch expects the weights of each block to be a separate tensor/array.
                        This is why we split the weights of each block into separate tensors/arrays.
   :type split_blocks: bool
   :param blocks_layer_name: The blocks layer name to split parameters by. Defaults to "blocks".
   :type blocks_layer_name: str

   :returns: The torch state dict.
   :rtype: dict[str, torch.Tensor]


