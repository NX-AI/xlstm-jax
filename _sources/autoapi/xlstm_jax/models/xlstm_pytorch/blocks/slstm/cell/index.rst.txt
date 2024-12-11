xlstm_jax.models.xlstm_pytorch.blocks.slstm.cell
================================================

.. py:module:: xlstm_jax.models.xlstm_pytorch.blocks.slstm.cell


Attributes
----------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.slstm.cell.LOGGER
   xlstm_jax.models.xlstm_pytorch.blocks.slstm.cell.DTYPE_DICT
   xlstm_jax.models.xlstm_pytorch.blocks.slstm.cell.DTYPES
   xlstm_jax.models.xlstm_pytorch.blocks.slstm.cell.curdir
   xlstm_jax.models.xlstm_pytorch.blocks.slstm.cell.rnn_function_registry
   xlstm_jax.models.xlstm_pytorch.blocks.slstm.cell._python_dtype_to_cuda_dtype


Classes
-------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.slstm.cell.sLSTMCellConfig
   xlstm_jax.models.xlstm_pytorch.blocks.slstm.cell.sLSTMCellBase
   xlstm_jax.models.xlstm_pytorch.blocks.slstm.cell.sLSTMCellCUDA
   xlstm_jax.models.xlstm_pytorch.blocks.slstm.cell.sLSTMCell_vanilla
   xlstm_jax.models.xlstm_pytorch.blocks.slstm.cell.sLSTMCell_cuda
   xlstm_jax.models.xlstm_pytorch.blocks.slstm.cell.sLSTMCell


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_pytorch.blocks.slstm.cell.sLSTMCellFuncGenerator


Module Contents
---------------

.. py:data:: LOGGER

.. py:data:: DTYPE_DICT

.. py:data:: DTYPES

.. py:data:: curdir

.. py:data:: rnn_function_registry

.. py:data:: _python_dtype_to_cuda_dtype

.. py:class:: sLSTMCellConfig

   .. py:attribute:: hidden_size
      :type:  int
      :value: -1



   .. py:attribute:: num_heads
      :type:  int
      :value: 4



   .. py:attribute:: num_states
      :type:  int
      :value: 4



   .. py:attribute:: backend
      :type:  Literal['vanilla', 'cuda']
      :value: 'cuda'



   .. py:attribute:: function
      :type:  str
      :value: 'slstm'



   .. py:attribute:: bias_init
      :type:  Literal['powerlaw_blockdependent', 'small_init', 'standard']
      :value: 'powerlaw_blockdependent'



   .. py:attribute:: recurrent_weight_init
      :type:  Literal['zeros', 'standard']
      :value: 'zeros'



   .. py:attribute:: _block_idx
      :type:  int
      :value: 0



   .. py:attribute:: _num_blocks
      :type:  int
      :value: 1



   .. py:attribute:: num_gates
      :type:  int
      :value: 4



   .. py:attribute:: gradient_recurrent_cut
      :type:  bool
      :value: False



   .. py:attribute:: gradient_recurrent_clipval
      :type:  float | None
      :value: None



   .. py:attribute:: forward_clipval
      :type:  float | None
      :value: None



   .. py:attribute:: batch_size
      :type:  int
      :value: 8



   .. py:attribute:: input_shape
      :type:  Literal['BSGNH', 'SBGNH']
      :value: 'BSGNH'



   .. py:attribute:: internal_input_shape
      :type:  Literal['SBNGH', 'SBGNH', 'SBNHG']
      :value: 'SBNGH'



   .. py:attribute:: output_shape
      :type:  Literal['BNSH', 'SBH', 'BSH', 'SBNH']
      :value: 'BNSH'



   .. py:attribute:: constants
      :type:  dict


   .. py:attribute:: dtype
      :type:  DTYPES
      :value: 'bfloat16'



   .. py:attribute:: dtype_b
      :type:  DTYPES | None
      :value: 'float32'



   .. py:attribute:: dtype_r
      :type:  DTYPES | None
      :value: None



   .. py:attribute:: dtype_w
      :type:  DTYPES | None
      :value: None



   .. py:attribute:: dtype_g
      :type:  DTYPES | None
      :value: None



   .. py:attribute:: dtype_s
      :type:  DTYPES | None
      :value: None



   .. py:attribute:: dtype_a
      :type:  DTYPES | None
      :value: None



   .. py:attribute:: enable_automatic_mixed_precision
      :type:  bool
      :value: True



   .. py:attribute:: initial_val
      :type:  float | collections.abc.Sequence[float]
      :value: 0.0



   .. py:property:: head_dim


   .. py:property:: input_dim


   .. py:property:: torch_dtype
      :type: torch.dtype



   .. py:property:: torch_dtype_b
      :type: torch.dtype



   .. py:property:: torch_dtype_r
      :type: torch.dtype



   .. py:property:: torch_dtype_w
      :type: torch.dtype



   .. py:property:: torch_dtype_s
      :type: torch.dtype



   .. py:property:: defines


.. py:class:: sLSTMCellBase(config)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: config_class


   .. py:attribute:: config


   .. py:attribute:: _recurrent_kernel_


   .. py:attribute:: recurrent_kernel


   .. py:attribute:: _bias_


   .. py:attribute:: bias


   .. py:property:: _recurrent_kernel


   .. py:property:: _bias


   .. py:method:: _recurrent_kernel_ext2int(recurrent_kernel_ext)


   .. py:method:: _bias_ext2int(bias_ext)


   .. py:method:: _recurrent_kernel_int2ext(recurrent_kernel_int)


   .. py:method:: _bias_int2ext(bias_int)


   .. py:method:: parameters_to_dtype()


   .. py:property:: head_dim


   .. py:method:: _permute_input(x)


   .. py:method:: _permute_output(x)


   .. py:method:: reset_parameters()

      Resets this layer's parameters to their initial values.



   .. py:method:: _check_input(input)


   .. py:method:: _zero_state(input)

      Return a zero state matching dtype and batch size of `input`.

      :param input: Tensor, to specify the device and dtype of the returned tensors.

      :returns: a nested structure of zero Tensors.
      :rtype: zero_state



   .. py:method:: _get_state(input, state = None)


   .. py:method:: _get_final_state(all_states)
      :staticmethod:


      All states have the structure [STATES, SEQUENCE, BATCH, HIDDEN]



   .. py:method:: _is_cuda()


   .. py:method:: step(input, state)


   .. py:method:: forward(input, state=None)


.. py:class:: sLSTMCellCUDA

   .. py:attribute:: mod


   .. py:method:: instance(config)
      :classmethod:



.. py:function:: sLSTMCellFuncGenerator(training, config)

.. py:class:: sLSTMCell_vanilla(config)

   Bases: :py:obj:`sLSTMCellBase`


   .. py:attribute:: config_class


   .. py:attribute:: pointwise


   .. py:method:: _recurrent_kernel_ext2int(recurrent_kernel_ext)


   .. py:method:: _recurrent_kernel_int2ext(recurrent_kernel_int)


   .. py:method:: _bias_ext2int(bias_ext)


   .. py:method:: _bias_int2ext(bias_int)


   .. py:method:: _impl(input, state)


   .. py:method:: _impl_step(input, state)


   .. py:attribute:: config


   .. py:attribute:: _recurrent_kernel_


   .. py:attribute:: recurrent_kernel


   .. py:attribute:: _bias_


   .. py:attribute:: bias


   .. py:property:: _recurrent_kernel


   .. py:property:: _bias


   .. py:method:: parameters_to_dtype()


   .. py:property:: head_dim


   .. py:method:: _permute_input(x)


   .. py:method:: _permute_output(x)


   .. py:method:: reset_parameters()

      Resets this layer's parameters to their initial values.



   .. py:method:: _check_input(input)


   .. py:method:: _zero_state(input)

      Return a zero state matching dtype and batch size of `input`.

      :param input: Tensor, to specify the device and dtype of the returned tensors.

      :returns: a nested structure of zero Tensors.
      :rtype: zero_state



   .. py:method:: _get_state(input, state = None)


   .. py:method:: _get_final_state(all_states)
      :staticmethod:


      All states have the structure [STATES, SEQUENCE, BATCH, HIDDEN]



   .. py:method:: _is_cuda()


   .. py:method:: step(input, state)


   .. py:method:: forward(input, state=None)


.. py:class:: sLSTMCell_cuda(config, skip_backend_init = False)

   Bases: :py:obj:`sLSTMCellBase`


   .. py:attribute:: config_class


   .. py:attribute:: internal_input_shape
      :value: 'SBNGH'



   .. py:method:: _recurrent_kernel_ext2int(recurrent_kernel_ext)


   .. py:method:: _recurrent_kernel_int2ext(recurrent_kernel_int)


   .. py:method:: _bias_ext2int(bias_ext)


   .. py:method:: _bias_int2ext(bias_int)


   .. py:method:: _impl_step(training, input, state)


   .. py:method:: _impl(training, input, state)


   .. py:attribute:: config


   .. py:attribute:: _recurrent_kernel_


   .. py:attribute:: recurrent_kernel


   .. py:attribute:: _bias_


   .. py:attribute:: bias


   .. py:property:: _recurrent_kernel


   .. py:property:: _bias


   .. py:method:: parameters_to_dtype()


   .. py:property:: head_dim


   .. py:method:: _permute_input(x)


   .. py:method:: _permute_output(x)


   .. py:method:: reset_parameters()

      Resets this layer's parameters to their initial values.



   .. py:method:: _check_input(input)


   .. py:method:: _zero_state(input)

      Return a zero state matching dtype and batch size of `input`.

      :param input: Tensor, to specify the device and dtype of the returned tensors.

      :returns: a nested structure of zero Tensors.
      :rtype: zero_state



   .. py:method:: _get_state(input, state = None)


   .. py:method:: _get_final_state(all_states)
      :staticmethod:


      All states have the structure [STATES, SEQUENCE, BATCH, HIDDEN]



   .. py:method:: _is_cuda()


   .. py:method:: step(input, state)


   .. py:method:: forward(input, state=None)


.. py:class:: sLSTMCell(config)

   Bases: :py:obj:`sLSTMCellBase`


   .. py:attribute:: config_class


   .. py:attribute:: config


   .. py:attribute:: _recurrent_kernel_


   .. py:attribute:: recurrent_kernel


   .. py:attribute:: _bias_


   .. py:attribute:: bias


   .. py:property:: _recurrent_kernel


   .. py:property:: _bias


   .. py:method:: _recurrent_kernel_ext2int(recurrent_kernel_ext)


   .. py:method:: _bias_ext2int(bias_ext)


   .. py:method:: _recurrent_kernel_int2ext(recurrent_kernel_int)


   .. py:method:: _bias_int2ext(bias_int)


   .. py:method:: parameters_to_dtype()


   .. py:property:: head_dim


   .. py:method:: _permute_input(x)


   .. py:method:: _permute_output(x)


   .. py:method:: reset_parameters()

      Resets this layer's parameters to their initial values.



   .. py:method:: _check_input(input)


   .. py:method:: _zero_state(input)

      Return a zero state matching dtype and batch size of `input`.

      :param input: Tensor, to specify the device and dtype of the returned tensors.

      :returns: a nested structure of zero Tensors.
      :rtype: zero_state



   .. py:method:: _get_state(input, state = None)


   .. py:method:: _get_final_state(all_states)
      :staticmethod:


      All states have the structure [STATES, SEQUENCE, BATCH, HIDDEN]



   .. py:method:: _is_cuda()


   .. py:method:: step(input, state)


   .. py:method:: forward(input, state=None)


