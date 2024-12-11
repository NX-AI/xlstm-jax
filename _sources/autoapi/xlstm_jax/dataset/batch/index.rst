xlstm_jax.dataset.batch
=======================

.. py:module:: xlstm_jax.dataset.batch


Attributes
----------

.. autoapisummary::

   xlstm_jax.dataset.batch.dataclass_kwonly


Classes
-------

.. autoapisummary::

   xlstm_jax.dataset.batch.Batch
   xlstm_jax.dataset.batch.LLMBatch
   xlstm_jax.dataset.batch.LLMIndexedBatch


Module Contents
---------------

.. py:data:: dataclass_kwonly

.. py:class:: Batch

   Batch of training data.


   .. py:attribute:: inputs
      :type:  jax.Array

      The input data.


   .. py:attribute:: targets
      :type:  jax.Array

      The target data.


.. py:class:: LLMBatch

   Bases: :py:obj:`Batch`


   Batch for LLM training.

   Contains inputs and targets along with their respective positions and segmentations.

   ------------------------------------------------------------------------------------
   Example: Consider sequences [1, 2, 3] and [4, 5], [6, 7].
   These will be grouped into one sequence of length 8 and separated by an end-of-document token E.
   Padding is token 0. Segmentations are used to separate the subsequences.

   Note that we use grain packing (FirstFitPackIterDataset) for grouping the smaller sequence.
   If the last subsequence does not fit in the same sequence, it will be used in the next sequence (not shown below).

   Using packing & shift inputs (right):
       targets:              [1, 2, 3, E, 4, 5, E, 0]
       inputs:               [E, 1, 2, 3, E, 4, 5, E] # The first token is always E (marking the beginning).
       targets_segmentation: [1, 1, 1, 1, 2, 2, 2, 0]
       inputs_segmentation:  [1, 1, 1, 1, 2, 2, 2, 0]
       targets_position:     [0, 1, 2, 3, 0, 1, 2, 0]
       inputs_position:      [0, 1, 2, 3, 0, 1, 2, 0]
       doc borders:          [1, 0, 0, 0, 1, 0, 0, 1]
   Note that the two segmentations are identical. They would be different for prediction with multiple prefix tokens.


   .. py:attribute:: inputs_position
      :type:  jax.Array

      np.int32

      :type: Positions of the input tokens. dtype


   .. py:attribute:: inputs_segmentation
      :type:  jax.Array

      np.int32

      :type: Segmentation of the input tokens. 0 to indicate padding. dtype


   .. py:attribute:: targets_position
      :type:  jax.Array

      np.int32

      :type: Positions of the target tokens. dtype


   .. py:attribute:: targets_segmentation
      :type:  jax.Array

      np.int32

      :type: Segmentation of the target tokens. 0 to indicate padding. dtype


   .. py:attribute:: _document_borders
      :type:  jax.Array | None
      :value: None


      Document borders for the input data. This buffer should only be used to explicitly overwrite the standard
      algorithm to calculate the document borders; for instance, if slicing the batch. Otherwise, use
      `:func:get_document_borders` to get the document borders. dtype: bool


   .. py:method:: get_document_borders()

      Get the document borders for the input data.

      A token represents a document border if its previous target token has a different target segmentation.
      For instance, if the input segmentation is [1, 1, 2, 2, 2, 3], the document borders are [1, 0, 1, 0, 0, 1].
      This mask can be useful for processing documents separately in a recurrent model, i.e. when to reset the
      hidden state.
      Note: If the last tokens are paddings, marking invalid tokens, the border between the last document and
      padding will also be marked as document border.

      :returns: A boolean array indicating the document borders.



   .. py:method:: from_inputs(inputs, targets = None)
      :staticmethod:


      Create LLMBatch from inputs.

      Helper function for quickly creating a default LLM Batch.

      :param inputs: The input data.
      :type inputs: jax.Array
      :param targets: The target data. If not provided, the inputs are used as targets and the
                      inputs are shifted right by one.
      :type targets: jax.Array, optional

      :returns: An LLMBatch with respective inputs and targets.



   .. py:method:: get_dtype_struct(batch_size, max_length)
      :staticmethod:


      Get the shape and dtype structure for LLMBatch.

      :param batch_size: The size of the batch.
      :type batch_size: int
      :param max_length: The maximum length of the sequences.
      :type max_length: int

      :returns: An LLMBatch with :class:`jax.ShapeDtypeStruct` typed components.



   .. py:method:: get_sample(batch_size, max_length)
      :classmethod:


      Get a real sample of an LLMBatch. Needed for compilation when using
      jax.debug.* in the model or anywhere else in the pipeline.

      :param batch_size: The size of the batch.
      :type batch_size: int
      :param max_length: The maximum length of the sequences.
      :type max_length: int



   .. py:attribute:: inputs
      :type:  jax.Array

      The input data.


   .. py:attribute:: targets
      :type:  jax.Array

      The target data.


.. py:class:: LLMIndexedBatch

   Bases: :py:obj:`LLMBatch`


   Batch for LLM data with document indices and sequence indices for correct ordering.

   `document_idx` equals zero means padding.


   .. py:attribute:: document_idx
      :type:  jax.Array

      np.int32

      :type: Document indices for batch sequences. dtype


   .. py:attribute:: sequence_idx
      :type:  jax.Array

      np.int32

      :type: Sequence indices within documents for batch sequences. dtype


   .. py:method:: from_inputs(inputs, document_idx, sequence_idx, targets = None)
      :staticmethod:


      Create LLMBatch from inputs.

      Helper function for quickly creating a default LLM Batch.

      :param inputs: The input data.
      :type inputs: jax.Array
      :param targets: The target data.
      :type targets: jax.Array
      :param sequence_idx: The sequence idx for each sample.
      :type sequence_idx: jax.Array
      :param document_idx: The document idx for each sample. A document might be composed of multiple
                           sequences.
      :type document_idx: jax.Array

      :returns: An LLMBatch with respective inputs and targets.



   .. py:method:: get_dtype_struct(batch_size, max_length)
      :staticmethod:


      Get the shape and dtype structure for LLMIndexedBatch.

      :param batch_size: The size of the batch.
      :type batch_size: int
      :param max_length: The maximum length of the sequences.
      :type max_length: int

      :returns: An LLMBatch with :class:`jax.ShapeDtypeStruct` typed components.



   .. py:attribute:: inputs_position
      :type:  jax.Array

      np.int32

      :type: Positions of the input tokens. dtype


   .. py:attribute:: inputs_segmentation
      :type:  jax.Array

      np.int32

      :type: Segmentation of the input tokens. 0 to indicate padding. dtype


   .. py:attribute:: targets_position
      :type:  jax.Array

      np.int32

      :type: Positions of the target tokens. dtype


   .. py:attribute:: targets_segmentation
      :type:  jax.Array

      np.int32

      :type: Segmentation of the target tokens. 0 to indicate padding. dtype


   .. py:attribute:: _document_borders
      :type:  jax.Array | None
      :value: None


      Document borders for the input data. This buffer should only be used to explicitly overwrite the standard
      algorithm to calculate the document borders; for instance, if slicing the batch. Otherwise, use
      `:func:get_document_borders` to get the document borders. dtype: bool


   .. py:method:: get_document_borders()

      Get the document borders for the input data.

      A token represents a document border if its previous target token has a different target segmentation.
      For instance, if the input segmentation is [1, 1, 2, 2, 2, 3], the document borders are [1, 0, 1, 0, 0, 1].
      This mask can be useful for processing documents separately in a recurrent model, i.e. when to reset the
      hidden state.
      Note: If the last tokens are paddings, marking invalid tokens, the border between the last document and
      padding will also be marked as document border.

      :returns: A boolean array indicating the document borders.



   .. py:method:: get_sample(batch_size, max_length)
      :classmethod:


      Get a real sample of an LLMBatch. Needed for compilation when using
      jax.debug.* in the model or anywhere else in the pipeline.

      :param batch_size: The size of the batch.
      :type batch_size: int
      :param max_length: The maximum length of the sequences.
      :type max_length: int



   .. py:attribute:: inputs
      :type:  jax.Array

      The input data.


   .. py:attribute:: targets
      :type:  jax.Array

      The target data.


