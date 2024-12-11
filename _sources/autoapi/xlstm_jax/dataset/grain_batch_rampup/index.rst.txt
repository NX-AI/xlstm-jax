xlstm_jax.dataset.grain_batch_rampup
====================================

.. py:module:: xlstm_jax.dataset.grain_batch_rampup

.. autoapi-nested-parse::

   Batch rampup schedule for grain IterDatasets.

   This module provides a BatchRampUpIterDataset that allows for a batch rampup
   schedule to be provided. The batch rampup schedule is a function that takes the
   current batch step and returns the batch size. It can be used to gradually
   increase the batch size over time.

   The implementation is based on the standard BatchIterDataset from grain, with
   the addition of a batch rampup schedule. See
   grain._src.python.dataset.transformations.batch for more details.

   NOTE: If the grain API for batching changes, this module may need to be updated.



Attributes
----------

.. autoapisummary::

   xlstm_jax.dataset.grain_batch_rampup.T
   xlstm_jax.dataset.grain_batch_rampup.S
   xlstm_jax.dataset.grain_batch_rampup.LOGGER


Classes
-------

.. autoapisummary::

   xlstm_jax.dataset.grain_batch_rampup._BatchRampUpDatasetIterator
   xlstm_jax.dataset.grain_batch_rampup.BatchRampUpIterDataset


Functions
---------

.. autoapisummary::

   xlstm_jax.dataset.grain_batch_rampup._make_batch
   xlstm_jax.dataset.grain_batch_rampup.batch_dataset_with_rampup
   xlstm_jax.dataset.grain_batch_rampup.create_batch_rampup_schedule
   xlstm_jax.dataset.grain_batch_rampup.constant_rampup_schedule
   xlstm_jax.dataset.grain_batch_rampup.stepwise_rampup_schedule


Module Contents
---------------

.. py:data:: T

.. py:data:: S

.. py:data:: LOGGER

.. py:function:: _make_batch(values)

   Returns a batch of values with a new batch dimension at the front.


.. py:class:: _BatchRampUpDatasetIterator(parent, batch_rampup_schedule, drop_remainder, batch_fn, stats)

   Bases: :py:obj:`grain._src.python.dataset.dataset.DatasetIterator`\ [\ :py:obj:`T`\ ]


   Iterator that batches elements with a batch rampup schedule.


   .. py:attribute:: _parent


   .. py:attribute:: _batch_rampup_schedule


   .. py:attribute:: _drop_remainder


   .. py:attribute:: _batch_fn


   .. py:attribute:: _batch_step
      :value: 0



   .. py:method:: get_state()

      Return the state of the iterator.



   .. py:method:: set_state(state)

      Set the state of the iterator.



.. py:class:: BatchRampUpIterDataset(parent, batch_rampup_schedule, drop_remainder = False, batch_fn = None)

   Bases: :py:obj:`grain._src.python.dataset.dataset.IterDataset`\ [\ :py:obj:`T`\ ]


   Batch transformation with ramp up for IterDatasets.


   .. py:attribute:: _batch_rampup_schedule


   .. py:attribute:: _drop_remainder
      :value: False



   .. py:attribute:: _batch_fn


.. py:function:: batch_dataset_with_rampup(parent, batch_size, drop_remainder = False, batch_fn = None, schedule_type = 'stepwise', boundaries_and_scales = None)

   Creates a BatchRampUpIterDataset from an IterDataset.

   :param parent: The parent IterDataset whose elements are batched.
   :param batch_size: The initial batch size.
   :param drop_remainder: Whether to drop the last batch if it is smaller than
                          batch_size.
   :param batch_fn: A function that takes a list of elements and returns a batch.
                    Defaults to stacking the elements along a new batch dimension.
   :param schedule_type: The type of the batch rampup schedule. Supported types are
                         "constant" and "stepwise".
   :param boundaries_and_scales: Used only for the "stepwise" schedule type. A dictionary
                                 mapping the boundaries b_i to non-negative scaling factors f_i. For any
                                 step count s, the schedule returns batch_size scaled by the product of
                                 factor f_i for the largest b_i such that b_i < s.

   :returns: A BatchRampUpIterDataset that batches elements. If no schedule is provided,
             falls back to the standard BatchIterDataset.


.. py:function:: create_batch_rampup_schedule(batch_size, schedule_type, boundaries_and_scales = None)

   Creates a batch rampup schedule.

   :param batch_size: The initial batch size.
   :param schedule_type: The type of the batch rampup schedule. Supported types are
                         "constant" and "stepwise".
   :param boundaries_and_scales: A dictionary mapping the boundaries b_i to non-negative
                                 scaling factors f_i. For any step count s, the schedule returns batch_size
                                 scaled by the product of factor f_i for the largest b_i such that b_i < s.
                                 Only required for the "stepwise" schedule.

   :returns: A function that takes the current batch step and returns the batch size.


.. py:function:: constant_rampup_schedule(batch_size)

   Returns a constant batch rampup schedule.

   :param batch_size: The constant batch size.

   :returns: A function that takes the current batch step and returns the batch size.


.. py:function:: stepwise_rampup_schedule(batch_size, boundaries_and_scales)

   Returns a stepwise batch rampup schedule.

   :param batch_size: The initial batch size on which the factors are applied.
   :param boundaries_and_scales: A dictionary mapping the boundaries b_i to non-negative scaling factors f_i.
                                 For any step count s, the schedule returns batch_size scaled by the product of factor f_i for the
                                 largest b_i such that b_i < s.

   :returns: A function that takes the current batch step and returns the batch size.


