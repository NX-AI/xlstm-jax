xlstm_jax.dataset.multihost_dataloading
=======================================

.. py:module:: xlstm_jax.dataset.multihost_dataloading

.. autoapi-nested-parse::

   Copyright 2023 Google LLC

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   ---

   SPMD Multihost Dataloading Utilities.

   Adapted from Sholto's:
   https://github.com/sholtodouglas/multihost_dataloading



Attributes
----------

.. autoapisummary::

   xlstm_jax.dataset.multihost_dataloading.LOGGER


Classes
-------

.. autoapisummary::

   xlstm_jax.dataset.multihost_dataloading.MultiHostDataLoadIterator


Functions
---------

.. autoapisummary::

   xlstm_jax.dataset.multihost_dataloading._build_global_shape_and_sharding
   xlstm_jax.dataset.multihost_dataloading._form_global_array
   xlstm_jax.dataset.multihost_dataloading._pad_array_to_shape
   xlstm_jax.dataset.multihost_dataloading.get_next_batch_sharded


Module Contents
---------------

.. py:data:: LOGGER

.. py:function:: _build_global_shape_and_sharding(local_shape, global_mesh)

   Create the global_shape and sharding based on the local_shape and global_mesh.

   :param local_shape: Local tensor shape
   :param global_mesh: Global mesh of devices

   :returns: Global tensor shape, Named Sharding of the mesh


.. py:function:: _form_global_array(path, array, global_mesh)

   Put host sharded array into devices within a global sharded array.

   :param path: Tree def path of the array in a PyTree struct (for debugging purposes only)
   :param array: Distributed host array.
   :param global_mesh: Global mesh for the distributed array.

   :returns: Distributed device array


.. py:function:: _pad_array_to_shape(array_and_shape, pad_value = 0)

   Pad an array to a given shape by given values. Array and shape are inside a shared tuple to
   enable easier mapping from a zip().

   :param array_and_shape: The array and shape it should be padded to.
   :param pad_value: Padding value.

   :returns: Padded array.

   >>> np.allclose(
   ...     _pad_array_to_shape((np.array([[1], [2]]), (3, 2)), pad_value=0),
   ...     np.array([[1, 0], [2, 0], [0, 0]]))
   True



.. py:function:: get_next_batch_sharded(local_iterator, global_mesh, pad = False, pad_value = 0)

   Splits the host loaded data equally over all devices. Optionally pad arrays for equal sizes.

   :param local_iterator: Local dataloader iterator.
   :param global_mesh: Global device mesh.
   :param pad: Whether to pad the batch.
   :param pad_value: Value to pad the batch with. Defaults to zero.

   :returns: Optionally padded, sharded data array.


.. py:class:: MultiHostDataLoadIterator(dataloader, global_mesh, iterator_length = None, dataset_size = None, reset_after_epoch = False, pad_shapes = False, pad_value = 0)

   Create a MultiHostDataLoadIterator.

   Wrapper around a :class:`tf.data.Dataset` or Iterable to iterate over data in a multi-host setup.
   Folds get_next_batch_sharded into an iterator class, and supports breaking indefinite iterator into epochs.

   :param dataloader: The dataloader to iterate over.
   :param global_mesh: The mesh to shard the data over.
   :param iterator_length: The length of the iterator. If provided, the iterator will stop after this many steps with a
                           :class:`StopIteration` exception. Otherwise, will continue over the iterator until it raises an exception
                           itself.
   :param dataset_size: size of the dataset. If provided, will be returned by get_dataset_size. Otherwise, will return
                        `None`. Can be used to communicate the dataset size to functions that use the iterator.
   :param reset_after_epoch: Whether to reset the iterator between epochs or not. If `True`, the iterator will reset
                             after each epoch, otherwise it will continue from where it left off. If you have an indefinite iterator
                             (e.g. train iterator with grain and shuffle), this should be set to `False`. For un-shuffled iterators in
                             grain (e.g. validation), this should be set to `True`.
   :param pad_shapes: Whether to pad arrays to a common shape across all devices before merging.
   :param pad_value: Value to use for padding. Defaults to zero.


   .. py:attribute:: global_mesh


   .. py:attribute:: dataloader


   .. py:attribute:: iterator_length
      :value: None



   .. py:attribute:: dataset_size
      :value: None



   .. py:attribute:: reset_after_epoch
      :value: False



   .. py:attribute:: state_set
      :value: False



   .. py:attribute:: step_counter
      :value: 0



   .. py:attribute:: pad_shapes
      :value: False



   .. py:attribute:: pad_value
      :value: 0



   .. py:method:: reset()


   .. py:method:: get_state()


   .. py:method:: set_state(state)


