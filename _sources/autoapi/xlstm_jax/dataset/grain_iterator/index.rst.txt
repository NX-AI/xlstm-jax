xlstm_jax.dataset.grain_iterator
================================

.. py:module:: xlstm_jax.dataset.grain_iterator

.. autoapi-nested-parse::

   Copyright 2023 Google LLC.

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

   LLM Data Iterator with Grain



Attributes
----------

.. autoapisummary::

   xlstm_jax.dataset.grain_iterator.LOGGER


Functions
---------

.. autoapisummary::

   xlstm_jax.dataset.grain_iterator.shard_grain_dataset
   xlstm_jax.dataset.grain_iterator.make_grain_llm_dataset
   xlstm_jax.dataset.grain_iterator.make_grain_multihost_iterator
   xlstm_jax.dataset.grain_iterator.make_grain_llm_iterator


Module Contents
---------------

.. py:data:: LOGGER

.. py:function:: shard_grain_dataset(grain_dataset, dataloading_host_index, dataloading_host_count)

   Shard a grain dataset for multi-host training.

   This function will slice the dataset into the correct shard for the host. If the dataset is not evenly divisible
   by the number of hosts, the first N hosts will have one more element than the rest. This is to ensure that the
   dataset is evenly divided across all hosts.

   :param grain_dataset: The grain dataset to shard.
   :param dataloading_host_index: The index of the dataloading host. Will be used to select the correct shard of the
                                  dataset.
   :param dataloading_host_count: The number of dataloading hosts. Will be used to determine the shard size.

   :returns: The sharded grain dataset.


.. py:function:: make_grain_llm_dataset(dataloading_host_index, dataloading_host_count, dataset, global_batch_size, max_target_length, shuffle, data_shuffle_seed, num_epochs = None, operations = None, grain_packing = False, grain_packing_bin_count = None, shift = True, shift_target = True, eod_token_id = 0, apply_padding = True)

   Create a grain IterDataset for LLM training.

   The dataset will perform packing, padding, and shifting of the data. However, no batching
   will be performed. The dataset will be returned as a grain IterDataset object that can be
   used to create a multi-host iterator.

   :param dataloading_host_index: The index of the dataloading host. Will be used to select the
                                  correct shard of the dataset. In JAX, this is equivalent to :func:`jax.process_index()`.
   :param dataloading_host_count: The number of dataloading hosts. Will be used to determine the
                                  shard size. In JAX, this is equivalent to :func:`jax.process_count()`.
   :param dataset: The dataset to load. Should provide a `__getitem__` method to access elements.
   :param global_batch_size: The global batch size.
   :param max_target_length: The maximum target length.
   :param shuffle: Whether to shuffle the dataset.
   :param data_shuffle_seed: The shuffle seed.
   :param num_epochs: The number of epochs to train for. The dataset will be repeated for so
                      many epochs, and the shuffle order will be different for each epoch. If None,
                      the dataset will be repeated infinitely. Note that batches of an epoch can
                      spill over into the first batch of the next epoch, to avoid dropping data.
                      The argument `drop_remainder` controls whether the very last batch of all epochs
                      together is dropped.
   :param operations: A list of `grain` operations to apply to the dataset before batching.
   :param grain_packing: Whether to perform packing of the data. This is useful for datasets
                         with a lot of padding, as batch elements will be packed together in a sequence
                         to reduce the amount of padding. This can improve throughput efficiency. NOTE:
                         if packing is enabled, the length of the iterator cannot be determined in advance
                         and is likely incorrect in the iterator (will be set to maximum number of batches).
   :param grain_packing_bin_count: The number of packing bins to use. If not provided, the
                                   bin count will be set to the batch size. It can be beneficial to increase the packing
                                   bins to reduce padding.
   :param shift: Whether to shift the input data to create the target data.
   :param shift_target: Whether to shift the targets left (True) or inputs right (False).
   :param eod_token_id: The token ID to use for the end-of-document token. Used if shifting the
                        inputs right and adding an end-of-document token to the sequence. If not
                        provided, the default value of 0 will be used. Recommended to set this to a value
                        explicitly with the tokenizer's EOD token ID.
   :param apply_padding: Pad sequence to the maximum target length.

   :returns: A :class:`grain.IterDataset` object that can be used to iterate over the dataset or apply further
             transformations.


.. py:function:: make_grain_multihost_iterator(grain_datasets, dataset_lengths, global_mesh, global_batch_size, dataloading_host_count, dataset_weights = None, worker_count = 1, worker_buffer_size = 1, drop_remainder = True, batch_class = LLMBatch, reset_after_epoch = False, use_thread_prefetch = False, batch_rampup_factors = None)

   Create a multi-host dataloader for LLM training.

   The dataloader will perform batch packing, padding, and shifting of the data to create
   a batch of LLMBatch objects. The LLMBatch object will contain the input and target data
   for the model.

   :param grain_datasets: The grain datasets to iterate over. If multiple datasets are provided,
                          they will be mixed together. The datasets should be provided as a list of grain datasets.
   :param dataset_weights: The weights for the datasets. If not provided, the datasets will be mixed
                           with equal weights.
   :param dataset_lengths: The lengths of the datasets for a single epoch. Used to determine the
                           length of the iterator. An epoch corresponds to the time until the longest dataset
                           is exhausted, i.e. at least one epoch for all datasets.
   :param global_mesh: The global mesh to shard the data over.
   :param global_batch_size: The global batch size.
   :param dataloading_host_count: The number of dataloading hosts. Will be used to determine the
                                  shard size. In JAX, this is equivalent to :func:`jax.process_count()`.
   :param worker_count: The number of workers to use. In `grain`, a single worker is usually
                        sufficient, as the data loading is done in parallel across hosts.
   :param worker_buffer_size: The buffer size for the workers.
   :param drop_remainder: Whether to drop the remainder of the dataset. Note that in case of
                          providing a number of epochs, the last batch of all epochs together will be
                          dropped if this is set to `True`. If set to `False`, the last batch of all epochs
                          together will be included in the iterator.
   :param batch_class: Batch class used to collate data to batch.
   :param reset_after_epoch: Whether to reset the iterator after each epoch. If set to `True`,
                             the iterator will start from the beginning of the dataset after each epoch. If set
                             to `False`, the iterator will continue from where it left off in the dataset. Note
                             that resetting the iterator can be expensive in a multi-host setup and can fail if
                             the multiprocessing pool could not be set up.
   :param use_thread_prefetch: Whether to use thread prefetching instead of multiprocessing.
   :param batch_rampup_factors: A dictionary of boundaries and scales for the batch
                                ramp-up schedule. If provided, the batch size will be ramped up according to the
                                schedule. The boundaries are the steps at which the batch size will be increased,
                                and the scales are the factors by which the batch size will be scaled. Note that
                                the factors are not accumulated, but applied to the initial batch size. If not provided,
                                the global batch size will be used as the batch size.

   :returns: A :class:`MultiHostDataLoadIterator` object that can be used to iterate over the dataset.


.. py:function:: make_grain_llm_iterator(dataloading_host_index, dataloading_host_count, global_mesh, dataset, global_batch_size, max_target_length, shuffle, data_shuffle_seed, num_epochs = None, operations = None, grain_packing = False, grain_packing_bin_count = None, shift = True, shift_target = True, eod_token_id = 0, worker_count = 1, worker_buffer_size = 1, apply_padding = True, drop_remainder = True, batch_class = LLMBatch, reset_after_epoch = False, use_thread_prefetch = False, batch_rampup_factors = None)

   Create a multi-host dataloader for LLM training.

   Combines the creation of the grain dataset and the multi-host iterator into a single function.
   See :func:`make_grain_llm_dataset` and :func:`make_grain_multihost_iterator` for more details.

   :param dataloading_host_index: The index of the dataloading host. Will be used to select the
                                  correct shard of the dataset. In JAX, this is equivalent to :func:`jax.process_index()`.
   :param dataloading_host_count: The number of dataloading hosts. Will be used to determine the
                                  shard size. In JAX, this is equivalent to :func:`jax.process_count()`.
   :param global_mesh: The global mesh to shard the data over.
   :param dataset: The dataset to load. Should provide a `__getitem__` method to access elements.
   :param global_batch_size: The global batch size.
   :param max_target_length: The maximum target length.
   :param shuffle: Whether to shuffle the dataset.
   :param data_shuffle_seed: The shuffle seed.
   :param num_epochs: The number of epochs to train for. The dataset will be repeated for so
                      many epochs, and the shuffle order will be different for each epoch. If None,
                      the dataset will be repeated infinitely. Note that batches of an epoch can
                      spill over into the first batch of the next epoch, to avoid dropping data.
                      The argument `drop_remainder` controls whether the very last batch of all epochs
                      together is dropped.
   :param operations: A list of `grain` operations to apply to the dataset before batching.
   :param grain_packing: Whether to perform packing of the data. This is useful for datasets
                         with a lot of padding, as batch elements will be packed together in a sequence
                         to reduce the amount of padding. This can improve throughput efficiency. NOTE:
                         if packing is enabled, the length of the iterator cannot be determined in advance
                         and is likely incorrect in the iterator (will be set to maximum number of batches).
   :param grain_packing_bin_count: The number of packing bins to use. If not provided, the
                                   bin count will be set to the batch size. It can be beneficial to increase the packing
                                   bins to reduce padding.
   :param shift: Whether to shift the input data to create the target data.
   :param shift_target: Whether to shift the targets left (True) or inputs right (False).
   :param eod_token_id: The token ID to use for the end-of-document token. Used if shifting
                        the inputs right and adding an end-of-document token to the sequence. If not
                        provided, the default value of 0 will be used. Recommended to set this to a value
                        explicitly with the tokenizer's EOD token ID.
   :param worker_count: The number of workers to use. In `grain`, a single worker is usually
                        sufficient, as the data loading is done in parallel across hosts.
   :param worker_buffer_size: The buffer size for the workers.
   :param apply_padding: Pad sequence to the maximum target length.
   :param drop_remainder: Whether to drop the remainder of the dataset. Note that in case of
                          providing a number of epochs, the last batch of all epochs together will be
                          dropped if this is set to `True`. If set to `False`, the last batch of all epochs
                          together will be included in the iterator.
   :param batch_class: Batch class used to collate data to batch.
   :param reset_after_epoch: Whether to reset the iterator after each epoch. If set to `True`,
                             the iterator will start from the beginning of the dataset after each epoch. If set
                             to `False`, the iterator will continue from where it left off in the dataset. Note
                             that resetting the iterator can be expensive in a multi-host setup and can fail if
                             the multiprocessing pool could not be set up.
   :param use_thread_prefetch: Whether to use thread prefetching instead of multiprocessing.
   :param batch_rampup_factors: A dictionary of boundaries and scales for the batch
                                rampup schedule. If provided, the batch size will be ramped up according to the
                                schedule. The boundaries are the steps at which the batch size will be increased,
                                and the scales are the factors by which the batch size will be scaled. Note that
                                the factors are not accumulated, but applied to the initial batch size. If not provided,
                                the global batch size will be used as the batch size.

   :returns: A :class:`MultiHostDataLoadIterator` object that can be used to iterate over the dataset.


