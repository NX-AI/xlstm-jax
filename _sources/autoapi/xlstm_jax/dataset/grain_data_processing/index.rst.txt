xlstm_jax.dataset.grain_data_processing
=======================================

.. py:module:: xlstm_jax.dataset.grain_data_processing


Attributes
----------

.. autoapisummary::

   xlstm_jax.dataset.grain_data_processing.LOGGER


Classes
-------

.. autoapisummary::

   xlstm_jax.dataset.grain_data_processing.PaddedDataset


Functions
---------

.. autoapisummary::

   xlstm_jax.dataset.grain_data_processing.preprocess_dataset
   xlstm_jax.dataset.grain_data_processing.make_grain_iterator
   xlstm_jax.dataset.grain_data_processing.pad_dataset
   xlstm_jax.dataset.grain_data_processing.load_array_record_dataset
   xlstm_jax.dataset.grain_data_processing.load_huggingface_dataset


Module Contents
---------------

.. py:data:: LOGGER

.. py:function:: preprocess_dataset(dataloading_host_index, dataloading_host_count, dataset, data_column_name, tokenize, global_batch_size, max_target_length, shuffle, data_shuffle_seed, tokenizer_path = None, hf_access_token = None, add_bos = True, add_eos = True, add_eod = True, grain_packing = False, grain_packing_bin_count = None, shift = True, drop_remainder = True, num_epochs = None, tokenizer_cache_dir = None, max_steps_per_epoch = None, eod_token_id = None)

   Pipeline for preprocessing an array_records or huggingface dataset.

   :param dataloading_host_index: The index of the data loading host. Will be used to select the
                                  correct shard of the dataset. In JAX, this is equivalent to :func:`jax.process_index()`.
   :param dataloading_host_count: The number of data loading hosts. Will be used to determine the
                                  shard size. In JAX, this is equivalent to :func:`jax.process_count()`.
   :param dataset: The dataset to load. Should provide a `__getitem__` method to access elements.
   :param data_column_name: The column name for the data in the dataset.
   :param tokenize: Whether to tokenize the data.
   :param global_batch_size: The global batch size.
   :param max_target_length: The maximum target length.
   :param shuffle: Whether to shuffle the dataset.
   :param data_shuffle_seed: The shuffle seed.
   :param tokenizer_path: The path to the tokenizer.
   :param hf_access_token: The access token for HuggingFace.
   :param add_bos: Whether to add the beginning of sequence token.
   :param add_eos: Whether to add the end of sequence token.
   :param add_eod: Whether to add an end of document token.
   :param grain_packing: Whether to perform packing of the data. This is useful for datasets
                         with a lot of padding, as batch elements will be packed together in a sequence
                         to reduce the amount of padding. This can improve throughput efficiency. Note:
                         if packing is enabled, the length of the iterator cannot be determined in advance
                         and is likely incorrect in the iterator (will be set to maximum number of batches).
   :param grain_packing_bin_count: The number of packing bins to use. If not provided, the
                                   bin count will be set to the batch size. It can be beneficial to increase the packing
                                   bins to reduce padding.
   :param shift: Whether to shift the input data to create the target data.
   :param drop_remainder: Whether to drop the remainder of the dataset. Note that in case of
                          providing a number of epochs, the last batch of all epochs together will be
                          dropped if this is set to `True`. If set to `False`, the last batch of all epochs
                          together will be included in the iterator.
   :param num_epochs: The number of epochs to train for. The dataset will be repeated for so
                      many epochs, and the shuffle order will be different for each epoch. If None,
                      the dataset will be repeated infinitely. Note that batches of an epoch can
                      spill over into the first batch of the next epoch, to avoid dropping data.
                      The argument `drop_remainder` controls whether the very last batch of all epochs
                      together is dropped. By default, use None (infinite epochs) for training and validation.
   :param tokenizer_cache_dir: The cache directory for the tokenizer.
   :param max_steps_per_epoch: The maximum number of steps per epoch. If provided, the iterator
                               will stop after this many steps with a :class:`StopIteration` exception. Otherwise,
                               will continue over the iterator until all batches are consumed.
   :param eod_token_id: The token ID to use for the end-of-document token. If `tokenizer_path` is
                        provided, the tokenizer's EOD token ID is used.

   :returns: The preprocessed grain dataset and the original data source.


.. py:function:: make_grain_iterator(configs, global_mesh, process_indices, dataset_weights = None)

   Load a dataset, create the preprocessing pipeline and return a multihost data-loading iterator.

   :param configs: dataset configuration object for huggingface or arrayrecords dataset. If multiple configs are provided,
                   the datasets will be loaded in parallel and the data will be interleaved in a mixing style. NOTE: the
                   global batch size, worker count, worker buffer size, drop remainder, and batch rampup will be only used
                   from the first config. The other configs are assumed to have the same values. Otherwise, warnings will be
                   raised.
   :param global_mesh: The global mesh to shard the data over.
   :param process_indices: List of process indices that should load the real data. This is used to determine the data
                           loading host index and host count.
   :param dataset_weights: The weights for the datasets. If provided, the datasets will be mixed according to the
                           weights. Otherwise, a uniform mixing is used. If a single dataset is provided, the weights are ignored.

   :returns: data-loading iterator (for training or evaluation).


.. py:class:: PaddedDataset(dataset, full_dataset_length, column_name)

   Bases: :py:obj:`grain.python.RandomAccessDataSource`


   Dataset wrapper to pad the dataset to be a multiple of the global batch size.


   .. py:attribute:: dataset


   .. py:attribute:: full_dataset_length


   .. py:attribute:: column_name


   .. py:property:: empty_sequence

      Returns and empty sequence for padding, depending on the type of dataset.


.. py:function:: pad_dataset(dataset, global_batch_size, column_name)

   Pads the dataset to match a multiple of the global batch size.

   :param dataset: The dataset to pad.
   :param global_batch_size: The global batch size.
   :param column_name: The column name in the dataset.

   :returns: The padded dataset.


.. py:function:: load_array_record_dataset(dataset_path, file_extension = '.arecord')

   Take all files located at dataset_path and load it as grain.ArrayRecordDataSource.

   Assumes that the filenames are multiple shards where the shard idx is in the filename, e.g. train_000001.arecord'.
   We load the files in the order of the shard idx.

   :param dataset_path: Path to the dataset folder, which contains .arecord files.
   :param file_extension: The file extension of the dataset files. Default is '.arecord'.

   :returns: The dataset as grain.ArrayRecordDataSource.
   :rtype: grain.ArrayRecordDataSource


.. py:function:: load_huggingface_dataset(config)

   Load a dataset from HuggingFace.

   :param config: The HFHubDataConfig object.

   :returns: The loaded dataset.
   :rtype: datasets.Dataset


