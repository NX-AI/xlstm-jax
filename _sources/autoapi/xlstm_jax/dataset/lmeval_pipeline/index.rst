xlstm_jax.dataset.lmeval_pipeline
=================================

.. py:module:: xlstm_jax.dataset.lmeval_pipeline


Attributes
----------

.. autoapisummary::

   xlstm_jax.dataset.lmeval_pipeline.LOGGER


Classes
-------

.. autoapisummary::

   xlstm_jax.dataset.lmeval_pipeline.ParseLMEval
   xlstm_jax.dataset.lmeval_pipeline.CompleteLLMIndexedBatch
   xlstm_jax.dataset.lmeval_pipeline.PadBatchDataset
   xlstm_jax.dataset.lmeval_pipeline.PadSequenceInBatchDataset
   xlstm_jax.dataset.lmeval_pipeline.SortedDataset
   xlstm_jax.dataset.lmeval_pipeline.MultihostSortedRemapDataset


Functions
---------

.. autoapisummary::

   xlstm_jax.dataset.lmeval_pipeline.empty_llm_indexed_sample
   xlstm_jax.dataset.lmeval_pipeline.token_length
   xlstm_jax.dataset.lmeval_pipeline._pad_batch_multiple
   xlstm_jax.dataset.lmeval_pipeline.lmeval_preprocessing_pipeline


Module Contents
---------------

.. py:data:: LOGGER

.. py:class:: ParseLMEval(request_name = 'req')

   Bases: :py:obj:`grain.python.MapTransform`


   Parses an LMEval request into a simple dictionary format with prefix and text.
   If there is no prefix, it is simply an empty string.

   :param request_name: The key in the input dictionary which corresponds to the LMEval Request.


   .. py:attribute:: request_name
      :value: 'req'



   .. py:method:: map(item)

      Maps a single request to a dictionary of prefix and text.

      :param dict[str: LMEval request instance in a dictionary with the index
      :param int | Instance]: LMEval request instance in a dictionary with the index

      :returns: Resulting item dictionary
      :rtype: dict[str, str | int]

      >>> from xlstm_jax.utils.pytree_utils import pytree_diff
      >>> pytree_diff(ParseLMEval().map(
      ...     {"idx": 1,
      ...      "req": Instance(
      ...         request_type="loglikelihood_rolling", doc={},
      ...         idx=0, arguments=("Prefix", "Main"))}),
      ...     {"idx": 1, "prefix": "Prefix", "text": "Main"})



.. py:class:: CompleteLLMIndexedBatch

   Bases: :py:obj:`grain.python.MapTransform`


   Grain Transform that uses an indexed dataset (with "idx") and fills it towards all
   components of a LLMIndexedBatch.

   >>> from xlstm_jax.utils.pytree_utils import pytree_diff
   >>> pytree_diff(
   ...     CompleteLLMIndexedBatch().map(
   ...         {"inputs": np.array([[1, 2]]), "targets": np.array([[1, 2]]), "idx": np.array(0)}),
   ...     {"inputs": np.array([[1, 2]]), "targets": np.array([[1, 2]]),
   ...      "document_idx": np.array([1]), "inputs_position": np.array([[0, 1]]),
   ...      "targets_position": np.array([[0, 1]]), "sequence_idx": np.array([0]),
   ...      "_document_borders": np.array([[False, False]])})


   .. py:method:: map(item)
      :staticmethod:


      Converts an incomplete dict to a dictionary with all components for an LLMIndexedBatch



.. py:function:: empty_llm_indexed_sample()

   Generator for an empty llm_indexed sample that is used in paddings.
   This creates just the data for a single sample not a full batch object.

   :returns: An empty / padding sample for an LLMIndexedBatch


.. py:function:: token_length(item)

   Get the token length of a data item for sorting (grouping) the dataset.

   :param item: A dataset item that contains an `inputs` element.

   :returns: Length of the `inputs` element.

   >>> token_length({"inputs": np.array([[1, 2, 3]])})
   3


.. py:class:: PadBatchDataset(dataset, multiple_of, min_length, pad_elem)

   Bases: :py:obj:`grain.python.MapDataset`


   Creates a dataset that has only full batches by adding padding elements.

   :param dataset: The existing dataset.
   :param multiple_of: Global batch size to be padded towards.
   :param min_length: Minimum (padded) length/size of the dataset.
   :param pad_elem: Empty element to be appended.

   >>> PadBatchDataset(grain.MapDataset.source([3, 1, 2]), multiple_of=4, min_length=0, pad_elem=0)[3]
   0
   >>> len(PadBatchDataset(grain.MapDataset.source([3, 1, 2]), multiple_of=4, min_length=0, pad_elem=0))
   4


   .. py:attribute:: dataset


   .. py:attribute:: multiple_of


   .. py:attribute:: min_length


   .. py:attribute:: pad_elem


.. py:function:: _pad_batch_multiple(batch, multiple_of = 64, axis = 1, pad_value = 0, batch_size_pad = None)

   Pads a list of arrays to a common length defined as a multiple of a pad_mulitple value, with a certain value.
   Then the arrays are concatenated along axis 0.

   :param batch: A list of np.ndarrays to be padded and concatenated
   :param multiple_of: A number that the padded size should be a multiple of.
   :param axis: The axis to be padded, typically a sequence dimension.
   :param pad_value: The padding value, typically zero.

   :returns: The concatenated, padded batch.

   >>> np.allclose(
   ...     _pad_batch_multiple([np.array([[1, 2]]), np.array([[11, 12, 13,]])], multiple_of=4, axis=1),
   ...     np.array([[1, 2, 0, 0], [11, 12, 13, 0]]))
   True


.. py:class:: PadSequenceInBatchDataset(dataset, batch_size, multiple_of = 64, pad_value = 0)

   Bases: :py:obj:`grain.python.MapDataset`


   Creates a dataset that has only full batches by padding elements.

   This pads single elements (no batches) enabling having distributed batches over more devices.
   Assumes a dataset that consists of a flat dictionary of arrays.

   :param dataset: The existing dataset, items are assumed to be dicts of array.
   :param batch_size: The batch size to pad towards.
   :param multiple_of: A number that the padded size should be a multiple of.
   :param pad_value: Value to pad with.

   >>> from xlstm_jax.utils.pytree_utils import pytree_diff
   >>> pytree_diff(list(PadSequenceInBatchDataset(grain.MapDataset.source(
   ...     [{"a": np.array([[3, 1, 5]])},
   ...      {"a": np.array([[2, 4]])},
   ...      {"a": np.array([[2, 4, 3, 3]])},
   ...      {"a": np.array([[2, 4]])}]
   ... ), batch_size=2, multiple_of=3 )), [
   ...     {"a": np.array([[3, 1, 5]])},
   ...     {"a": np.array([[2, 4, 0]])},
   ...     {"a": np.array([[2, 4, 3, 3, 0, 0]])},
   ...     {"a": np.array([[2, 4, 0, 0, 0, 0]])}])


   .. py:attribute:: dataset


   .. py:attribute:: batch_size


   .. py:attribute:: multiple_of
      :value: 64



   .. py:attribute:: pad_value
      :value: 0



.. py:class:: SortedDataset(dataset, key, reverse = False)

   Bases: :py:obj:`grain.python.MapDataset`


   Creates a sorted dataset based on a key (applied to all items) and an existing dataset.

   :param dataset: The existing dataset
   :param key: Key Function to be applied for sorting
   :param reverse: If the sorting should be ascending (False, default) or descending

   >>> SortedDataset(grain.MapDataset.source([3, 1, 2]), lambda x: x)[0]
   1
   >>> SortedDataset(grain.MapDataset.source([3, 1, 2]), lambda x: x, reverse=True)[0]
   3
   >>> SortedDataset(grain.MapDataset.source(
   ...     [{"inputs": np.array([[1, 2, 3]])},
   ...      {"inputs": np.array([[1, 2]])}]), token_length)[0]
   {'inputs': array([[1, 2]])}


   .. py:attribute:: key


   .. py:attribute:: dataset


.. py:class:: MultihostSortedRemapDataset(dataset, global_batch_size, dataloader_host_count)

   Bases: :py:obj:`grain.python.MapDataset`


   This implements an index re-shuffling for a SortedDataset.
   The problem:
   Given a sorted dataset, and multi-host dataloaders using .slice,
   the sorting is broken.
   Examplary dataset:
   [1 2 3 4 5 6 7 8]

   Multi-host (standard slicing - assumed as input):
   [1 2 3 4] [5 6 7 8]
   Multi-host batched:
   [[1 2] [3 4]] [[5 6] [7 8]]

   What we want actually for proper batching:
   [[1 2] [5 6]] [[3 4] [7 8]]
   such that the global batch still looks like:
   [[1 2 3 4] [5 6 7 8]]

   :param dataset: Original (sorted) dataset of which the order within batches should be kept.
   :param global_batch_size: The global batch size.
   :param dataloader_host_count: The number of dataloaders that a global batch is created from.

   >>> ds = MultihostSortedRemapDataset(
   ...     grain.MapDataset.source([1, 2, 3, 4, 5, 6, 7, 8]),
   ...     global_batch_size=4, dataloader_host_count=2)
   >>> host_slices = [slice(0, 4), slice(4, 8)]
   >>> [list(ds.slice(host_slices[0]).batch(2)), list(ds.slice(host_slices[1]).batch(2))]
   [[array([1, 2]), array([5, 6])], [array([3, 4]), array([7, 8])]]


   .. py:attribute:: dataset


   .. py:attribute:: global_batch_size


   .. py:attribute:: dataloader_host_count


.. py:function:: lmeval_preprocessing_pipeline(dataloading_host_index, dataloading_host_count, global_mesh, dataset, global_batch_size, tokenizer_path, hf_access_token = None, tokenizer_cache_dir = None, eos_token_id = None, bos_token_id = None, worker_count = 1, worker_buffer_size = 1, padding_multiple = 128, use_thread_prefetch = False)

   Create a mult-host dataloader for LMEval datasets for loglikelihood and
   loglikelihood_rolling tasks. This does not support generation tasks currently.
   Also, it just support recurrent models that can take infinite sequence lengths.
   For sequence_length limited models use the `HFTokenizeLogLikelihoodRolling` from
   `lmeval_dataset.py`.

   :param dataloading_host_index: The index of the dataloading host. Will be used to select the
                                  correct shard of the dataset. In JAX, this is equivalent to :func:`jax.process_index()`.
   :param dataloading_host_count: The number of dataloading hosts. Will be used to determine the
                                  shard size. In JAX, this is equivalent to :func:`jax.process_count()`.
   :param global_mesh: The global mesh to shard the data over.
   :param dataset: The dataset to load. Should provide a `__getitem__` method to access elements.
   :param global_batch_size: The global batch size.
   :param tokenizer_path: Path to the tokenizer.
   :param hf_access_token: The access token for HuggingFace.
   :param tokenizer_cache_dir: The cache directory for the tokenizer.
   :param eos_token_id: The token ID to use for the end-of-sequence token. If `tokenizer_path` is
                        provided, the tokenizer's EOS token ID is used.
   :param bos_token_id: The token ID to use for the beginning-of-sequence token. If `tokenizer_path` is
                        provided, the tokenizer's BOS token ID is used.
   :param worker_count: The number of workers to use. In grain, a single worker is usually
                        sufficient, as the data loading is done in parallel across hosts.
   :param worker_buffer_size: The buffer size for the workers.
   :param padding_multiple: Pad to size being a multiple of.
   :param use_thread_prefetch: Use thread prefetching instead of multiprocessing.

   :returns: MultiHostDataLoadIterator for the lmeval dataset.


