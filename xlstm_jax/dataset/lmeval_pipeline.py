import logging
from collections.abc import Callable
from typing import Any

import grain.python as grain
import numpy as np
from grain._src.python.dataset.transformations.prefetch import ThreadPrefetchIterDataset
from jax.sharding import Mesh
from lm_eval.api.instance import Instance

from xlstm_jax.dataset import grain_transforms
from xlstm_jax.dataset.batch import LLMIndexedBatch
from xlstm_jax.dataset.hf_tokenizer import load_tokenizer

from .multihost_dataloading import MultiHostDataLoadIterator

LOGGER = logging.getLogger(__name__)


class ParseLMEval(grain.MapTransform):
    """
    Parses an LMEval request into a simple dictionary format with prefix and text.
    If there is no prefix, it is simply an empty string.

    Args:
        request_name: The key in the input dictionary which corresponds to the LMEval Request.
    """

    def __init__(self, request_name: str = "req"):
        self.request_name = request_name

    def map(self, item: dict[str, int | Instance]) -> dict[str, int | str]:
        """
        Maps a single request to a dictionary of prefix and text.

        Args:
            dict[str, int | Instance]: LMEval request instance in a dictionary with the index

        Returns:
            dict[str, str | int]: Resulting item dictionary

        >>> from xlstm_jax.utils import pytree_diff
        >>> pytree_diff(ParseLMEval().map(
        ...     {"idx": 1,
        ...      "req": Instance(
        ...         request_type="loglikelihood_rolling", doc={},
        ...         idx=0, arguments=("Prefix", "Main"))}),
        ...     {"idx": 1, "prefix": "Prefix", "text": "Main"})
        """
        if len(item[self.request_name].args) == 1:
            prefix = ""
            text = item[self.request_name].args[0]
        else:
            prefix = item[self.request_name].args[0]
            text = item[self.request_name].args[1]

        return {"prefix": prefix, "text": text, **{key: val for key, val in item.items() if key != self.request_name}}


class CompleteLLMIndexedBatch(grain.MapTransform):
    """
    Grain Transform that uses an indexed dataset (with "idx") and fills it towards all
    components of a LLMIndexedBatch.

    >>> from xlstm_jax.utils import pytree_diff
    >>> pytree_diff(
    ...     CompleteLLMIndexedBatch().map(
    ...         {"inputs": np.array([[1, 2]]), "targets": np.array([[1, 2]]), "idx": np.array(0)}),
    ...     {"inputs": np.array([[1, 2]]), "targets": np.array([[1, 2]]),
    ...      "document_idx": np.array([1]), "inputs_position": np.array([[0, 1]]),
    ...      "targets_position": np.array([[0, 1]]), "sequence_idx": np.array([0]),
    ...      "_document_borders": np.array([[False, False]])})
    """

    @staticmethod
    def map(item: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Converts an incomplete dict to a dictionary with all components for an LLMIndexedBatch
        """
        res = item.copy()
        batch_size = res["inputs"].shape[0]
        if "sequence_idx" not in res:
            res["sequence_idx"] = np.zeros([batch_size], dtype=np.int32)
        if "idx" in res:
            doc_idx = np.asarray(res["idx"] + 1, dtype=np.int32)
            if doc_idx.ndim == 0:
                res["document_idx"] = doc_idx[None]
            else:
                res["document_idx"] = doc_idx
            del res["idx"]
        if "inputs_position" not in res:
            res["inputs_position"] = np.arange(item["inputs"].shape[1])[None, :].repeat(batch_size, 1)
        if "targets_position" not in res:
            res["targets_position"] = np.arange(item["inputs"].shape[1])[None, :].repeat(batch_size, 1)
        if "_document_borders" not in res:
            res["_document_borders"] = np.zeros_like(res["inputs"], dtype=bool)
        return res


def empty_llm_indexed_sample():
    return {
        "sequence_idx": np.zeros([1], dtype=np.int32),
        "document_idx": np.zeros([1], dtype=np.int32),
        "inputs": np.zeros([1, 1], dtype=np.int32),
        "inputs_segmentation": np.zeros([1, 1], dtype=np.int32),
        "targets": np.zeros([1, 1], dtype=np.int32),
        "targets_segmentation": np.zeros([1, 1], dtype=np.int32),
        "inputs_position": np.zeros([1, 1], dtype=np.int32),
        "targets_position": np.zeros([1, 1], dtype=np.int32),
        "_document_borders": np.zeros([1, 1], dtype=bool),
    }


def token_length(item: dict[str, np.ndarray]) -> int:
    """
    Get the token length of a data item for sorting (grouping) the dataset.

    Args:
        item: A dataset item that contains an `inputs` element.

    Returns:
        Length of the `inputs` element.

    >>> token_length({"inputs": np.array([[1, 2, 3]])})
    3
    """
    return item["inputs"].shape[1]


class PadBatchDataset(grain.MapDataset):
    """
    Creates a dataset that has only full batches by adding padding elements.

    Args:
        dataset: The existing dataset
        multiple_of:  Global batch size to be padded towards
        pad_elem:  Empty element to be appended

    >>> PadBatchDataset(grain.MapDataset.source([3, 1, 2]), multiple_of=4, pad_elem=0)[3]
    0
    >>> len(PadBatchDataset(grain.MapDataset.source([3, 1, 2]), multiple_of=4, pad_elem=0))
    4
    """

    def __init__(self, dataset: grain.MapDataset, multiple_of: int, pad_elem: Any):
        super().__init__(dataset)
        self.dataset = dataset
        self.multiple_of = multiple_of
        self.pad_elem = pad_elem

    def __getitem__(self, idx: int) -> Any:
        """
        Args:
            idx: Item index

        Returns:
            Dataset element from the PadBatchDataset
        """
        if idx < len(self.dataset):
            return self.dataset[idx]
        if idx < ((len(self.dataset) - 1) // self.multiple_of + 1) * self.multiple_of:
            return self.pad_elem
        raise IndexError

    def __len__(self):
        return ((len(self.dataset) - 1) // self.multiple_of + 1) * self.multiple_of


def _pad_batch_multiple(
    batch: list[np.ndarray],
    multiple_of: int = 64,
    axis: int = 1,
    pad_value: int | float = 0,
    batch_size_pad: int | None = None,
) -> np.ndarray:
    """
    Pads a list of arrays to a common length defined as a multiple of a pad_mulitple value, with a certain value.
    Then the arrays are concatenated along axis 0.

    Args:
        batch: A list of np.ndarrays to be padded and concatenated
        multiple_of: A number that the padded size should be a multiple of.
        axis: The axis to be padded, typically a sequence dimension.
        pad_value: The padding value, typically zero.

    Returns:
        The concatenated, padded batch.

    >>> np.allclose(
    ...     _pad_batch_multiple([np.array([[1, 2]]), np.array([[11, 12, 13,]])], multiple_of=4, axis=1),
    ...     np.array([[1, 2, 0, 0], [11, 12, 13, 0]]))
    True
    """
    if batch[0].ndim == 0:
        if batch_size_pad is not None:
            return np.pad(np.stack(batch, axis=0), ((0, batch_size_pad - len(batch)),), constant_values=pad_value)
        return np.stack(batch, axis=0)
    if axis >= batch[0].ndim:
        if batch_size_pad is not None:
            return np.pad(np.concatenate(batch, axis=0), ((0, batch_size_pad - len(batch)),), constant_values=pad_value)
        return np.concatenate(batch, axis=0)

    max_len = max(item.shape[axis] for item in batch)
    pad_len = ((max_len - 1) // multiple_of + 1) * multiple_of
    padded_batch = np.concatenate(
        [
            np.pad(
                item,
                ((0, 0), (0, pad_len - item.shape[axis])),
                constant_values=pad_value,
            )
            for item in batch
        ],
        axis=0,
    )
    if batch_size_pad:
        padded_batch = np.pad(
            padded_batch,
            ((0, batch_size_pad - padded_batch.shape[0]), *((0, 0) for _ in range(padded_batch.ndim - 1))),
            constant_values=pad_value,
        )
    return padded_batch


class PadSequenceInBatchDataset(grain.MapDataset):
    """
    Creates a dataset that has only full batches by padding elements.

    This pads single elements (no batches) enabling having distributed batches over more devices.
    Assumes a dataset that consists of a flat dictionary of arrays.

    Args:
        dataset: The existing dataset, items are assumed to be dicts of array.
        batch_size: The batch size to pad towards.
        multiple_of: A number that the padded size should be a multiple of.
        pad_value:  Value to pad with.

    >>> from xlstm_jax.utils.pytree_diff import pytree_diff
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
    """

    def __init__(self, dataset: grain.MapDataset, batch_size: int, multiple_of: int = 64, pad_value: int = 0):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.multiple_of = multiple_of
        self.pad_value = pad_value

    def __getitem__(self, idx: int) -> Any:
        """
        Currently this actually creates a full batch and returns the single element later on.

        Args:
            idx: Item index

        Returns:
            Dataset element from the SplitBatchDataset
        """
        return {
            key: _pad_batch_multiple(
                [
                    self.dataset[(idx // self.batch_size) * self.batch_size + sub_idx][key]
                    for sub_idx in range(self.batch_size)
                ],
                multiple_of=self.multiple_of,
                axis=1,
                pad_value=self.pad_value,
                batch_size_pad=None,
            )[idx % self.batch_size : idx % self.batch_size + 1]
            if isinstance(self.dataset[idx][key], np.ndarray)
            else self.dataset[idx][key]
            for key in self.dataset[idx]
        }

    def __len__(self):
        return len(self.dataset)


class SortedDataset(grain.MapDataset):
    """
    Creates a sorted dataset based on a key (applied to all items) and an existing dataset.

    Args:
        dataset: The existing dataset
        key:  Key Function to be applied for sorting
        reverse: If the sorting should be ascending (False, default) or descending

    >>> SortedDataset(grain.MapDataset.source([3, 1, 2]), lambda x: x)[0]
    1
    >>> SortedDataset(grain.MapDataset.source([3, 1, 2]), lambda x: x, reverse=True)[0]
    3
    >>> SortedDataset(grain.MapDataset.source(
    ...     [{"inputs": np.array([[1, 2, 3]])},
    ...      {"inputs": np.array([[1, 2]])}]), token_length)[0]
    {'inputs': array([[1, 2]])}
    """

    def __init__(self, dataset: grain.MapDataset, key: Callable, reverse: bool = False):
        super().__init__(dataset)
        self.key = key
        self.dataset = sorted(dataset, key=key, reverse=reverse)

    def __getitem__(self, idx: int) -> Any:
        """
        Args:
            idx: Item index

        Returns:
            Dataset element from the sorted dataset
        """
        return self.dataset[idx]

    def __len__(self) -> int:
        """Dataset length

        Returns:
            Length of the dataset
        """
        return len(self.dataset)


class MultihostSortedRemapDataset(grain.MapDataset):
    """
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

    Args:
        dataset: Original (sorted) dataset of which the order within batches should be kept.
        global_batch_size: The global batch size.
        dataloader_host_count: The number of dataloaders that a global batch is created from.

    >>> ds = MultihostSortedRemapDataset(
    ...     grain.MapDataset.source([1, 2, 3, 4, 5, 6, 7, 8]),
    ...     global_batch_size=4, dataloader_host_count=2)
    >>> host_slices = [slice(0, 4), slice(4, 8)]
    >>> [list(ds.slice(host_slices[0]).batch(2)), list(ds.slice(host_slices[1]).batch(2))]
    [[array([1, 2]), array([5, 6])], [array([3, 4]), array([7, 8])]]
    """

    def __init__(self, dataset: grain.MapDataset, global_batch_size: int, dataloader_host_count: int):
        super().__init__(dataset)
        self.dataset = dataset
        self.global_batch_size = global_batch_size
        self.dataloader_host_count = dataloader_host_count

    def __getitem__(self, idx: int) -> Any:
        """
        Args:
            idx: Item index

        Returns:
            Dataset element from the mapped idx.
        """
        local_batch_size = self.global_batch_size // self.dataloader_host_count
        dataloader_internal_idx = idx % (len(self.dataset) // self.dataloader_host_count)
        host_idx = idx // (len(self.dataset) // self.dataloader_host_count)
        local_batch_subidx = dataloader_internal_idx % local_batch_size
        local_batch_idx = dataloader_internal_idx // local_batch_size
        return self.dataset[local_batch_idx * self.global_batch_size + host_idx * local_batch_size + local_batch_subidx]

    def __len__(self) -> int:
        """Dataset length

        Returns:
            Length of the dataset
        """
        return len(self.dataset)


def lmeval_preprocessing_pipeline(
    dataloading_host_index: int,
    dataloading_host_count: int,
    global_mesh: Mesh,
    dataset: list[Instance],
    global_batch_size: int,
    tokenizer_path: str,
    hf_access_token: str | None = None,
    tokenizer_cache_dir: str | None = None,
    eos_token_id: int | None = None,
    bos_token_id: int | None = None,
    worker_count: int = 1,
    worker_buffer_size: int = 1,
    padding_multiple: int = 128,
    use_thread_prefetch: bool = False,
) -> MultiHostDataLoadIterator:
    """
    Create a mult-host dataloader for LMEval datasets for loglikelihood and
    loglikelihood_rolling tasks. This does not support generation tasks currently.
    Also, it just supports recurrent models that can take infinite sequence lengths.
    For sequence_length limited models use the `HFTokenizeLogLikelihoodRolling` from
    `lmeval_dataset.py`.
    Internal Operation:
    The dataset is fully loaded on all hosts / workers, sorted by sequence length,
    padded in batch and sequence length. Only then it is sharded for a combined global
    batch that has consistent sequence length over all hosts.

    Args:
        dataloading_host_index: The index of the dataloading host. Will be used to select the
            correct shard of the dataset. In JAX, this is equivalent to :func:`jax.process_index()`.
        dataloading_host_count: The number of dataloading hosts. Will be used to determine the
            shard size. In JAX, this is equivalent to :func:`jax.process_count()`.
        global_mesh: The global mesh to shard the data over.
        dataset: The dataset to load. Should provide a `__getitem__` method to access elements.
        global_batch_size: The global batch size.
        tokenizer_path: Path to the tokenizer.
        hf_access_token: The access token for HuggingFace.
        tokenizer_cache_dir: The cache directory for the tokenizer.
        eos_token_id: The token ID to use for the end-of-sequence token. If `tokenizer_path` is
            provided, the tokenizer's EOS token ID is used.
        bos_token_id: The token ID to use for the beginning-of-sequence token. If `tokenizer_path` is
            provided, the tokenizer's BOS token ID is used.
        worker_count: The number of workers to use. In grain, a single worker is usually
            sufficient, as the data loading is done in parallel across hosts.
        worker_buffer_size: The buffer size for the workers.
        padding_multiple: Pad to size being a multiple of.
        use_thread_prefetch: Use thread prefetching instead of multiprocessing.

    Returns:
        MultiHostDataLoadIterator for the lmeval dataset.
    """
    grain_dataset = grain.MapDataset.source(dataset).map_with_index(lambda idx, x: {"idx": idx, "req": x})
    dataset_size = len(grain_dataset)
    # do NOT! slice dataset here as single sequences have to have the same padded length for proper batching

    if tokenizer_path is not None:
        tokenizer = load_tokenizer(tokenizer_path, hf_access_token, tokenizer_cache_dir)
        LOGGER.info(f"Loaded tokenizer from {tokenizer_path} with vocab size {tokenizer.vocab_size}.")
        if eos_token_id is not None and eos_token_id != tokenizer.eos_token_id:
            LOGGER.warning(
                "EOS token ID has been set explicitly and differs from the tokenizer's EOS token ID: "
                f"tokenizer={tokenizer.eos_token_id}, provided={eos_token_id}. Using provided EOS token ID."
            )
        else:
            eos_token_id = tokenizer.eos_token_id
            LOGGER.info(f"Using EOS token ID: {eos_token_id}.")
    else:
        tokenizer = None

    operations = [
        ParseLMEval(request_name="req"),
        grain_transforms.HFPrefixTokenize(
            tokenizer=tokenizer,
            prefix_tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
        ),
        CompleteLLMIndexedBatch(),
    ]

    # Apply operations.
    if operations is not None:
        for operation in operations:
            grain_dataset = grain_dataset.map(operation)

    grain_dataset = PadBatchDataset(grain_dataset, multiple_of=global_batch_size, pad_elem=empty_llm_indexed_sample())

    # Sort dataset wrt. token sequence length (longest sequence first), enables minimal padding
    grain_dataset = SortedDataset(grain_dataset, key=token_length, reverse=True)

    grain_dataset = PadSequenceInBatchDataset(
        grain_dataset, batch_size=global_batch_size, multiple_of=padding_multiple, pad_value=0
    )

    # Re-Shuffle indices such that ordering is kept inside a global batch
    grain_dataset = MultihostSortedRemapDataset(
        grain_dataset, global_batch_size=global_batch_size, dataloader_host_count=dataloading_host_count
    )

    def batch_concatenate(inp: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {key: np.concatenate([elem[key] for elem in inp], axis=0) for key in inp[0]}

    # Moved from the early sequence processing to this point after tokenization+padding+batching
    dataset_size = len(grain_dataset)
    slice_size = dataset_size // dataloading_host_count
    slice_remainder = dataset_size % dataloading_host_count
    assert slice_remainder == 0, "Padding should have taken care of remaining sliced samples"
    host_slice_start = dataloading_host_index * slice_size
    host_slice_end = host_slice_start + slice_size
    host_slice = slice(host_slice_start, host_slice_end)
    grain_dataset = grain_dataset.slice(host_slice)
    LOGGER.info(f"Host {dataloading_host_index} has slice {host_slice} with global dataset size {dataset_size}.")
    assert slice_remainder == 0, (
        f"Bad Padding, the should actually be no leftover slice ds: {dataset_size}, "
        f"sl: {slice_size}, dl: {dataloading_host_count}"
    )

    grain_dataset = grain_dataset.batch(
        global_batch_size // dataloading_host_count, drop_remainder=False, batch_fn=batch_concatenate
    )
    grain_dataset = grain_dataset.map(grain_transforms.InferSegmentations(eod_token_id=eos_token_id))

    iterator_length = (len(dataset) - 1) // global_batch_size + 1

    # Create LLMBatch objects.
    grain_dataset = grain_dataset.map(grain_transforms.CollateToBatch(batch_class=LLMIndexedBatch))

    # Setup prefetching with thread or multiprocessing.
    grain_dataset = grain_dataset.to_iter_dataset()
    multiprocessing_options = None
    if use_thread_prefetch:
        LOGGER.info("Using thread prefetching.")
        grain_dataset = ThreadPrefetchIterDataset(
            grain_dataset,
            prefetch_buffer_size=int(worker_buffer_size * worker_count),
        )
    elif worker_count > 0:
        LOGGER.info(f"Using multiprocessing with {worker_count} workers.")
        multiprocessing_options = grain.MultiprocessingOptions(
            num_workers=worker_count,
            per_worker_buffer_size=worker_buffer_size,
        )
        # NOTE: At the moment this gives a warning about being deprecated. In the nightly version of
        # grain, this will be under the name `grain_dataset.mp_prefetch`.
        grain_dataset = grain_dataset.prefetch(multiprocessing_options)

    multihost_gen = MultiHostDataLoadIterator(
        grain_dataset,
        global_mesh,
        iterator_length=iterator_length,
        dataset_size=len(dataset),
        reset_after_epoch=True,
    )

    return multihost_gen
