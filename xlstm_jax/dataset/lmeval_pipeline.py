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
from xlstm_jax.dataset.hf_data_processing import load_tokenizer

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
    ...         {"inputs": np.array([[1, 2]]), "targets": np.array([[1, 2]]), "idx": np.array([0])}),
    ...     {"inputs": np.array([[1, 2]]), "targets": np.array([[1, 2]]),
    ...      "document_idx": np.array([0]), "inputs_position": np.array([[0, 1]]),
    ...      "targets_position": np.array([[0, 1]]), "sequence_idx": np.array([0])})
    """

    def map(self, item: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Converts an incomplete dict to a dictionary with all components for an LLMIndexedBatch
        """
        res = item.copy()
        batch_size = res["inputs"].shape[0]
        if "sequence_idx" not in res:
            res["sequence_idx"] = np.zeros([batch_size], dtype=np.int32)
        if "idx" in res:
            res["document_idx"] = np.asarray(res["idx"], dtype=np.int32)
            del res["idx"]
        if "inputs_position" not in res:
            res["inputs_position"] = np.arange(item["inputs"].shape[1])[None, :].repeat(batch_size, 1)
        if "targets_position" not in res:
            res["targets_position"] = np.arange(item["inputs"].shape[1])[None, :].repeat(batch_size, 1)
        return res


def token_length(item: dict[str, np.ndarray]) -> int:
    """
    Get the token length of a data item for sorting (grouping) the dataset.

    Args:
        item: A dataset item that contains an inputs element.

    Returns:
        Length of the inputs element.

    >>> token_length({"inputs": np.array([[1, 2, 3]])})
    3
    """
    return item["inputs"].shape[1]


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
            int: Length of the dataset
        """
        return len(self.dataset)


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
        else:
            return np.stack(batch, axis=0)
    if axis >= batch[0].ndim:
        if batch_size_pad is not None:
            return np.pad(np.concatenate(batch, axis=0), ((0, batch_size_pad - len(batch)),), constant_values=pad_value)
        else:
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
    Also, it just support recurrent models that can take infinite sequence lengths.
    For sequence_length limited models use the `HFTokenizeLogLikelihoodRolling` from
    `lmeval_dataset.py`.

    Args:
        dataloading_host_index: The index of the dataloading host. Will be used to select the
            correct shard of the dataset. In JAX, this is equivalent to :func:`jax.process_index()`.
        dataloading_host_count: The number of dataloading hosts. Will be used to determine the
            shard size. In JAX, this is equivalent to :func:`jax.process_count()`.
        global_mesh: The global mesh to shard the data over.
        dataset: The dataset to load. Should provide a `__getitem__` method to access elements.

    Returns:
        MultiHostDataLoadIterator for the lmeval dataset.
    """
    grain_dataset = grain.MapDataset.source(dataset).map_with_index(lambda idx, x: {"idx": idx, "req": x})
    dataset_size = len(grain_dataset)
    slice_size = dataset_size // dataloading_host_count
    slice_remainder = dataset_size % dataloading_host_count
    host_slice_start = dataloading_host_index * slice_size + min(dataloading_host_index, slice_remainder)
    host_slice_end = host_slice_start + slice_size + (1 if dataloading_host_index < slice_remainder else 0)
    host_slice = slice(host_slice_start, host_slice_end)
    grain_dataset = grain_dataset.slice(host_slice)
    LOGGER.info(f"Host {dataloading_host_index} has slice {host_slice} with global dataset size {dataset_size}.")

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

    # Sort dataset for token sequence length, enables minimal padding, make it longest first
    grain_dataset = SortedDataset(grain_dataset, key=token_length, reverse=True)

    def batch_pad(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {
            key: _pad_batch_multiple(
                [item[key] for item in batch],
                multiple_of=padding_multiple,
                batch_size_pad=global_batch_size // dataloading_host_count,
                axis=1,
            )
            for key in batch[0]
        }

    grain_dataset = grain_dataset.batch(
        global_batch_size // dataloading_host_count, drop_remainder=False, batch_fn=batch_pad
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
