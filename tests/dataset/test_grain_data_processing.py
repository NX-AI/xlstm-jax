from pathlib import Path

from scripts.hf_to_arrayrecord import convert_dataset
from tests.dataset.test_hf_data_processing import _load_n_batches, _setup_data as _hf_setup_data

import grain.python as grain
import numpy as np
import pytest
import transformers
from jax.sharding import Mesh

from xlstm_jax.dataset import grain_data_processing
from xlstm_jax.dataset.configs import GrainArrayRecordsDataConfig
from xlstm_jax.dataset.input_pipeline_interface import create_data_iterator
from xlstm_jax.dataset.multihost_dataloading import MultiHostDataLoadIterator
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models.configs import ParallelConfig


@pytest.mark.skipif(not pytest.grain_available, reason="Grain is not available.")
def test_array_records_identical_to_hf_dataset_and_loader(tmp_path: Path):
    """Test that pure-grain array_records dataset and dataloader pipeline are identical to hf dataset and loader."""

    # Convert Wiki dataset to ArrayRecords.
    shard_size = 2000  # Small shard size such that even wiki is stored in multiple files (for testing).
    hf_path = "Salesforce/wikitext"
    hf_data_dir = "wikitext-2-v1"
    base_out_path = tmp_path / "array_records" / f"shardsize_{shard_size}"

    convert_dataset(
        hf_path=hf_path,
        hf_data_name=None,
        hf_data_dir=hf_data_dir,
        hf_cache_dir=tmp_path / "hf_cache",
        splits=["train", "validation"],
        num_processes=2,
        base_out_path=base_out_path,
        shard_size=shard_size,
    )

    # Next, get dataset and iterators for both grain and hf.
    batch_size_per_device = 8
    context_length = 128

    _, _, grain_train_ds, grain_eval_ds, grain_train_iterator, grain_eval_iterator, grain_tokenizer = _grain_setup_data(
        data_path=base_out_path / hf_path.replace("/", "_") / hf_data_dir,
        batch_size_per_device=batch_size_per_device,
        context_length=context_length,
    )

    _, _, hf_train_ds, hf_eval_ds, hf_train_iterator, hf_eval_iterator, hf_tokenizer = _hf_setup_data(
        tmp_path=tmp_path,
        batch_size_per_device=batch_size_per_device,
        context_length=context_length,
    )

    # Check dataset lengths are identical.
    assert len(grain_train_ds) == len(hf_train_ds), "Train datasets have different number of examples."
    assert len(grain_eval_ds) == len(hf_eval_ds), "Eval datasets have different number of examples."

    # Check train dataset examples are identical.
    for i in range(len(grain_train_ds)):
        grain_example = grain_train_ds[i].decode()  # array_records stored as bytestring in dict, i.e. b'abc'
        hf_example = hf_train_ds[i]["text"]  # HF stores dict[str, str], i.e. {'text': 'abc'}
        assert grain_example == hf_example, f"Example {i} is different in the two datasets."

    # Check eval dataset examples are identical.
    for i in range(len(grain_eval_ds)):
        grain_example = grain_eval_ds[i].decode()
        hf_example = hf_eval_ds[i]["text"]
        assert grain_example == hf_example, f"Example {i} is different in the two datasets."

    # Check dataloaders (using preprocessing pipeline) load identical batches.
    max_batches = 100  # validation has less than 100 batches for wiki v2. So we also test loading entire eval iterator.
    grain_train_batches = _load_n_batches(grain_train_iterator, max_batches=max_batches)
    grain_eval_batches = _load_n_batches(grain_eval_iterator, max_batches=max_batches)
    hf_train_batches = _load_n_batches(hf_train_iterator, max_batches=max_batches)
    hf_eval_batches = _load_n_batches(hf_eval_iterator, max_batches=max_batches)
    assert len(grain_train_batches) == len(hf_train_batches), "Different number of train batches loaded."
    assert len(grain_eval_batches) == len(hf_eval_batches), "Different number of eval batches loaded."

    for i, (grain_batch, hf_batch) in enumerate(zip(grain_train_batches, hf_train_batches)):
        for field in hf_batch.__dict__.keys():  # inputs, targets, input_segmentation, etc.
            assert np.all(getattr(grain_batch, field) == getattr(hf_batch, field)), f"{field} is different in batch {i}"

    for i, (grain_batch, hf_batch) in enumerate(zip(grain_eval_batches, hf_eval_batches)):
        for field in hf_batch.__dict__.keys():  # inputs, targets, input_segmentation, etc.
            assert np.all(getattr(grain_batch, field) == getattr(hf_batch, field)), f"{field} is different in batch {i}"


def _grain_setup_data(
    data_path, batch_size_per_device: int = 2, context_length: int = 128
) -> tuple[
    ParallelConfig,
    Mesh,
    grain.ArrayRecordDataSource,
    grain.ArrayRecordDataSource,
    MultiHostDataLoadIterator,
    MultiHostDataLoadIterator,
    transformers.AutoTokenizer,
]:
    """Helper function to get data iterators, datasets, mesh, and tokenizer for
    testing.

    Note: We have a lot of code duplication compared to the helper function in
    tests/test_hf_data_processing. But here we later want to check
    different datasets. So keep it separate for now.

    Args:
        data_path: path to the dataset folder, which contains train and validation subfolder with .arecord files.
        batch_size_per_device: The (local) batch size per device.
        context_length: The context/sequence length.

    Returns:
        Tuple of parallel config, mesh, train_ds, eval_ds, train_iterator, eval_iterator, tokenizer.
    """

    # Define data configuration.
    parallel = ParallelConfig(
        data_axis_name="dp",
        fsdp_axis_name="fsdp",
        model_axis_name="tp",
        pipeline_axis_name="pp",
        fsdp_modules=["Embed", "LMHead", "mLSTMBlock"],
        fsdp_min_weight_size=2**8,
        fsdp_axis_size=1,
        model_axis_size=1,
        data_axis_size=-1,
    )

    # Initialize mesh.
    mesh = initialize_mesh(init_distributed_on_slurm=False, parallel_config=parallel)

    data_config = GrainArrayRecordsDataConfig(
        num_train_epochs=2,
        global_batch_size=batch_size_per_device * mesh.shape[parallel.data_axis_name],
        max_target_length=context_length,
        data_path=data_path,
        train_data_column="text",
        eval_data_column="text",
        train_split="train",
        eval_split="validation",
        tokenize_train_data=True,
        tokenize_eval_data=True,
        tokenizer_path="gpt2",
        add_bos=False,
        add_eos=False,
        add_eod=True,
        grain_packing=True,
    )

    train_path = data_config.data_path / data_config.train_split
    train_ds = grain_data_processing.load_array_record_dataset(dataset_path=train_path)

    eval_path = data_config.data_path / data_config.eval_split
    eval_ds = grain_data_processing.load_array_record_dataset(dataset_path=eval_path)

    # Define iterators
    train_iterator, eval_iterator = create_data_iterator(config=data_config, mesh=mesh)

    # Get the tokenizer that is used for the train_iterator and eval_iterator. We need it for testing.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        data_config.tokenizer_path,
        clean_up_tokenization_spaces=False,  # See https://github.com/huggingface/transformers/issues/31884
        legacy=False,
        token=None,
        use_fast=True,
        add_bos=data_config.add_bos,
        add_eos=data_config.add_eos,
        cache_dir=data_config.hf_cache_dir,
    )
    return parallel, mesh, train_ds, eval_ds, train_iterator, eval_iterator, tokenizer
