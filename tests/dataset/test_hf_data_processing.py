from pathlib import Path

import datasets
import jax.numpy as jnp
import numpy as np
import pytest
import transformers
from datasets import Dataset
from jax.sharding import Mesh, PartitionSpec as P

from xlstm_jax.dataset import HFHubDataConfig, create_data_iterator
from xlstm_jax.dataset.batch import LLMBatch
from xlstm_jax.dataset.multihost_dataloading import MultiHostDataLoadIterator
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models.configs import ParallelConfig


@pytest.mark.skipif(not pytest.grain_available, reason="Grain is not available.")
def test_hf_dataset_with_group_texts(tmp_path: Path):
    """Test data loading with huggingface datasets, using group_texts applied
    via arrow_dataset.map method."""
    fsdp_axis_size = min(2, pytest.num_devices)
    model_axis_size = min(2, pytest.num_devices // fsdp_axis_size)

    parallel = ParallelConfig(
        data_axis_name="dp",
        fsdp_axis_name="fsdp",
        model_axis_name="tp",
        pipeline_axis_name="pp",
        fsdp_modules=("Embed", "LMHead", "mLSTMBlock"),
        fsdp_min_weight_size=2**8,
        fsdp_axis_size=fsdp_axis_size,
        model_axis_size=model_axis_size,
        data_axis_size=-1,
    )

    # Initialize mesh.
    mesh = initialize_mesh(init_distributed_on_slurm=False, parallel_config=parallel)

    # Define data configuration.
    global_batch_size = 64
    context_length = 128
    train_config, eval_config = HFHubDataConfig.create_train_eval_configs(
        global_batch_size=global_batch_size,
        max_target_length=context_length,
        hf_path="Salesforce/wikitext",
        hf_data_dir="wikitext-2-v1",
        hf_cache_dir=tmp_path / "hf_cache",
        data_column="text",
        tokenize_data=True,
        tokenizer_path="gpt2",
        data_shuffle_seed=42,
        add_bos=True,
        add_eos=True,
        add_eod=True,
        grain_packing=False,
    )

    # Define iterators
    train_iterator = create_data_iterator(config=train_config, mesh=mesh)
    eval_iterator = create_data_iterator(config=eval_config, mesh=mesh)

    # Get batch from training iterator and make sure that the batch size is correct.
    train_batch = next(train_iterator)
    assert train_batch.inputs.shape == (global_batch_size, context_length)
    assert train_batch.inputs.sharding.spec == P(mesh.axis_names)
    assert not jnp.all(
        train_batch.inputs_segmentation == 1
    ), "Not all segmentations should not be 1, indicates the end-of-document tokens are missing."

    # Get batches from evaluation iterator and make sure that the batch size is correct,
    # and that the batches are the same for each epoch.
    eval_batches = list()
    last_eval_batch = None
    eval_dataset_size = len(eval_iterator)
    for epoch_idx in range(2):
        step = 0
        for batch_idx, batch in enumerate(eval_iterator):
            step += 1
            assert batch.inputs.shape == (global_batch_size, context_length)
            assert batch.inputs.sharding.spec == P(mesh.axis_names)
            if epoch_idx == 0:
                eval_batches.append(batch)
            else:
                ep0_batch = eval_batches[batch_idx]
                np.testing.assert_allclose(batch.inputs, ep0_batch.inputs)
                np.testing.assert_allclose(batch.targets, ep0_batch.targets)
        assert step == eval_dataset_size, "Should have the same number of steps for each epoch."
    assert len(eval_batches) == eval_dataset_size, "Should have the same number of batches for each epoch."
    last_eval_batch = eval_batches[-1]
    # For all attributes in the batch, we should have zeros.
    for attr_name in [
        "inputs",
        "targets",
        "inputs_segmentation",
        "targets_segmentation",
        "inputs_position",
        "targets_position",
    ]:
        val = getattr(last_eval_batch, attr_name)[-1]
        assert np.all(val == 0), f"Last batch should have all zeros for {attr_name}, but got {val}."
    assert not np.all(
        last_eval_batch.inputs[0] == 0
    ), "In the last batch, there should be at least one non-padded element."

    # Test max steps per epoch and a continuous validation iterator.
    eval_max_steps_per_epoch = len(eval_iterator) // 2
    eval_config.max_steps_per_epoch = eval_max_steps_per_epoch

    # Define iterators
    eval_iterator = create_data_iterator(config=eval_config, mesh=mesh)
    assert len(eval_iterator) == eval_max_steps_per_epoch, "Eval dataset have the specified number of steps."
    for epoch_idx in range(3):
        batch_idx = 0
        for batch in eval_iterator:
            np.testing.assert_allclose(
                batch.inputs,
                eval_batches[batch_idx].inputs,
                err_msg=f"Failed at batch idx: {batch_idx}, epoch idx: {epoch_idx}",
            )
            np.testing.assert_allclose(
                batch.targets,
                eval_batches[batch_idx].targets,
                err_msg=f"Failed at batch idx: {batch_idx}, epoch idx: {epoch_idx}",
            )
            batch_idx += 1
        assert (
            batch_idx == eval_max_steps_per_epoch
        ), f"Should have the same number of steps for each epoch, failed at {epoch_idx}."


@pytest.mark.skipif(not pytest.grain_available, reason="Grain is not available.")
def test_hf_dataset_with_packing(tmp_path: Path):
    """Test data loading with huggingface datasets, using grain
    PackAndBatchOperation."""

    batch_size_per_device = 8
    context_length = 128

    # Prepare the dataset and dataloaders for the test.
    parallel, mesh, train_ds, eval_ds, train_iterator, eval_iterator, tokenizer = _setup_data(
        tmp_path=tmp_path,
        batch_size_per_device=batch_size_per_device,
        context_length=context_length,
    )
    global_batch_size = batch_size_per_device * mesh.shape[parallel.data_axis_name]

    # Load a few batches from the iterators.
    loaded_batches = {
        "train": _load_n_batches(train_iterator, max_batches=20),
        "validation": _load_n_batches(eval_iterator, max_batches=20),
    }
    padding_token_id = 0  # hard-coded to the default value 0 for now

    # 1) Make sure that the batch size is correct.
    train_batch = loaded_batches["train"][0]
    assert train_batch.inputs.shape == (
        global_batch_size,
        context_length,
    ), f"Incorrect batch shape: {train_batch.inputs.shape}. Expected {(global_batch_size, context_length)}."
    assert train_batch.inputs.sharding.spec == P(
        mesh.axis_names
    ), f"Incorrect sharding spec: {train_batch.inputs.sharding.spec}. Expected {P(mesh.axis_names)}."

    # 2) Check that not too many padded values and EOD tokens are in the loaded dataset.
    # Compute the fraction of padding and EOD tokens in the dataset.
    stats = {"train": {}, "validation": {}}
    for split, batches in loaded_batches.items():
        n_padded = sum([(batch.inputs == padding_token_id).sum() for batch in batches])
        n_eod = sum([(batch.inputs == tokenizer.eos_token_id).sum() for batch in batches])
        n_total = sum([batch.inputs.size for batch in batches])
        stats[split]["frac_padded"] = n_padded / n_total
        stats[split]["frac_eod"] = n_eod / n_total

    # Check that padding fraction is not too large.
    for split in ["train", "validation"]:
        frac_padded = stats[split]["frac_padded"]
        assert frac_padded < 0.1, f"Got {frac_padded:.2%} padded values in {split}. Should be less than 10%."
        frac_eod = stats[split]["frac_eod"]
        assert frac_eod < 0.05, f"Got {frac_eod:.2%} EOD tokens in {split}. Should be less than 5%."

    for batch in loaded_batches["train"] + loaded_batches["validation"]:
        # 3) Check that inputs and targets are shifted by 1. Do the same for segmentations.
        assert np.all(batch.inputs[:, 1:] == batch.targets[:, :-1]), "inputs and targets are not shifted by 1."
        assert np.all(
            batch.inputs_segmentation[:, 1:] == batch.targets_segmentation[:, :-1]
        ), "Inputs_segmentation and targets_segmentation are not shifted by 1."
        # 4) Check that segmentations are correct.
        # When there is a new document, the segmentation number increases. The token at this position must be EOD token.
        is_new_sequence = (batch.targets_segmentation[:, 1:] - batch.targets_segmentation[:, :-1]) == 1
        assert np.all(
            batch.targets[:, :-1][is_new_sequence] == tokenizer.eos_token_id
        ), "Expected EOD token at positions where the segmentation increases. Found othe tokens."
        # 5) Check that targets are invalidated at positions where the targets_segmentation mask is 0.
        assert np.all(
            batch.targets[batch.targets_segmentation == 0] == padding_token_id
        ), "Targets are not invalidated at positions where the targets_segmentation mask is 0."

    # 6) Check that the packed batch can be found in the actual dataset. Test this for eval batch 0.
    # First, extract all subsequences using segmentation mask.
    batch = loaded_batches["validation"][0]
    subsequences, paddings, eod_positions = [], [], []
    for idx_batch in range(len(batch.inputs)):
        example = batch[idx_batch]  # Do not iterate over the batch via "for example in batch", this iterates forever.
        paddings.append(example.inputs[example.inputs_segmentation == 0])
        eod_positions.append(example.inputs[example.inputs == tokenizer.eos_token_id])
        for sequence_id in range(1, max(example.inputs_segmentation) + 1):
            is_subsequence = np.bitwise_and(
                example.inputs_segmentation == sequence_id, example.inputs != tokenizer.eos_token_id
            )
            subsequences.append(example.inputs[is_subsequence])

    # First sanity-check that we extracted all subsequences and paddings correctly before testing.
    n_subsequences = sum([len(ex) for ex in subsequences])
    n_paddings = sum([len(ex) for ex in paddings])
    n_eod_positions = sum([len(ex) for ex in eod_positions])
    assert n_subsequences + n_paddings + n_eod_positions == batch.targets.size, (
        f"A sanity check that extracted subsequences and paddings cover the entire batch failed: "
        f"Got {n_subsequences} subsequences, {n_subsequences} paddings, {n_eod_positions} eod tokens "
        f"for {batch.targets.size} total tokens in batch."
    )

    decoded_texts = [tokenizer.decode(subseq) for subseq in subsequences]
    original_eval_texts = [eval_ds[i]["text"] for i in range(len(eval_ds["text"]))]

    # Test that all decoded texts (from batch 0) are found in the original texts (eval).
    for i in range(len(decoded_texts)):
        assert any(
            orig_text.startswith(decoded_texts[i]) for orig_text in original_eval_texts
        ), f"The {i}-th decoded document from unpacked batch was not found in any of the documents."

    # Test that the first few (10) original texts are found in eval batch 0.
    counter_empty_doc = 0
    for i in range(10):
        orig_text = original_eval_texts[i]
        if len(orig_text) == 0:
            counter_empty_doc += 1
            continue  # Skip empty documents.
        assert any(
            orig_text.startswith(decoded_text) for decoded_text in decoded_texts
        ), f"Could not find the {i}-th original document in the decoded documents from unpacked batch."
    assert counter_empty_doc <= 5, "Failed sanity-check: more than 5 out of the first 10 documents are empty."

    # 7) Test that we can iterate 3x through the validation set without running into issues.
    # Note: atm the iterator is non-deterministic. The first batch is not the same for each epoch.
    # But we test that we have close to the same number of batches for each epoch.
    n_batches_per_epoch = [0 for _ in range(3)]
    for epoch_idx in range(3):
        for batch in eval_iterator:
            n_batches_per_epoch[epoch_idx] += 1
            assert n_batches_per_epoch[epoch_idx] < 1000, "Eval iterator loaded more than 1k batches."
    # assert that we have +-2 difference in numbers of batches
    assert (
        max(n_batches_per_epoch) - min(n_batches_per_epoch) <= 2
    ), f"Difference in number of batches is too large. Got {n_batches_per_epoch} batches per epoch."


def _load_n_batches(iterator: MultiHostDataLoadIterator, max_batches: int = 20) -> list[LLMBatch]:
    """Loads a given number of batches from an iterator.

    Args:
        iterator: MultiHostDataLoadIterator from which to load batches.
        max_batches: Maximum number of batches to load.

    Returns:
        List of loaded batches.
    """
    batches = []
    for batch in iterator:
        if len(batches) >= max_batches:
            break
        batches.append(batch)
    return batches


def _setup_data(
    tmp_path: str | Path, batch_size_per_device: int = 2, context_length: int = 128
) -> tuple[
    ParallelConfig,
    Mesh,
    Dataset,
    Dataset,
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
        tmp_path: Temporary path for pytests storing cache files.
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
        fsdp_modules=("Embed", "LMHead", "mLSTMBlock"),
        fsdp_min_weight_size=2**8,
        fsdp_axis_size=1,
        model_axis_size=1,
        data_axis_size=-1,
    )

    # Initialize mesh.
    mesh = initialize_mesh(init_distributed_on_slurm=False, parallel_config=parallel)

    global_batch_size = batch_size_per_device * mesh.shape[parallel.data_axis_name]
    train_config, eval_config = HFHubDataConfig.create_train_eval_configs(
        global_batch_size=global_batch_size,
        max_target_length=context_length,
        hf_path="Salesforce/wikitext",
        hf_data_dir="wikitext-2-v1",
        hf_cache_dir=tmp_path / "hf_cache",
        data_column="text",
        tokenize_data=True,
        tokenizer_path="gpt2",
        data_shuffle_seed=42,
        add_bos=False,
        add_eos=False,
        add_eod=True,
        grain_packing=True,
        hf_num_data_processes=2,
    )

    # Load the train and validation datasets.
    train_ds = datasets.load_dataset(
        train_config.hf_path,
        data_dir=train_config.hf_data_dir,
        data_files=train_config.hf_data_files,
        cache_dir=train_config.hf_cache_dir,
        split=train_config.split,
        streaming=False,
        token=train_config.hf_access_token,
        num_proc=train_config.hf_num_data_processes,
    )
    eval_ds = datasets.load_dataset(
        eval_config.hf_path,
        data_dir=eval_config.hf_data_dir,
        data_files=eval_config.hf_data_files,
        cache_dir=eval_config.hf_cache_dir,
        split="validation",
        streaming=False,
        token=eval_config.hf_access_token,
        num_proc=eval_config.hf_num_data_processes,
    )

    # Define iterators
    train_iterator = create_data_iterator(config=train_config, mesh=mesh)
    eval_iterator = create_data_iterator(config=eval_config, mesh=mesh)

    # Get the tokenizer that is used for the train_iterator and eval_iterator. We need it for testing.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        train_config.tokenizer_path,
        clean_up_tokenization_spaces=False,  # See https://github.com/huggingface/transformers/issues/31884
        legacy=False,
        token=train_config.hf_access_token,
        use_fast=True,
        add_bos=train_config.add_bos,
        add_eos=train_config.add_eos,
        cache_dir=train_config.hf_cache_dir,
    )
    return parallel, mesh, train_ds, eval_ds, train_iterator, eval_iterator, tokenizer
