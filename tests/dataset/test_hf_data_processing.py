#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import itertools
from collections.abc import Sequence
from pathlib import Path

import datasets
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
def test_hf_dataset_with_packing(tmp_path: Path):
    """Test data loading with huggingface datasets, using grain
    PackAndBatchOperation."""

    batch_size_per_device = 8
    context_length = 128
    eod_token_id = 50256  # for "gpt2" tokenizer

    # Prepare the dataset and dataloaders for the test.
    parallel, mesh, _train_ds, eval_ds, train_iterator, eval_iterator, tokenizer = _setup_data(
        tmp_path=tmp_path,
        batch_size_per_device=batch_size_per_device,
        context_length=context_length,
    )
    global_batch_size = batch_size_per_device * mesh.shape[parallel.data_axis_name]

    # Load a few batches from the iterators.
    loaded_batches = {
        "train": list(itertools.islice(train_iterator, 20)),
        "validation": list(itertools.islice(eval_iterator, 20)),
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
        n_padded = sum((batch.inputs == padding_token_id).sum() for batch in batches)
        n_eod = sum((batch.inputs == tokenizer.eos_token_id).sum() for batch in batches)
        n_total = sum(batch.inputs.size for batch in batches)
        stats[split]["frac_padded"] = n_padded / n_total
        stats[split]["frac_eod"] = n_eod / n_total

    # Check that padding fraction is not too large.
    for split in ["train", "validation"]:
        frac_padded = stats[split]["frac_padded"]
        assert frac_padded < 0.1, f"Got {frac_padded:.2%} padded values in {split}. Should be less than 10%."
        frac_eod = stats[split]["frac_eod"]
        assert frac_eod < 0.05, f"Got {frac_eod:.2%} EOD tokens in {split}. Should be less than 5%."

    for batch in loaded_batches["train"] + loaded_batches["validation"]:
        # 3) Check that inputs and targets are shifted by 1.
        assert np.all(batch.inputs[:, 1:] == batch.targets[:, :-1]), "inputs and targets are not shifted by 1."
        # 4) Check that segmentations are correct.
        # When there is a new document, the segmentation number increases. The token at this position must be EOD token.
        is_new_sequence = (batch.targets_segmentation[:, 1:] - batch.targets_segmentation[:, :-1]) == 1
        assert np.all(
            batch.targets[:, :-1][is_new_sequence] == tokenizer.eos_token_id
        ), "Expected EOD token at positions where the segmentation increases. Found other tokens."
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
        # TODO: it was nicer to take example.inputs_segmentation == 0, but we no longer shift it (different assumption)
        paddings.append(example.inputs[example.inputs == 0])
        eod_positions.append(example.inputs[example.inputs == tokenizer.eos_token_id])
        for sequence_id in range(1, max(example.inputs_segmentation) + 1):
            is_subsequence = np.bitwise_and(
                example.inputs_segmentation == sequence_id, example.inputs != tokenizer.eos_token_id
            )
            subsequences.append(example.inputs[is_subsequence])

    # First sanity-check that we extracted all subsequences and paddings correctly before testing.
    n_subsequences = sum(len(ex) for ex in subsequences)
    n_paddings = sum(len(ex) for ex in paddings)
    n_eod_positions = sum(len(ex) for ex in eod_positions)
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
        for _batch in eval_iterator:
            n_batches_per_epoch[epoch_idx] += 1
            assert n_batches_per_epoch[epoch_idx] < 1000, "Eval iterator loaded more than 1k batches."
    # assert that we have +-2 difference in numbers of batches
    assert (
        max(n_batches_per_epoch) - min(n_batches_per_epoch) <= 2
    ), f"Difference in number of batches is too large. Got {n_batches_per_epoch} batches per epoch."

    # Test if document borders are correctly computed.
    eval_batches = list(eval_iterator)
    _check_document_borders(batches=eval_batches + [train_batch], eod_token_id=eod_token_id)


def _check_document_borders(batches: Sequence[LLMBatch], eod_token_id: int):
    """Check that the document borders computed in LLMBatch (using segmentation) are at indices,
    where inputs are EOD tokens, since an EOD token marks the beginning of a new sequence."""
    assert len(batches) > 0, "No batches to check."
    for idx, batch in enumerate(batches):
        manual_document_borders = batch.inputs == eod_token_id
        document_borders = batch.get_document_borders()
        assert document_borders[:, 0].all(), "The first token should always be a document border."
        assert (manual_document_borders[:, 1:] == document_borders[:, 1:]).all(), (
            f"Manual document borders should be the same as from LLMBatch.get_document_borders, "
            f"except for the first token which is always a document border. {idx}"
        )
        assert (batch.inputs[manual_document_borders] == eod_token_id).all(), "Borders should be EOD tokens in inputs"
        assert (batch.inputs[~manual_document_borders] != eod_token_id).all(), "Non-borders should not be EOD tokens"


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
        tmp_path: Temporary path for pytest storing cache files.
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
        tokenizer_path="gpt2",
        data_shuffle_seed=42,
        add_bos=False,
        add_eos=False,
        add_eod=True,
        grain_packing=True,
        hf_num_data_processes=2,
        worker_count=0,
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
