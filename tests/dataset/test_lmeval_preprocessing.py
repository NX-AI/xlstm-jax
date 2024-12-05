import grain.python as grain
import jax
import numpy as np
import pytest
from lm_eval.api.instance import Instance
from transformers import AutoTokenizer

from xlstm_jax.dataset.grain_transforms import HFPrefixTokenize
from xlstm_jax.dataset.lmeval_dataset import HFTokenizeLogLikelihoodRolling
from xlstm_jax.dataset.lmeval_pipeline import lmeval_preprocessing_pipeline
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models.configs import ParallelConfig


# TODO add parametric tests for tokenization and preprocessing for lm_eval
@pytest.mark.skipif(not pytest.grain_available, reason="Grain is not available.")
def test_lmeval_preprocessing():
    max_length = 8
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, legacy=False, clean_up_tokenization_spaces=False)

    llr = HFTokenizeLogLikelihoodRolling(tokenizer_path="gpt2", max_length=max_length)

    inst1 = Instance(request_type="loglikelihood_rolling", doc={}, idx=0, arguments=("This is some exemplary text.",))
    inst2 = Instance(
        request_type="loglikelihood_rolling",
        doc={},
        idx=1,
        arguments=("This is actually a sequence beyond the maximum sequence length.",),
    )
    inst3 = Instance(request_type="loglikelihood_rolling", doc={}, idx=2, arguments=("This is some exemplary text.",))
    inst4 = Instance(
        request_type="loglikelihood_rolling",
        doc={},
        idx=1,
        arguments=("This is actually a sequence beyond the maximum sequence length at the right.",),
    )

    tok_seq = llr.map([inst1, inst2, inst3, inst4])

    assert tok_seq is not None
    assert len(tok_seq) == 6
    assert np.allclose(tok_seq[1]["inputs_segmentation"], np.ones(max_length, dtype=np.int32))

    for toks in tok_seq:
        assert np.all(
            toks["inputs"][1:] == toks["targets"][:-1]
        ), f"Inputs are not shifted targets, {toks['inputs'][1:]} == {toks['targets'][:-1]}"

    all_tokens = tokenizer(inst2.args[0], return_attention_mask=False)["input_ids"]
    assert np.sum(tok_seq[1]["targets_segmentation"]) + np.sum(tok_seq[2]["targets_segmentation"]) == len(
        all_tokens
    ), "Targets do not match all tokens for longer sequence."

    all_tokens = tokenizer(inst1.args[0], return_attention_mask=False)["input_ids"]
    assert np.sum(tok_seq[0]["targets_segmentation"]) == len(all_tokens), "Targets do not match all tokens."

    all_tokens = tokenizer(inst4.args[0], return_attention_mask=False)["input_ids"]
    assert np.sum(tok_seq[4]["targets_segmentation"]) + np.sum(tok_seq[5]["targets_segmentation"]) == len(
        all_tokens
    ), "Targets do not match all tokens for longer sequence at the right limit."


@pytest.mark.skipif(not pytest.grain_available, reason="Grain is not available.")
def test_hfprefix_tokenize():
    """Tests the HuggingFace prefix tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, legacy=False, clean_up_tokenization_spaces=False)

    # test with prefix
    transform = HFPrefixTokenize(tokenizer=tokenizer, prefix_tokenizer=tokenizer)
    ds = grain.MapDataset.source([{"prefix": "Prefix:", "text": "This text uses a prefix."}])
    output = transform.map(ds[0])
    assert np.all(
        np.sum(output["targets_segmentation"]) < output["targets_segmentation"].shape[1]
    ), "Not all tokens should be targets here!"
    assert all(key in output for key in ["inputs", "targets", "inputs_segmentation", "targets_segmentation"])

    # test no prefix
    transform = HFPrefixTokenize(tokenizer=tokenizer, prefix_tokenizer=tokenizer)
    ds = grain.MapDataset.source([{"prefix": "", "text": "This text uses NO prefix."}])
    output = transform.map(ds[0])
    assert all(key in output for key in ["inputs", "targets", "inputs_segmentation", "targets_segmentation"])
    first_real_token = output["inputs"][0][1]
    assert np.all(output["targets_segmentation"] == 1)

    # test no bos_token
    transform = HFPrefixTokenize(tokenizer=tokenizer, prefix_tokenizer=tokenizer, add_bos_token=False)
    ds = grain.MapDataset.source([{"prefix": "", "text": "This text uses NO prefix."}])
    output = transform.map(ds[0])
    assert output["inputs"][0][0] == first_real_token
    assert all(key in output for key in ["inputs", "targets", "inputs_segmentation", "targets_segmentation"])
    assert np.all(output["targets_segmentation"] == 1)

    # test eos_token
    transform = HFPrefixTokenize(tokenizer=tokenizer, prefix_tokenizer=tokenizer, add_eos_token=True)
    ds = grain.MapDataset.source([{"prefix": "", "text": "This text uses NO prefix."}])
    output = transform.map(ds[0])
    assert all(key in output for key in ["inputs", "targets", "inputs_segmentation", "targets_segmentation"])
    assert output["targets"][0][-1] == tokenizer.eos_token_id
    assert np.all(output["targets_segmentation"] == 1)


# Tests with dataloading_host_count > 1 fail now, as there is synchronization between workers needed to properly
# pad batches
# This is more suitable for JIT dataset processing instead of global preprocessing all data in every worker.
@pytest.mark.skipif(not pytest.grain_available, reason="Grain is not available.")
@pytest.mark.parametrize("dataloading_host_count", [1])
def test_lmeval_iterator(dataloading_host_count: int):
    """Tests the LMEval dataset iterator"""
    # Initialize mesh.
    parallel = ParallelConfig(
        fsdp_axis_size=1,
        model_axis_size=1,
        data_axis_size=1,
    )
    # Use only one device for this test, as in practice, each data loader process only sees a single device.
    mesh = initialize_mesh(
        init_distributed_on_slurm=False, parallel_config=parallel, device_array=np.array(jax.devices())[0:1]
    )
    # this dataset should have sequences of very different size to test proper sequence padding across
    # multiple dataloader hosts
    # adding text with % 5 and len=12 are chosen to fulfill this purpose, but there is room for improvement to
    # specifically test all possibilities
    dataset = [
        Instance(request_type="loglikelihood_rolling", doc={}, idx=0, arguments=("This is some exemplary text.",)),
    ] + [
        Instance(
            request_type="loglikelihood_rolling",
            doc={},
            idx=0,
            arguments=(
                "This is some exemplary text. This is longer to test padding."
                + "Really a lot more text repeated" * (sample_len % 5),
            ),
        )
        for sample_len in range(12)
    ]

    if dataloading_host_count == 1:
        pipeline = lmeval_preprocessing_pipeline(
            dataloading_host_index=0,
            dataloading_host_count=1,
            global_mesh=mesh,
            dataset=dataset,
            global_batch_size=2 * dataloading_host_count,
            tokenizer_path="gpt2",
            padding_multiple=128,
        )

        total_docs = 0
        for batch in pipeline:
            total_docs += (batch.document_idx != 0).sum()
            assert batch.inputs.shape == (2, 128)
            assert batch.targets.shape == (2, 128)
            assert batch.inputs_position.shape == (2, 128)
            assert batch.targets_position.shape == (2, 128)
            assert batch.inputs_segmentation.shape == (2, 128)
            assert batch.targets_segmentation.shape == (2, 128)
            assert batch.document_idx.shape == (2,)
            assert batch.sequence_idx.shape == (2,)
    else:
        all_batches = {}
        for host_index in range(dataloading_host_count):
            pipeline = lmeval_preprocessing_pipeline(
                dataloading_host_index=host_index,
                dataloading_host_count=dataloading_host_count,
                global_mesh=mesh,
                dataset=dataset,
                global_batch_size=2 * dataloading_host_count,
                tokenizer_path="gpt2",
                padding_multiple=4,
            )
            all_batches[host_index] = list(pipeline)

        all_batches = jax.device_get(all_batches)
        all_document_idx = np.concatenate(
            [
                np.concatenate([all_batches[host_idx][idx].document_idx for host_idx in all_batches], axis=0)
                for idx in range(len(all_batches[0]))
            ],
            axis=0,
        )
        total_docs = (all_document_idx != 0).sum()

    assert total_docs == len(dataset), f"Documents in batches {total_docs} != {len(dataset)} inserted docs."
