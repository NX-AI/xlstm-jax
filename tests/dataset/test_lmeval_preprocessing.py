import numpy as np
from lm_eval.api.instance import Instance
from transformers import AutoTokenizer

from xlstm_jax.dataset.lmeval_dataset import HFTokenizeLogLikelihoodRolling


# TODO add parametric tests for tokenization and preprocessing for lm_eval
def test_lmeval_preprocessing():
    max_length = 8
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, legacy=False, clean_up_tokenization_spaces=False)

    llr = HFTokenizeLogLikelihoodRolling(tokenizer_path="gpt2", max_length=max_length)

    inst1 = Instance(request_type="loglikelihood_rolling", doc={}, idx=0, arguments=("This is some exemplary text."))
    inst2 = Instance(
        request_type="loglikelihood_rolling",
        doc={},
        idx=1,
        arguments=("This is actually a sequence beyond the maximum sequence length."),
    )
    inst3 = Instance(request_type="loglikelihood_rolling", doc={}, idx=2, arguments=("This is some exemplary text."))
    inst4 = Instance(
        request_type="loglikelihood_rolling",
        doc={},
        idx=1,
        arguments=("This is actually a sequence beyond the maximum sequence length at the right."),
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
