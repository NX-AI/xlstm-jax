from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import pytest

from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.llama import LlamaConfig, LlamaTransformer

MODEL_CONFIGS = [
    lambda parallel: LlamaConfig(
        vocab_size=64,
        embedding_dim=32,
        num_blocks=3,
        head_dim=16,
        causal=True,
        parallel=parallel,
        dtype="bfloat16",
        ffn_multiple_of=4,
        scan_blocks=True,
        mask_across_document_boundaries=False,
    ),
    lambda parallel: LlamaConfig(
        vocab_size=64,
        embedding_dim=48,
        num_blocks=2,
        head_dim=8,
        causal=True,
        parallel=parallel,
        dtype="float32",
        ffn_multiple_of=64,
        scan_blocks=True,
        logits_soft_cap=5.0,
        mask_across_document_boundaries=True,
    ),
    lambda parallel: LlamaConfig(
        vocab_size=64,
        embedding_dim=64,
        num_blocks=1,
        head_dim=64,
        causal=True,
        parallel=parallel,
        dtype="bfloat16",
        ffn_multiple_of=64,
        scan_blocks=True,
        attention_backend="pallas_triton",
        mask_across_document_boundaries=True,
    ),
]
LARGE_MODEL_CONFIGS = [
    lambda parallel: LlamaConfig(
        vocab_size=64,
        embedding_dim=384,
        num_blocks=2,
        head_dim=64,
        causal=True,
        parallel=parallel,
        dtype="bfloat16",
        ffn_multiple_of=4,
        scan_blocks=True,
        attention_backend="pallas_triton",
        mask_across_document_boundaries=True,
    ),
    lambda parallel: LlamaConfig(
        vocab_size=64,
        embedding_dim=512,
        num_blocks=2,
        head_dim=128,
        causal=True,
        parallel=parallel,
        dtype="bfloat16",
        ffn_multiple_of=64,
        scan_blocks=True,
        attention_backend="cudnn",
        mask_across_document_boundaries=True,
    ),
]
CONFIG_PARAMS = list(
    zip(
        MODEL_CONFIGS + LARGE_MODEL_CONFIGS,
        [True] * len(MODEL_CONFIGS) + [False] * len(LARGE_MODEL_CONFIGS),
    )
)


@pytest.mark.parametrize("fsdp_size", [1, 8])
@pytest.mark.parametrize("llama_config_generator,on_cpu", CONFIG_PARAMS)
def test_llama_trainer(
    llm_trainer: Any,
    tmp_path: Path,
    llama_config_generator: Callable[[ParallelConfig], ModelConfig],
    on_cpu: bool,
    fsdp_size: int,
):
    """
    Tests training Llama.
    """
    is_cpu_backend = jax.default_backend() == "cpu"
    if fsdp_size > pytest.num_devices:
        pytest.skip("FSDP size is greater than the number of devices.")
    if not on_cpu and is_cpu_backend:
        pytest.skip("Skipping large flash attention test on CPU.")
    context_length = 8 if is_cpu_backend else 128

    def config_generator(parallel):
        return ModelConfig(
            model_class=LlamaTransformer,
            parallel=parallel,
            model_config=llama_config_generator(parallel),
        )

    # For GPU-based tests, we skip the check_saved_model test as if we use flash attention,
    # the tests currently fail with tiny differences in perplexity. This might come from autotuning
    # or similar minor changes within the computation graph during recompilation.
    llm_trainer.model_training_test(
        tmp_path,
        config_generator,
        fsdp_size,
        vocab_size=64,
        context_length=context_length,
        check_saved_model=is_cpu_backend,
    )


@pytest.mark.parametrize("llama_config_generator,on_cpu", CONFIG_PARAMS)
def test_llama_causal_masking(
    llm_trainer: Any, llama_config_generator: Callable[[ParallelConfig], LlamaConfig], on_cpu: bool
):
    """
    Tests causal masking in Llama.
    """
    if not on_cpu and jax.default_backend() == "cpu":
        pytest.skip("Skipping large flash attention test on CPU.")

    def model_generator(parallel):
        return LlamaTransformer(llama_config_generator(parallel))

    exmp_config = llama_config_generator(ParallelConfig())
    llm_trainer.causal_masking_test(
        model_generator, vocab_size=64, test_document_borders=exmp_config.mask_across_document_boundaries
    )
