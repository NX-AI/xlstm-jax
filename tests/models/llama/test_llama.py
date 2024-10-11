from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest

from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.llama import LlamaConfig, LlamaTransformer

MODEL_CONFIGS = [
    lambda parallel: LlamaConfig(
        vocab_size=64,
        embed_dim=32,
        num_layers=3,
        head_dim=16,
        parallel=parallel,
        dtype=jnp.bfloat16,
        ffn_multiple_of=4,
        scan_blocks=True,
    ),
    lambda parallel: LlamaConfig(
        vocab_size=64,
        embed_dim=48,
        num_layers=2,
        head_dim=8,
        parallel=parallel,
        dtype=jnp.float32,
        ffn_multiple_of=64,
        scan_blocks=True,
    ),
]


@pytest.mark.parametrize("fsdp_size", [1, 8])
@pytest.mark.parametrize("llama_config_generator", MODEL_CONFIGS)
def test_llama_trainer(
    llm_trainer: Any, tmp_path: Path, llama_config_generator: Callable[[ParallelConfig], ModelConfig], fsdp_size: int
):
    """
    Tests training Llama.
    """

    def config_generator(parallel):
        return ModelConfig(
            model_class=LlamaTransformer,
            parallel=parallel,
            model_config=llama_config_generator(parallel),
        )

    llm_trainer.model_training_test(tmp_path, config_generator, fsdp_size, vocab_size=64)


@pytest.mark.parametrize("llama_config_generator", MODEL_CONFIGS)
def test_llama_causal_masking(llm_trainer: Any, llama_config_generator: Callable[[ParallelConfig], LlamaConfig]):
    """
    Tests causal masking in Llama.
    """

    def model_generator(parallel):
        return LlamaTransformer(llama_config_generator(parallel))

    llm_trainer.causal_masking_test(model_generator, vocab_size=64)
