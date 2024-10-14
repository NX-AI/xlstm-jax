from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest

from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.cell import mLSTMCellConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_jax.models.xlstm_parallel.components.feedforward import FeedForwardConfig
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig

# Define configuration
MODEL_CONFIGS = [
    lambda parallel: xLSTMLMModelConfig(
        vocab_size=100,
        embedding_dim=16,
        logits_soft_cap=30.0,
        num_blocks=1,
        context_length=16,
        tie_weights=False,
        norm_type="layernorm",
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        parallel=parallel,
        scan_blocks=False,
        dtype=jnp.float32,
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                proj_factor=2.0,
                conv1d_kernel_size=4,
                num_heads=2,
                dropout=0.2,
                embedding_dim=16,
                context_length=32,
                gate_input="qkv",
                mlstm_cell=mLSTMCellConfig(
                    gate_linear_headwise=True,
                    gate_soft_cap=30.0,
                ),
            )
        ),
    ),
    lambda parallel: xLSTMLMModelConfig(
        vocab_size=100,
        embedding_dim=16,
        logits_soft_cap=None,
        num_blocks=1,
        context_length=16,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        parallel=parallel,
        scan_blocks=True,
        dtype=jnp.float32,
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                layer_type="mlstm_v1",
                num_heads=2,
                embedding_dim=16,
                context_length=32,
                mlstm_cell=mLSTMCellConfig(
                    igate_bias_init_range=-3.0,
                ),
            ),
            feedforward=FeedForwardConfig(
                proj_factor=2.0,
                act_fn="gelu",
                embedding_dim=16,
                dropout=0.0,
                bias=False,
                ff_type="ffn",
                dtype=jnp.float32,
            ),
        ),
    ),
]


@pytest.mark.parametrize("fsdp_size", [8])
@pytest.mark.parametrize("xlstm_config_generator", MODEL_CONFIGS[1:2])
def test_xlstm_trainer(
    llm_trainer: Any, tmp_path: Path, xlstm_config_generator: Callable[[ParallelConfig], ModelConfig], fsdp_size: int
):
    """
    Tests training xLSTM.
    """

    def config_generator(parallel):
        return ModelConfig(
            model_class=xLSTMLMModel,
            parallel=parallel,
            model_config=xlstm_config_generator(parallel),
        )

    llm_trainer.model_training_test(tmp_path, config_generator, fsdp_size, vocab_size=100, context_length=16)


@pytest.mark.parametrize("xlstm_config_generator", MODEL_CONFIGS)
def test_xlstm_causal_masking(llm_trainer: Any, xlstm_config_generator: Callable[[ParallelConfig], xLSTMLMModelConfig]):
    """
    Tests causal masking in xLSTM.
    """

    def model_generator(parallel):
        return xLSTMLMModel(xlstm_config_generator(parallel))

    llm_trainer.causal_masking_test(model_generator, vocab_size=100, context_length=16)