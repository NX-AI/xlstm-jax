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
    # v2 model
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
        dtype=jnp.bfloat16,
        lm_head_dtype="float32",
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
    # v1 model
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
        dtype=jnp.bfloat16,
        lm_head_dtype="bfloat16",
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
                dtype=jnp.bfloat16,
            ),
        ),
    ),
    # v2 model with different qk_dim_factor and v_dim_factor
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
                qk_dim_factor=0.5,
                v_dim_factor=2.0,
                mlstm_cell=mLSTMCellConfig(
                    gate_linear_headwise=True,
                    gate_soft_cap=30.0,
                ),
            )
        ),
    ),
    # v1 model with different qk_dim_factor and v_dim_factor
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
                qk_dim_factor=0.5,
                v_dim_factor=2.0,
            ),
            feedforward=FeedForwardConfig(
                proj_factor=8.0 / 3.0,
                act_fn="swish",
                embedding_dim=16,
                dropout=0.0,
                bias=False,
                ff_type="ffn_gated",
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
    if fsdp_size > pytest.num_devices:
        pytest.skip("FSDP size is greater than the number of devices.")

    def config_generator(parallel):
        return ModelConfig(
            model_class=xLSTMLMModel,
            parallel=parallel,
            model_config=xlstm_config_generator(parallel),
        )

    llm_trainer.model_training_test(tmp_path, config_generator, fsdp_size, vocab_size=100, context_length=16)


@pytest.mark.parametrize("xlstm_config_generator", MODEL_CONFIGS)
def test_xlstm_causal_masking_document_borders(
    llm_trainer: Any, xlstm_config_generator: Callable[[ParallelConfig], xLSTMLMModelConfig]
):
    """
    Tests causal masking in xLSTM.
    """

    def model_generator(parallel):
        config = xlstm_config_generator(parallel)
        config.mlstm_block.mlstm.mlstm_cell.reset_at_document_boundaries = True
        # In the v2 cell, the convolution may still look across document borders. We set it to 1 to avoid this.
        config.mlstm_block.mlstm.conv1d_kernel_size = 1
        return xLSTMLMModel(config)

    llm_trainer.causal_masking_test(model_generator, vocab_size=100, context_length=16, test_document_borders=True)
