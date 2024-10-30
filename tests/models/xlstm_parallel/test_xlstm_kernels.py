from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.cell import mLSTMBackendNameAndKwargs, mLSTMCellConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_jax.models.xlstm_parallel.components.feedforward import FeedForwardConfig
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig

# Define configuration
MODEL_CONFIGS = [
    # mlstm kernel max_triton
    lambda parallel: xLSTMLMModelConfig(
        vocab_size=100,
        embedding_dim=64,
        logits_soft_cap=None,
        num_blocks=1,
        context_length=64,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        parallel=parallel,
        scan_blocks=True,
        dtype="float32",
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                layer_type="mlstm_v1",
                num_heads=2,
                embedding_dim=64,
                context_length=64,
                mlstm_cell=mLSTMCellConfig(
                    igate_bias_init_range=-3.0,
                    backend=mLSTMBackendNameAndKwargs(name="triton_kernels", kwargs={"backend_name": "max_triton"}),
                ),
                qk_dim_factor=0.5,
                v_dim_factor=2.0,
            ),
            feedforward=FeedForwardConfig(
                proj_factor=8.0 / 3.0,
                act_fn="swish",
                embedding_dim=64,
                dropout=0.0,
                bias=False,
                ff_type="ffn_gated",
                dtype="float32",
            ),
        ),
    ),
    # mlstm kernel triton_stablef
    lambda parallel: xLSTMLMModelConfig(
        vocab_size=100,
        embedding_dim=64,
        logits_soft_cap=None,
        num_blocks=1,
        context_length=64,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        parallel=parallel,
        scan_blocks=True,
        dtype="float32",
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                layer_type="mlstm_v1",
                num_heads=2,
                embedding_dim=64,
                context_length=64,
                mlstm_cell=mLSTMCellConfig(
                    igate_bias_init_range=-3.0,
                    backend=mLSTMBackendNameAndKwargs(name="triton_kernels", kwargs={"backend_name": "triton_stablef"}),
                ),
                qk_dim_factor=0.5,
                v_dim_factor=2.0,
            ),
            feedforward=FeedForwardConfig(
                proj_factor=8.0 / 3.0,
                act_fn="swish",
                embedding_dim=64,
                dropout=0.0,
                bias=False,
                ff_type="ffn_gated",
                dtype="float32",
            ),
        ),
    ),
]


@pytest.mark.skipif(not pytest.triton_available, reason="Triton is not available.")
@pytest.mark.parametrize("fsdp_size", [1])
@pytest.mark.parametrize("xlstm_config_generator", MODEL_CONFIGS)
def test_xlstm_trainer_kernels(
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

    llm_trainer.model_training_test(tmp_path, config_generator, fsdp_size, vocab_size=100, context_length=256)
