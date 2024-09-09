import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
USE_CPU = False
if USE_CPU:
    from distributed.utils import simulate_CPU_devices
    simulate_CPU_devices(8)
from model_parallel.xlstm_lm_model import xLSTMLMModelConfig
from model_parallel.blocks.mlstm.block import mLSTMBlockConfig
from model_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from model_parallel.benchmark import benchmark_model
from model_parallel.utils import ParallelConfig
import jax.numpy as jnp

MODEL_CONFIGS = {
    "debug": {
        "config": xLSTMLMModelConfig(
            vocab_size=100,
            embedding_dim=16,
            num_blocks=4,
            context_length=128,
            tie_weights=False,
            add_embedding_dropout=True,
            add_post_blocks_norm=True,
            parallel=ParallelConfig(
                data_axis_name="dp",
                model_axis_name="tp",
                pipeline_axis_name="pp",
            ),
            dtype=jnp.float32,
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    proj_factor=2.0,
                    conv1d_kernel_size=4,
                    num_heads=4,
                    dropout=0.2,
                    embedding_dim=16,
                    context_length=128,
                )
            )
        ),
    },
    "debug_tp": {
        "config": xLSTMLMModelConfig(
            vocab_size=100,
            embedding_dim=128,
            num_blocks=4,
            context_length=128,
            tie_weights=False,
            add_embedding_dropout=True,
            add_post_blocks_norm=True,
            parallel=ParallelConfig(
                data_axis_name="dp",
                model_axis_name="tp",
                pipeline_axis_name="pp",
            ),
            dtype=jnp.float32,
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    proj_factor=2.0,
                    conv1d_kernel_size=4,
                    num_heads=4,
                    dropout=0.2,
                    embedding_dim=128,
                    context_length=128,
                )
            )
        ), 
        "model_axis_size": 4
    },
    "120M": {
        "config": xLSTMLMModelConfig(
            vocab_size=50304,
            embedding_dim=768,
            num_blocks=12,
            context_length=2048,
            tie_weights=False,
            add_embedding_dropout=False,
            add_post_blocks_norm=True,
            parallel=ParallelConfig(
                data_axis_name="dp",
                model_axis_name="tp",
                pipeline_axis_name="pp",
            ),
            scan_blocks=False,
            dtype=jnp.bfloat16,
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    num_heads=4,
                )
            )
        ), 
        "batch_size": 128, 
        "gradient_accumulate_steps": 1
    },
    "120M_fsdp": {
        "config": xLSTMLMModelConfig(
            vocab_size=50304,
            embedding_dim=768,
            num_blocks=12,
            context_length=2048,
            tie_weights=False,
            add_embedding_dropout=False,
            add_post_blocks_norm=True,
            parallel=ParallelConfig(
                data_axis_name="dp",
                model_axis_name="tp",
                pipeline_axis_name="pp",
                fsdp_modules=("Embed", "LMHead", "mLSTMBlock"),
            ),
            scan_blocks=False,
            dtype=jnp.bfloat16,
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    num_heads=4,
                )
            )
        ), 
        "batch_size": 128, 
        "gradient_accumulate_steps": 1
    },
    "1.3B": {
        "config": xLSTMLMModelConfig(
            vocab_size=50304,
            embedding_dim=2048,
            num_blocks=48,
            context_length=2048,
            tie_weights=False,
            add_embedding_dropout=False,
            add_post_blocks_norm=True,
            parallel=ParallelConfig(
                data_axis_name="dp",
                model_axis_name="tp",
                pipeline_axis_name="pp",
                fsdp_modules=("Embed", "LMHead", "mLSTMBlock"),
                fsdp_min_weight_size=2 ** 18,
                remat=(),
            ),
            dtype=jnp.bfloat16,
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    num_heads=4,
                )
            )
        ), 
        "batch_size": 32, 
        "gradient_accumulate_steps": 2
    },
    "1.3B_remat": {
        "config": xLSTMLMModelConfig(
            vocab_size=50304,
            embedding_dim=2048,
            num_blocks=48,
            context_length=2048,
            tie_weights=False,
            add_embedding_dropout=False,
            add_post_blocks_norm=True,
            parallel=ParallelConfig(
                data_axis_name="dp",
                model_axis_name="tp",
                pipeline_axis_name="pp",
                fsdp_modules=("Embed", "LMHead", "mLSTMBlock"),
                fsdp_min_weight_size=2 ** 18,
                remat=("mLSTMBlock"),
            ),
            dtype=jnp.bfloat16,
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    num_heads=4,
                )
            )
        ), 
        "batch_size": 64
    },
    "1.3B_tp": {
        "config": xLSTMLMModelConfig(
            vocab_size=50304,
            embedding_dim=2048,
            num_blocks=48,
            context_length=2048,
            tie_weights=False,
            add_embedding_dropout=False,
            add_post_blocks_norm=True,
            parallel=ParallelConfig(
                data_axis_name="dp",
                model_axis_name="tp",
                pipeline_axis_name="pp",
                fsdp_modules=("Embed", "LMHead", "mLSTMBlock"),
                fsdp_min_weight_size=2 ** 18,
                remat=(),
            ),
            dtype=jnp.bfloat16,
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    num_heads=4,
                )
            )
        ), 
        "batch_size": 16, 
        "gradient_accumulate_steps": 1, 
        "model_axis_size": 4
    },
}

if __name__ == "__main__":
    benchmark_model(**MODEL_CONFIGS["120M"], num_steps=100, log_num_steps=3)