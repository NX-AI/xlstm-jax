import subprocess

from xlstm_jax.distributed.xla_utils import set_XLA_flags, simulate_CPU_devices

try:
    subprocess.check_output("nvidia-smi")
    print("GPU found, setting XLA flags.")
    set_XLA_flags()
    USE_CPU = False
except Exception:
    print("No GPU found, using CPU instead.")
    simulate_CPU_devices(8)
    USE_CPU = True

from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.xlstm_parallel.benchmark import benchmark_model
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModelConfig

import jax.numpy as jnp
import optax

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
                fsdp_axis_name="fsdp",
                model_axis_name="tp",
                pipeline_axis_name="pp",
                fsdp_modules=("Embed", "LMHead", "mLSTMBlock"),
                fsdp_min_weight_size=2**8,
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
            ),
        ),
        "batch_size_per_device": 2,
        "fsdp_axis_size": 2,
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
                fsdp_axis_name="fsdp",
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
            ),
        ),
        "batch_size_per_device": 2,
        "model_axis_size": 4,
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
            ),
        ),
        "batch_size_per_device": 16,
        "gradient_accumulate_steps": 1,
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
                fsdp_axis_name="fsdp",
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
            ),
        ),
        "batch_size_per_device": 16,
        "gradient_accumulate_steps": 1,
        "data_axis_size": 1,
        "fsdp_axis_size": -1,
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
                fsdp_axis_name="fsdp",
                model_axis_name="tp",
                pipeline_axis_name="pp",
                fsdp_modules=("Embed", "LMHead", "mLSTMBlock"),
                fsdp_min_weight_size=2**18,
                remat=(),
            ),
            dtype=jnp.bfloat16,
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    num_heads=4,
                )
            ),
        ),
        "batch_size_per_device": 4,
        "gradient_accumulate_steps": 1,
        "data_axis_size": 1,
        "fsdp_axis_size": -1,
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
            scan_blocks=True,
            parallel=ParallelConfig(
                data_axis_name="dp",
                fsdp_axis_name="fsdp",
                model_axis_name="tp",
                pipeline_axis_name="pp",
                fsdp_modules=("Embed", "LMHead", "mLSTMBlock"),
                fsdp_min_weight_size=2**18,
                remat=("mLSTMBlock"),
            ),
            dtype=jnp.bfloat16,
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    num_heads=4,
                )
            ),
        ),
        "batch_size_per_device": 32,
        "data_axis_size": 1,
        "fsdp_axis_size": -1,
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
                fsdp_axis_name="fsdp",
                model_axis_name="tp",
                pipeline_axis_name="pp",
                fsdp_modules=("Embed", "LMHead", "mLSTMBlock"),
                fsdp_min_weight_size=2**18,
                remat=(),
            ),
            dtype=jnp.bfloat16,
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    num_heads=4,
                )
            ),
        ),
        "batch_size_per_device": 4,
        "gradient_accumulate_steps": 1,
        "model_axis_size": 4,
        "data_axis_size": 1,
        "fsdp_axis_size": -1,
    },
    "7B_shallow": {
        "config": xLSTMLMModelConfig(
            vocab_size=50304,
            embedding_dim=4096,
            num_blocks=12,
            context_length=2048,
            tie_weights=False,
            add_embedding_dropout=False,
            add_post_blocks_norm=True,
            parallel=ParallelConfig(
                data_axis_name="dp",
                fsdp_axis_name="fsdp",
                model_axis_name="tp",
                pipeline_axis_name="pp",
                fsdp_modules=("Embed", "LMHead", "mLSTMBlock"),
                fsdp_min_weight_size=2**18,
                remat=(),
            ),
            dtype=jnp.bfloat16,
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    num_heads=4,
                )
            ),
        ),
        "batch_size_per_device": 4,
        "gradient_accumulate_steps": 1,
        "model_axis_size": 1,
        "data_axis_size": 1,
        "fsdp_axis_size": -1,
    },
    "7B": {
        "config": xLSTMLMModelConfig(
            vocab_size=50304,
            embedding_dim=4096,
            num_blocks=64,
            context_length=2048,
            tie_weights=False,
            add_embedding_dropout=False,
            add_post_blocks_norm=True,
            scan_blocks=True,
            parallel=ParallelConfig(
                data_axis_name="dp",
                fsdp_axis_name="fsdp",
                model_axis_name="tp",
                pipeline_axis_name="pp",
                fsdp_modules=("Embed", "LMHead", "mLSTMBlock"),
                fsdp_min_weight_size=2**18,
                remat=("mLSTMBlock"),
                tp_async_dense=False,
            ),
            dtype=jnp.bfloat16,
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    num_heads=4,
                    vmap_qk=False,
                )
            ),
        ),
        "batch_size_per_device": 4,
        "gradient_accumulate_steps": 1,
        "model_axis_size": 4,
        "fsdp_axis_size": 1,
        "optimizer": optax.adamw(
            learning_rate=optax.schedules.warmup_exponential_decay_schedule(
                init_value=0.0, peak_value=5e-4, warmup_steps=100, decay_rate=0.99, transition_steps=1000
            ),
            b1=0.9,
            b2=0.98,
            eps=1e-9,
        ),
    },
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=MODEL_CONFIGS.keys(), default="7B" if not USE_CPU else "debug")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--log_num_steps", type=int, default=3)
    args = parser.parse_args()
    benchmark_model(**MODEL_CONFIGS[args.model], num_steps=args.num_steps, log_num_steps=args.log_num_steps)
