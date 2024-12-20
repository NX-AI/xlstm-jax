#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import subprocess

import optax

from xlstm_jax.distributed import set_XLA_flags, simulate_CPU_devices
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.xlstm_parallel.benchmark import benchmark_model
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend import mLSTMBackendNameAndKwargs
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMCellConfig, mLSTMLayerConfig
from xlstm_jax.models.xlstm_parallel.components.feedforward import FeedForwardConfig
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModelConfig

try:
    subprocess.check_output("nvidia-smi")
    print("GPU found, setting XLA flags.")
    set_XLA_flags()
    USE_CPU = False
except Exception:
    print("No GPU found, using CPU instead.")
    simulate_CPU_devices(8)
    USE_CPU = True


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
                fsdp_modules=["Embed", "LMHead", "mLSTMBlock"],
                fsdp_min_weight_size=2**8,
            ),
            dtype="float32",
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
            dtype="float32",
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
            dtype="bfloat16",
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
                fsdp_modules=["Embed", "LMHead", "mLSTMBlock"],
            ),
            scan_blocks=False,
            dtype="bfloat16",
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
                fsdp_modules=["Embed", "LMHead", "mLSTMBlock"],
                fsdp_min_weight_size=2**18,
                remat=[],
            ),
            dtype="bfloat16",
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
                fsdp_modules=["Embed", "LMHead", "mLSTMBlock"],
                fsdp_min_weight_size=2**18,
                remat=["mLSTMBlock"],
            ),
            dtype="bfloat16",
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
                fsdp_modules=["Embed", "LMHead", "mLSTMBlock"],
                fsdp_min_weight_size=2**18,
                remat=[],
            ),
            dtype="bfloat16",
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
    "1.3B_v1": {
        "config": xLSTMLMModelConfig(
            vocab_size=50304,
            embedding_dim=2048,
            num_blocks=24,
            context_length=2048,
            tie_weights=False,
            add_embedding_dropout=False,
            add_post_blocks_norm=True,
            parallel=ParallelConfig(
                data_axis_name="dp",
                fsdp_axis_name="fsdp",
                model_axis_name="tp",
                pipeline_axis_name="pp",
                fsdp_modules=[],
                fsdp_min_weight_size=2**18,
                remat=["xLSTMResBlock", "FFNResBlock"],
            ),
            scan_blocks=True,
            norm_eps=1e-6,
            norm_type="rmsnorm",
            dtype="bfloat16",
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    layer_type="mlstm_v1",
                    num_heads=4,
                    mlstm_cell=mLSTMCellConfig(
                        backend=mLSTMBackendNameAndKwargs(name="triton_kernels"),
                        # Lowering the input bias init appears to stabilize training.
                        igate_bias_init_range=-10.0,
                        add_qk_norm=False,
                        norm_type="rmsnorm",
                        norm_eps=1e-6,
                    ),
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=4.0,
                    act_fn="gelu",
                    ff_type="ffn",
                    dtype="bfloat16",
                ),
                add_post_norm=False,
            ),
        ),
        "batch_size_per_device": 16,
        "gradient_accumulate_steps": 1,
        "model_axis_size": 1,
        "data_axis_size": -1,
        "fsdp_axis_size": 1,
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
                fsdp_modules=["Embed", "LMHead", "mLSTMBlock"],
                fsdp_min_weight_size=2**18,
                remat=[],
            ),
            dtype="bfloat16",
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
                fsdp_modules=["Embed", "LMHead", "mLSTMBlock"],
                fsdp_min_weight_size=2**18,
                fsdp_gather_dtype="bfloat16",
                remat=["mLSTMBlock"],
                tp_async_dense=False,
            ),
            dtype="bfloat16",
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    num_heads=4,
                    vmap_qk=False,
                )
            ),
        ),
        "batch_size_per_device": 4,
        "gradient_accumulate_steps": 1,
        "model_axis_size": 1,
        "fsdp_axis_size": 8,
        "optimizer": optax.adamw(
            learning_rate=optax.schedules.warmup_exponential_decay_schedule(
                init_value=0.0, peak_value=5e-4, warmup_steps=100, decay_rate=0.99, transition_steps=1000
            ),
            b1=0.9,
            b2=0.98,
            eps=1e-9,
        ),
    },
    "7B_v1": {
        "config": xLSTMLMModelConfig(
            vocab_size=50304,
            embedding_dim=4096,
            num_blocks=30,
            context_length=2048,
            tie_weights=False,
            add_embedding_dropout=False,
            add_post_blocks_norm=True,
            parallel=ParallelConfig(
                data_axis_name="dp",
                fsdp_axis_name="fsdp",
                model_axis_name="tp",
                pipeline_axis_name="pp",
                fsdp_modules=["Embed", "LMHead", "mLSTMBlock"],
                fsdp_min_weight_size=2**18,
                fsdp_gather_dtype="bfloat16",
                remat=["xLSTMResBlock", "FFNResBlock"],
                tp_async_dense=False,
            ),
            scan_blocks=True,
            norm_eps=1e-6,
            norm_type="rmsnorm",
            dtype="bfloat16",
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    layer_type="mlstm_v1",
                    num_heads=8,
                    mlstm_cell=mLSTMCellConfig(
                        gate_dtype="float32",
                        backend=mLSTMBackendNameAndKwargs(name="triton_kernels"),
                        # Lowering the input bias init appears to stabilize training.
                        igate_bias_init_range=-10.0,
                        add_qk_norm=False,
                        norm_type="rmsnorm",
                        norm_eps=1e-6,
                    ),
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=4.0,
                    act_fn="gelu",
                    ff_type="ffn",
                    dtype="bfloat16",
                ),
                add_post_norm=False,
            ),
        ),
        "batch_size_per_device": 16,
        "gradient_accumulate_steps": 1,
        "model_axis_size": 1,
        "fsdp_axis_size": 8,
        "optimizer": optax.adamw(
            learning_rate=optax.schedules.warmup_exponential_decay_schedule(
                init_value=0.0, peak_value=5e-4, warmup_steps=100, decay_rate=0.99, transition_steps=1000
            ),
            b1=0.9,
            b2=0.98,
            eps=1e-9,
        ),
    },
    "30B": {
        "config": xLSTMLMModelConfig(
            vocab_size=50304,
            embedding_dim=8192,
            num_blocks=72,
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
                fsdp_modules=["Embed", "LMHead", "mLSTMBlock"],
                fsdp_min_weight_size=2**18,
                fsdp_gather_dtype="bfloat16",
                remat=["mLSTMBlock"],
                tp_async_dense=False,
            ),
            dtype="bfloat16",
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    num_heads=4,
                    vmap_qk=False,
                )
            ),
        ),
        "batch_size_per_device": 2,
        "gradient_accumulate_steps": 1,
        "model_axis_size": 4,
        "fsdp_axis_size": 2,
        "optimizer": optax.adamw(
            learning_rate=optax.schedules.warmup_exponential_decay_schedule(
                init_value=0.0, peak_value=5e-4, warmup_steps=100, decay_rate=0.99, transition_steps=1000
            ),
            b1=0.9,
            b2=0.98,
            eps=1e-9,
        ),
    },
    "70B": {
        "config": xLSTMLMModelConfig(
            vocab_size=50304,
            embedding_dim=12_288,
            num_blocks=76,
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
                fsdp_modules=["Embed", "LMHead", "mLSTMBlock"],
                fsdp_min_weight_size=2**18,
                fsdp_gather_dtype="bfloat16",
                remat=["mLSTMBlock"],
                tp_async_dense=False,
            ),
            dtype="bfloat16",
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    num_heads=4,
                    vmap_qk=False,
                )
            ),
        ),
        "batch_size_per_device": 1,
        "gradient_accumulate_steps": 1,
        "model_axis_size": 4,
        "fsdp_axis_size": 8,
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
