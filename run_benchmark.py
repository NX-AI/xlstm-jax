import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
# A lot of XLA flags, most of them have no impact on performance.
os.environ['XLA_FLAGS'] = (
    # '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=true '
    # '--xla_gpu_enable_latency_hiding_scheduler=true '
    # '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
    # '--xla_gpu_enable_while_loop_double_buffering=true '
    # '--xla_gpu_enable_pipelined_all_gather=true '
    # '--xla_gpu_enable_pipelined_reduce_scatter=true '
    # '--xla_gpu_enable_pipelined_all_reduce=true '
    # '--xla_gpu_enable_all_gather_combine_by_dim=false '
    # '--xla_gpu_enable_reduce_scatter_combine_by_dim=false '
    # '--xla_gpu_all_gather_combine_threshold_bytes=8589934592 '
    # '--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 '
    # '--xla_gpu_all_reduce_combine_threshold_bytes=8589934592 '
    # '--xla_gpu_enable_pipelined_collectives=false '
    # '--xla_gpu_enable_pipelined_p2p=true '
    # '--xla_gpu_collective_permute_decomposer_threshold=1024 '
    '--xla_gpu_lhs_enable_gpu_async_tracker=true '
    '--xla_gpu_multi_streamed_windowed_einsum=true '
    '--xla_gpu_threshold_for_windowed_einsum_mib=0 '
    # '--xla_gpu_enable_nccl_user_buffers=true '
)
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
        "gradient_accumulate_steps": 1
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
        "batch_size": 256
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
        "model_axis_size": 1
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
                model_axis_name="tp",
                pipeline_axis_name="pp",
                # fsdp_modules=("Embed", "LMHead", "mLSTMBlock"),
                fsdp_modules=(),  # Not needed if TP 4
                fsdp_min_weight_size=2 ** 18,
                remat=("mLSTMBlock"),
                tp_async_dense=False,
            ),
            dtype=jnp.bfloat16,
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    num_heads=4,
                )
            )
        ), 
        "batch_size": 32, 
        "gradient_accumulate_steps": 1, 
        "model_axis_size": 4,
        "optimizer": optax.adamw(learning_rate=optax.schedules.warmup_exponential_decay_schedule(init_value=0.0, peak_value=5e-4, warmup_steps=100, decay_rate=0.99, transition_steps=1000), b1=0.9, b2=0.98, eps=1e-9)
    },
}

if __name__ == "__main__":
    benchmark_model(**MODEL_CONFIGS["7B"], num_steps=30, log_num_steps=3)