from model_parallel.xlstm_lm_model import xLSTMLMModelConfig
from model_parallel.blocks.mlstm.block import mLSTMBlockConfig
from model_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from model_parallel.benchmark import benchmark_model
from model_parallel.utils import ParallelConfig
import jax.numpy as jnp


if __name__ == "__main__":
    config = xLSTMLMModelConfig(
        vocab_size=100,
        embedding_dim=16,
        num_blocks=4,
        context_length=128,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        parallel=ParallelConfig(
            data_axis_name="batch",
            model_axis_name="model",
            pipeline_axis_name="pipeline",
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
    )
    benchmark_model(config, model_axis_size=1, pipeline_axis_size=1, num_steps=100)