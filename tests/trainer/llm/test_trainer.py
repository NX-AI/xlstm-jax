import os

from xlstm_jax.distributed import simulate_CPU_devices

if os.environ["JAX_PLATFORMS"] == "cpu":
    NUM_DEVICES = 8
    simulate_CPU_devices(NUM_DEVICES)
else:
    NUM_DEVICES = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from xlstm_jax.dataset import Batch
from xlstm_jax.distributed import ModelParallelismWrapper, TPDense, shard_module_params, split_array_over_mesh
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from xlstm_jax.trainer import TrainerConfig
from xlstm_jax.trainer.callbacks.checkpointing import ModelCheckpointConfig
from xlstm_jax.trainer.llm.trainer import LLMTrainer
from xlstm_jax.trainer.logger import LoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig


class LLMToyModel(nn.Module):
    """LLM toy model for testing purposes.

    Contains one TP, one FSDP+TP, and one pure FSDP layer.
    """

    config: ModelConfig
    vocab_size: int = 50
    num_blocks: int = 2

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = False, **kwargs) -> jax.Array:
        tp_size = jax.lax.psum(1, self.config.parallel.model_axis_name)
        # Input layer with TP. All devices share the same input already (hence skip_communication),
        # but each will have a different output. We split the output features over the TP axis.
        x = ModelParallelismWrapper(
            module_fn=partial(nn.Embed, num_embeddings=self.vocab_size, features=32 // tp_size),
            model_axis_name=self.config.parallel.model_axis_name,
        )(x)
        x = nn.Dropout(rate=0.1)(x, deterministic=not train)
        for block_idx in range(self.num_blocks):
            x_skip = x
            # Example LayerNorm with model parallelism. Uses the model axis for parameter sharding and reduction of statistics.
            x = ModelParallelismWrapper(
                module_fn=partial(nn.LayerNorm, axis_name=self.config.parallel.model_axis_name),
                model_axis_name=self.config.parallel.model_axis_name,
            )(x)
            dense_fn = partial(
                TPDense,
                dense_fn=partial(nn.Dense, features=64 // tp_size),
                model_axis_name=self.config.parallel.model_axis_name,
                tp_mode="gather",
            )
            dense_fn = shard_module_params(
                dense_fn,
                axis_name=self.config.parallel.fsdp_axis_name,
                min_weight_size=NUM_DEVICES,
            )
            x = dense_fn(name=f"block_{block_idx}_in")(x)
            x = nn.Dropout(rate=0.1)(x, deterministic=not train)
            x = nn.swish(x)
            # Intermediate layer with FSDP+TP. Each device has a different input and need to gather first, and the output is split over the TP axis.
            dense_fn = partial(
                TPDense,
                dense_fn=partial(nn.Dense, features=32),
                model_axis_name=self.config.parallel.model_axis_name,
                tp_mode="scatter",
            )
            dense_fn = shard_module_params(
                dense_fn,
                axis_name=self.config.parallel.fsdp_axis_name,
                min_weight_size=NUM_DEVICES,
            )
            x = dense_fn(name=f"block_{block_idx}_out")(x)
        x = ModelParallelismWrapper(
            module_fn=partial(nn.LayerNorm, axis_name=self.config.parallel.model_axis_name),
            model_axis_name=self.config.parallel.model_axis_name,
        )(x)
        # For the output layer, we only use FSDP. We first gather all inputs, split them over the model and pipeline axis over the batch dimension.
        # Then, we apply the layer and calculate the outputs.
        x = jax.lax.all_gather(x, axis_name=self.config.parallel.model_axis_name, axis=-1, tiled=True)
        x = split_array_over_mesh(x, axis_name=self.config.parallel.pipeline_axis_name, split_axis=1)
        x = split_array_over_mesh(x, axis_name=self.config.parallel.model_axis_name, split_axis=1)
        x = shard_module_params(
            partial(nn.Dense, features=self.vocab_size),
            self.config.parallel.fsdp_axis_name,
            min_weight_size=NUM_DEVICES,
        )(name="out")(x)
        return x


@pytest.mark.parametrize("tp_size,fsdp_size", [(1, 1), (2, 2), (1, 8), (8, 1)])
def test_llm_trainer(tmp_path: Path, tp_size: int, fsdp_size: int):
    """Tests training a simple model with LLM loss under different mesh configs.

    Also reproduces the checkpointing test from the checkpointing test file for
    this new trainer.
    """
    trainer = LLMTrainer(
        TrainerConfig(
            callbacks=(
                ModelCheckpointConfig(
                    monitor="perplexity",
                    max_to_keep=4,
                    save_optimizer_state=True,
                    enable_async_checkpointing=True,
                ),
            ),
            logger=LoggerConfig(log_path=tmp_path),
            check_val_every_n_epoch=1,
        ),
        ModelConfig(
            model_class=LLMToyModel,
            parallel=ParallelConfig(
                data_axis_size=-1,
                model_axis_size=tp_size,
                fsdp_axis_size=fsdp_size,
                fsdp_min_weight_size=NUM_DEVICES,
            ),
        ),
        OptimizerConfig(
            name="adam",
            scheduler=SchedulerConfig(
                name="constant",
                lr=1e-4,
            ),
        ),
        batch=Batch(
            inputs=jax.ShapeDtypeStruct((8, 20), jnp.int32),
            labels=jax.ShapeDtypeStruct((8, 20), jnp.int32),
        ),
    )

    def data_gen_fn(idx: int) -> Batch:
        inputs = jax.random.randint(jax.random.PRNGKey(idx), (8, 20), minval=0, maxval=50)
        labels = jnp.mod(inputs + 1, 50)
        return Batch(inputs=inputs, labels=labels)

    train_loader = [data_gen_fn(idx) for idx in range(100)]
    val_loader = train_loader[:20]
    final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=5,
    )
    assert final_metrics is not None
    assert all(f"val_epoch_{i}" in final_metrics for i in range(1, 6))
    assert "perplexity" in final_metrics["val_epoch_1"]
    assert (
        final_metrics["val_epoch_5"]["perplexity"] < final_metrics["val_epoch_4"]["perplexity"]
    ), "Validation perplexity should decrease over epochs."
    # Check that checkpoints have been created.
    assert os.path.exists(tmp_path)
    assert os.path.exists(tmp_path / "checkpoints")
    assert len(os.listdir(tmp_path / "checkpoints/")) == 4
    # Check that validation performance can be reproduced with checkpoint.
    trainer.load_model(step_idx=200)
    new_metrics = trainer.eval_model(val_loader, "val", epoch_idx=2)
    assert "perplexity" in new_metrics
    assert "loss" in new_metrics
    assert (
        new_metrics["perplexity"] == final_metrics["val_epoch_2"]["perplexity"]
    ), f"Perplexity should match the loaded model: {new_metrics} versus {final_metrics['val_epoch_2']}"
    assert new_metrics["loss"] == final_metrics["val_epoch_2"]["loss"], "Loss should match the loaded model."
    # Check that we can continue training from this checkpoint as before.
    new_final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=5,
    )
    assert new_final_metrics is not None
    assert (
        new_final_metrics["val_epoch_5"]["perplexity"] == final_metrics["val_epoch_5"]["perplexity"]
    ), "Perplexity should match the loaded model."


@pytest.mark.parametrize("tp_size,fsdp_size", [(1, 1), (1, 8), (2, 2), (4, 1)])
def test_xlstm_training(tmp_path: Path, tp_size: int, fsdp_size: int):
    """Tests training a xLSTM model with debug configuration.

    Also reproduces the checkpointing test from the checkpointing test file for
    this new trainer.
    """
    # General hyperparameters.
    batch_size = 8
    context_length = 32
    parallel = ParallelConfig(
        data_axis_name="dp",
        fsdp_axis_name="fsdp",
        model_axis_name="tp",
        pipeline_axis_name="pp",
        fsdp_modules=("Embed", "LMHead", "mLSTMBlock"),
        fsdp_min_weight_size=2**8,
        fsdp_axis_size=fsdp_size,
        model_axis_size=tp_size,
        data_axis_size=-1,
    )
    # Define model config as before.
    xlstm_config = xLSTMLMModelConfig(
        vocab_size=20,
        embedding_dim=128,
        num_blocks=2,
        context_length=context_length,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        parallel=parallel,
        dtype=jnp.float32,
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                proj_factor=2.0,
                conv1d_kernel_size=4,
                num_heads=4,
                dropout=0.2,
                embedding_dim=128,
                context_length=context_length,
            )
        ),
    )

    trainer = LLMTrainer(
        TrainerConfig(
            callbacks=(
                ModelCheckpointConfig(
                    monitor="perplexity",
                    max_to_keep=4,
                    save_optimizer_state=True,
                    enable_async_checkpointing=True,
                ),
            ),
            logger=LoggerConfig(log_path=tmp_path),
        ),
        ModelConfig(
            model_class=xLSTMLMModel,
            parallel=parallel,
            model_config=xlstm_config,
        ),
        OptimizerConfig(
            name="adam",
            scheduler=SchedulerConfig(
                name="constant",
                lr=1e-4,
            ),
        ),
        batch=Batch(
            inputs=jax.ShapeDtypeStruct((batch_size, xlstm_config.context_length), jnp.int32),
            labels=jax.ShapeDtypeStruct((batch_size, xlstm_config.context_length), jnp.int32),
        ),
    )

    def data_gen_fn(idx: int) -> Batch:
        inputs = jax.random.randint(
            jax.random.PRNGKey(idx), (batch_size, xlstm_config.context_length), minval=0, maxval=xlstm_config.vocab_size
        )
        labels = jnp.mod(inputs + 1, xlstm_config.vocab_size)
        return Batch(inputs=inputs, labels=labels)

    train_loader = [data_gen_fn(idx) for idx in range(50)]
    val_loader = train_loader[:10]
    final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=5,
    )
    # Check that metrics are present as expected.
    assert final_metrics is not None
    assert all(f"val_epoch_{i}" in final_metrics for i in range(1, 6))
    assert "perplexity" in final_metrics["val_epoch_1"]
    assert (
        final_metrics["val_epoch_5"]["perplexity"] < final_metrics["val_epoch_4"]["perplexity"]
    ), "Validation perplexity should decrease over epochs."
    # Check that checkpoints have been created.
    assert os.path.exists(tmp_path)
    assert os.path.exists(tmp_path / "checkpoints")
    assert len(os.listdir(tmp_path / "checkpoints/")) == 4
    # Check that validation performance can be reproduced with checkpoint.
    trainer.load_model(step_idx=150)
    new_metrics = trainer.eval_model(val_loader, "val", epoch_idx=3)
    assert "perplexity" in new_metrics
    assert "loss" in new_metrics
    assert (
        new_metrics["perplexity"] == final_metrics["val_epoch_3"]["perplexity"]
    ), f"Perplexity should match the loaded model: {new_metrics} versus {final_metrics['val_epoch_3']}"
    assert new_metrics["loss"] == final_metrics["val_epoch_3"]["loss"], "Loss should match the loaded model."
    # Check that we can continue training from this checkpoint as before.
    new_final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=5,
    )
    assert new_final_metrics is not None
    assert (
        new_final_metrics["val_epoch_5"]["perplexity"] == final_metrics["val_epoch_5"]["perplexity"]
    ), "Perplexity should match the loaded model."
