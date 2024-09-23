import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from xlstm_jax.dataset import Batch
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.trainer import TrainerConfig
from xlstm_jax.trainer.base.param_utils import flatten_dict
from xlstm_jax.trainer.callbacks.checkpointing import ModelCheckpointConfig
from xlstm_jax.trainer.logger import LoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig

from ..helpers.mse_trainer import MSETrainer, ToyModel


@pytest.mark.parametrize("tp_size,fsdp_size", [(1, 1), (2, 2), (1, 8), (8, 1)])
def test_checkpointing_per_epoch(tmp_path: Path, tp_size: int, fsdp_size: int):
    """
    Tests checkpointing with ModelCheckpoint callback with per-epoch eval.

    The test trains a simple model with MSE loss under different mesh configs. We then check whether the checkpoints
    have been created as expected, load an older model, and reproduce the training and validation metrics.
    """
    log_path = tmp_path / "test_checkpointing_per_epoch" / f"tp_{tp_size}_fsdp_{fsdp_size}"
    trainer = MSETrainer(
        TrainerConfig(
            callbacks=(
                ModelCheckpointConfig(
                    monitor="loss",
                    max_to_keep=2,
                    save_optimizer_state=True,
                    enable_async_checkpointing=False,
                ),
            ),
            logger=LoggerConfig(log_path=log_path),
            check_val_every_n_epoch=1,
        ),
        ModelConfig(
            model_class=ToyModel,
            parallel=ParallelConfig(
                data_axis_size=-1,
                model_axis_size=tp_size,
                fsdp_axis_size=fsdp_size,
                fsdp_min_weight_size=pytest.num_devices,
            ),
        ),
        OptimizerConfig(
            name="adam",
            scheduler=SchedulerConfig(
                name="exponential_decay",
                lr=1e-3,
                decay_steps=500,
                warmup_steps=50,
                cooldown_steps=50,
                end_lr_factor=0.1,
            ),
        ),
        batch=Batch(
            inputs=jax.ShapeDtypeStruct((8, 64), jnp.float32),
            targets=jax.ShapeDtypeStruct((8, 1), jnp.float32),
        ),
    )

    def data_gen_fn(idx: int) -> Batch:
        inputs = jax.random.normal(jax.random.PRNGKey(idx), (8, 64))
        labels = inputs[:, 0:1]
        return Batch(inputs=inputs, targets=labels)

    train_loader = [data_gen_fn(idx) for idx in range(100)]
    val_loader = train_loader[:20]
    final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=4,
    )
    assert final_metrics is not None
    assert all(
        f"val_epoch_{i}" in final_metrics for i in range(1, 5)
    ), f"Validation metrics should be present for all epochs, got {final_metrics}."
    assert (
        final_metrics["val_epoch_4"]["loss"] < final_metrics["val_epoch_3"]["loss"]
    ), "Validation loss should decrease over epochs."
    # Check that checkpoints have been created.
    assert log_path.exists()
    checkpoint_path = log_path / "checkpoints"
    assert checkpoint_path.exists()
    assert len(list(checkpoint_path.glob("*"))) == 2
    assert (checkpoint_path / "checkpoint_300").exists()
    assert (checkpoint_path / "checkpoint_400").exists()
    # Load an older model and reproduce the validation metric at this point.
    trainer.load_model(step_idx=300)
    new_metrics = trainer.eval_model(val_loader, "val", epoch_idx=3)
    assert new_metrics is not None
    assert new_metrics["loss"] == final_metrics["val_epoch_3"]["loss"], "Loss should be the same."
    # Train the model from this point and check that the previous training can be reproduced.
    new_final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=4,
    )
    assert new_final_metrics is not None
    assert new_final_metrics["val_epoch_4"]["loss"] == final_metrics["val_epoch_4"]["loss"], "Loss should be the same."


@pytest.mark.parametrize("tp_size,fsdp_size", [(1, 1), (2, 2), (1, 8), (8, 1)])
def test_checkpointing_per_step(tmp_path: Path, tp_size: int, fsdp_size: int):
    """
    Tests checkpointing with ModelCheckpoint callback with per-step eval.

    The test trains a simple model with MSE loss under different mesh configs. We then check whether the checkpoints
    have been created as expected, load an older model, and reproduce the training and validation metrics.
    """
    log_path = tmp_path / "test_checkpointing_per_step" / f"tp_{tp_size}_fsdp_{fsdp_size}"
    trainer = MSETrainer(
        TrainerConfig(
            callbacks=(
                ModelCheckpointConfig(
                    monitor="loss",
                    max_to_keep=2,
                    save_optimizer_state=True,
                    enable_async_checkpointing=False,
                ),
            ),
            logger=LoggerConfig(log_path=log_path),
            check_val_every_n_steps=100,
        ),
        ModelConfig(
            model_class=ToyModel,
            parallel=ParallelConfig(
                data_axis_size=-1,
                model_axis_size=tp_size,
                fsdp_axis_size=fsdp_size,
                fsdp_min_weight_size=pytest.num_devices,
            ),
        ),
        OptimizerConfig(
            name="adam",
            scheduler=SchedulerConfig(
                name="exponential_decay",
                lr=1e-3,
                decay_steps=500,
                warmup_steps=50,
                cooldown_steps=50,
                end_lr_factor=0.1,
            ),
        ),
        batch=Batch(
            inputs=jax.ShapeDtypeStruct((8, 64), jnp.float32),
            targets=jax.ShapeDtypeStruct((8, 1), jnp.float32),
        ),
    )

    def data_gen_fn(idx: int) -> Batch:
        inputs = jax.random.normal(jax.random.PRNGKey(idx), (8, 64))
        labels = inputs[:, 0:1]
        return Batch(inputs=inputs, targets=labels)

    train_loader = [data_gen_fn(idx) for idx in range(100)]
    val_loader = train_loader[:20]
    train_loader = itertools.cycle(train_loader)
    final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_train_steps=400,
    )
    assert final_metrics is not None
    assert all(
        f"val_step_{i * 100}" in final_metrics for i in range(1, 5)
    ), f"Validation metrics should be present for all steps, got {final_metrics}."
    assert (
        final_metrics["val_step_400"]["loss"] < final_metrics["val_step_300"]["loss"]
    ), "Validation loss should decrease over epochs."
    # Check that checkpoints have been created.
    assert log_path.exists()
    checkpoint_path = log_path / "checkpoints"
    assert checkpoint_path.exists()
    assert len(list(checkpoint_path.glob("*"))) == 2
    assert (checkpoint_path / "checkpoint_300").exists()
    assert (checkpoint_path / "checkpoint_400").exists()
    # Load an older model and reproduce the validation metric at this point.
    trainer.load_model(step_idx=300)
    new_metrics = trainer.eval_model(val_loader, "val", epoch_idx=3)
    assert new_metrics is not None
    assert new_metrics["loss"] == final_metrics["val_step_300"]["loss"], "Loss should be the same."
    # Train the model from this point and check that the previous training can be reproduced.
    new_final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_train_steps=400,
    )
    assert new_final_metrics is not None
    assert (
        new_final_metrics["val_step_400"]["loss"] == final_metrics["val_step_400"]["loss"]
    ), "Loss should be the same."


def test_checkpointing_per_epoch_and_step(tmp_path: Path):
    """
    Tests checkpointing with ModelCheckpoint callback with both per-epoch and per-step eval.

    The test trains a simple model with MSE loss under different mesh configs. We then check whether the checkpoints
    have been created as expected, load an older model, and reproduce the training and validation metrics.
    """
    tp_size = 1
    fsdp_size = 1
    log_path = tmp_path / "test_checkpointing_per_epoch_and_step" / f"tp_{tp_size}_fsdp_{fsdp_size}"
    trainer = MSETrainer(
        TrainerConfig(
            callbacks=(
                ModelCheckpointConfig(
                    max_to_keep=None,
                    save_optimizer_state=False,
                    enable_async_checkpointing=False,
                ),
            ),
            logger=LoggerConfig(log_path=log_path),
            check_val_every_n_steps=80,
            check_val_every_n_epoch=1,
        ),
        ModelConfig(
            model_class=ToyModel,
            parallel=ParallelConfig(
                data_axis_size=-1,
                model_axis_size=tp_size,
                fsdp_axis_size=fsdp_size,
                fsdp_min_weight_size=pytest.num_devices,
            ),
        ),
        OptimizerConfig(
            name="adam",
            scheduler=SchedulerConfig(
                name="exponential_decay",
                lr=1e-3,
                decay_steps=500,
                warmup_steps=50,
                cooldown_steps=50,
                end_lr_factor=0.1,
            ),
        ),
        batch=Batch(
            inputs=jax.ShapeDtypeStruct((8, 64), jnp.float32),
            targets=jax.ShapeDtypeStruct((8, 1), jnp.float32),
        ),
    )

    def data_gen_fn(idx: int) -> Batch:
        inputs = jax.random.normal(jax.random.PRNGKey(idx), (8, 64))
        labels = inputs[:, 0:1]
        return Batch(inputs=inputs, targets=labels)

    train_loader = [data_gen_fn(idx) for idx in range(100)]
    val_loader = train_loader[:20]
    final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=4,
    )
    assert final_metrics is not None
    assert all(
        f"val_step_{i}" in final_metrics for i in range(80, 401, 80)
    ), f"Validation metrics should be present for all steps, got {final_metrics}."
    assert all(
        f"val_epoch_{i}" in final_metrics for i in range(1, 5)
    ), f"Validation metrics should be present for all epochs, got {final_metrics}."
    assert (
        final_metrics["val_epoch_4"]["loss"] < final_metrics["val_epoch_3"]["loss"]
    ), "Validation loss should decrease over epochs."
    assert (
        final_metrics["val_step_400"]["loss"] < final_metrics["val_step_320"]["loss"]
    ), "Validation loss should decrease over steps."
    assert (
        final_metrics["val_step_400"]["loss"] == final_metrics["val_epoch_4"]["loss"]
    ), "Validation loss should be equal for last step and epoch."
    # Check that checkpoints have been created.
    assert log_path.exists()
    checkpoint_path = log_path / "checkpoints"
    assert checkpoint_path.exists()
    assert len(list(checkpoint_path.glob("*"))) == 8
    for step in set(list(range(80, 401, 80)) + list(range(100, 500, 100))):
        assert (checkpoint_path / f"checkpoint_{step}").exists()


def test_loading_to_new_topology(tmp_path: Path):
    """
    Tests checkpointing and loading to new topology.

    We hereby consider a scenario where we train a model with FSDP, and want to load it to a new topology with no FSDP.
    We check that the model can be loaded and achieves the same validation performance.
    """
    log_path = tmp_path / "test_checkpointing" / "fsdp_to_dp"

    def get_trainer(fsdp_size: int):
        trainer_config = TrainerConfig(
            callbacks=(
                ModelCheckpointConfig(
                    monitor="loss",
                    max_to_keep=1,
                    save_optimizer_state=True,
                    enable_async_checkpointing=False,
                ),
            ),
            logger=LoggerConfig(log_path=log_path),
            check_val_every_n_epoch=1,
        )
        model_config = ModelConfig(
            model_class=ToyModel,
            parallel=ParallelConfig(
                data_axis_size=-1,
                model_axis_size=1,
                fsdp_axis_size=fsdp_size,
                fsdp_min_weight_size=pytest.num_devices,
            ),
        )
        optimizer_config = OptimizerConfig(
            name="adam",
            scheduler=SchedulerConfig(
                name="exponential_decay",
                lr=1e-3,
                decay_steps=500,
                warmup_steps=50,
                cooldown_steps=50,
                end_lr_factor=0.1,
            ),
        )
        batch = Batch(
            inputs=jax.ShapeDtypeStruct((8, 64), jnp.float32),
            targets=jax.ShapeDtypeStruct((8, 1), jnp.float32),
        )
        return MSETrainer(trainer_config, model_config, optimizer_config, batch)

    def data_gen_fn(idx: int) -> Batch:
        inputs = jax.random.normal(jax.random.PRNGKey(idx), (8, 64))
        labels = inputs[:, 0:1]
        return Batch(inputs=inputs, targets=labels)

    trainer = get_trainer(8)
    train_loader = [data_gen_fn(idx) for idx in range(100)]
    val_loader = train_loader[:20]
    final_metrics = trainer.train_model(train_loader, val_loader, num_epochs=4)
    assert final_metrics is not None
    # Check that checkpoints have been created.
    assert log_path.exists()
    checkpoint_path = log_path / "checkpoints"
    assert checkpoint_path.exists()
    assert (checkpoint_path / "checkpoint_400").exists()
    # trainer.load_model(epoch_idx=4)
    # Load the model into different FSDP sizes
    for fsdp_size in [1, 2, 4, 8]:
        new_trainer = get_trainer(fsdp_size)
        new_params = flatten_dict(new_trainer.state.params)
        old_params = flatten_dict(trainer.state.params)
        for key in old_params:
            assert key in new_params, f"Key {key} not found in new model."
            if isinstance(old_params[key], nn.Partitioned):
                assert isinstance(
                    new_params[key], nn.Partitioned
                ), f"Key {key} should be Partitioned in the new trainer."
                assert old_params[key].value.shape == new_params[key].value.shape, f"Shape mismatch for key {key}."
                assert old_params[key].names == new_params[key].names, f"Names mismatch for key {key}."
            else:
                assert not isinstance(
                    new_params[key], nn.Partitioned
                ), f"Key {key} should not be Partitioned in the new trainer."
                assert old_params[key].shape == new_params[key].shape, f"Shape mismatch for key {key}."
        new_trainer.load_model(step_idx=400)
        new_metrics = trainer.eval_model(val_loader, "val", epoch_idx=4)
        assert new_metrics is not None
        assert (
            new_metrics["loss"] == final_metrics["val_epoch_4"]["loss"]
        ), f"Loss should be the same, but did not match for FSDP size {fsdp_size}."
