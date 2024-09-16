import os

from xlstm_jax.distributed.xla_utils import simulate_CPU_devices

if os.environ["JAX_PLATFORMS"] == "cpu":
    NUM_DEVICES = 8
    simulate_CPU_devices(NUM_DEVICES)
else:
    NUM_DEVICES = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))


from xlstm_jax.distributed.single_gpu import Batch
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.trainer import TrainerConfig
from xlstm_jax.trainer.base.param_utils import flatten_dict
from xlstm_jax.trainer.callbacks.checkpointing import ModelCheckpointConfig
from xlstm_jax.trainer.logger import LoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from ..helpers.mse_trainer import MSETrainer, ToyModel


@pytest.mark.parametrize("tp_size,fsdp_size", [(1, 1), (2, 2), (1, 8), (8, 1)])
def test_checkpointing(tmp_path, tp_size: int, fsdp_size: int):
    """Tests checkpointing with ModelCheckpoint callback.

    The test trains a simple model with MSE loss under different mesh configs.
    We then check whether the checkpoints have been created as expected, load
    an older model, and reproduce the training and validation metrics.
    """
    log_dir = tmp_path / "test_checkpointing" / f"tp_{tp_size}_fsdp_{fsdp_size}"
    log_dir = log_dir.absolute().as_posix()
    trainer = MSETrainer(
        TrainerConfig(
            callbacks=(
                ModelCheckpointConfig(
                    monitor="loss",
                    save_top_k=2,
                    save_optimizer_state=True,
                    enable_async_checkpointing=False,
                ),
            ),
            logger=LoggerConfig(log_dir=log_dir),
        ),
        ModelConfig(
            model_class=ToyModel,
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
            labels=jax.ShapeDtypeStruct((8, 1), jnp.float32),
        ),
    )

    def data_gen_fn(idx: int) -> Batch:
        inputs = jax.random.normal(jax.random.PRNGKey(idx), (8, 64))
        labels = inputs[:, 0:1]
        return Batch(inputs=inputs, labels=labels)

    train_loader = [data_gen_fn(idx) for idx in range(100)]
    val_loader = train_loader[:20]
    final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=4,
    )
    assert final_metrics is not None
    assert all(f"val_{i}" in final_metrics for i in range(1, 5))
    assert (
        final_metrics["val_4"]["loss"] < final_metrics["val_3"]["loss"]
    ), "Validation loss should decrease over epochs."
    # Check that checkpoints have been created.
    assert os.path.exists(log_dir)
    assert os.path.exists(os.path.join(log_dir, "checkpoints/"))
    assert len(os.listdir(os.path.join(log_dir, "checkpoints/"))) == 2
    assert os.path.exists(os.path.join(log_dir, "checkpoints/checkpoint_3/"))
    assert os.path.exists(os.path.join(log_dir, "checkpoints/checkpoint_4/"))
    # Load an older model and reproduce the validation metric at this point.
    trainer.load_model(epoch_idx=3)
    new_metrics = trainer.eval_model(val_loader, "val", epoch_idx=3)
    assert new_metrics is not None
    assert new_metrics["loss"] == final_metrics["val_3"]["loss"], "Loss should be the same."
    # Train the model from this point and check that the previous training can be reproduced.
    new_final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=1,
    )
    assert new_final_metrics is not None
    assert new_final_metrics["val_1"]["loss"] == final_metrics["val_4"]["loss"], "Loss should be the same."


def test_loading_to_new_topology(tmp_path):
    """Tests checkpointing and loading to new topology.

    We hereby consider a scenario where we train a model with FSDP, and want to
    load it to a new topology with no FSDP. We check that the model can be loaded
    and achieves the same validation performance.
    """
    log_dir = tmp_path / "test_checkpointing" / f"fsdp_to_dp"
    log_dir = log_dir.absolute().as_posix()
    get_trainer = lambda fsdp_size: MSETrainer(
        TrainerConfig(
            callbacks=(
                ModelCheckpointConfig(
                    monitor="loss",
                    save_top_k=1,
                    save_optimizer_state=True,
                    enable_async_checkpointing=False,
                ),
            ),
            logger=LoggerConfig(log_dir=log_dir),
        ),
        ModelConfig(
            model_class=ToyModel,
            parallel=ParallelConfig(
                data_axis_size=-1,
                model_axis_size=1,
                fsdp_axis_size=fsdp_size,
                fsdp_min_weight_size=NUM_DEVICES,
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
            labels=jax.ShapeDtypeStruct((8, 1), jnp.float32),
        ),
    )
    trainer = get_trainer(8)

    def data_gen_fn(idx: int) -> Batch:
        inputs = jax.random.normal(jax.random.PRNGKey(idx), (8, 64))
        labels = inputs[:, 0:1]
        return Batch(inputs=inputs, labels=labels)

    train_loader = [data_gen_fn(idx) for idx in range(100)]
    val_loader = train_loader[:20]
    final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=4,
    )
    assert final_metrics is not None
    # Check that checkpoints have been created.
    assert os.path.exists(log_dir)
    assert os.path.exists(os.path.join(log_dir, "checkpoints/"))
    assert os.path.exists(os.path.join(log_dir, "checkpoints/checkpoint_4/"))
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
        new_trainer.load_model(epoch_idx=4)
        new_metrics = trainer.eval_model(val_loader, "val", epoch_idx=4)
        assert new_metrics is not None
        assert (
            new_metrics["loss"] == final_metrics["val_4"]["loss"]
        ), f"Loss should be the same, but did not match for FSDP size {fsdp_size}."
