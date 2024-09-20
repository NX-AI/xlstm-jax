import jax
import jax.numpy as jnp
import pytest

from xlstm_jax.dataset import Batch
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.trainer import TrainerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig

from ..helpers.mse_trainer import MSETrainer, ToyModel


@pytest.mark.parametrize("tp_size,fsdp_size", [(1, 1), (2, 2), (1, 8), (8, 1)])
def test_mse_trainer(tp_size: int, fsdp_size: int):
    """Tests training a simple model with MSE loss under different mesh configs."""
    trainer = MSETrainer(
        TrainerConfig(
            check_val_every_n_epoch=1,
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
                name="constant",
                lr=1e-3,
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

    train_loader = [data_gen_fn(idx) for idx in range(250)]
    val_loader = train_loader[:20]
    final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=2,
    )
    assert final_metrics is not None
    epoch_keys = [f"val_epoch_{i}" for i in range(1, 3)]
    step_keys = [f"val_step_{i}" for i in range(100, 501, 100)]
    assert all(
        k in final_metrics for k in epoch_keys
    ), f"Validation metrics should be logged at the end of each epoch, instead got keys: {final_metrics.keys()}."
    assert all(
        k in final_metrics for k in step_keys
    ), f"Validation metrics should be logged at the end of each step, instead got keys: {final_metrics.keys()}."
    assert set(final_metrics[epoch_keys[0]].keys()) == {
        "loss",
        "l1_dist",
        "epoch_time",
    }, f"Keys should be the same as specified in the loss function, but got {final_metrics[epoch_keys[0]].keys()}."
    assert set(final_metrics[epoch_keys[0]].keys()) == set(
        final_metrics[epoch_keys[1]].keys()
    ), f"Keys should be the same for all validation metrics, but got {final_metrics[epoch_keys[1]].keys()}."
    assert set(final_metrics[epoch_keys[0]].keys()) == set(
        final_metrics[step_keys[1]].keys()
    ), f"Keys should be the same for all validation metrics, but got {final_metrics[step_keys[1]].keys()}."
    assert final_metrics[epoch_keys[0]]["loss"] < 0.1, "Validation loss should be less than 0.1."
    assert final_metrics[epoch_keys[1]]["loss"] < 0.1, "Validation loss should be less than 0.1."
    assert (
        final_metrics[epoch_keys[1]]["loss"] < final_metrics[epoch_keys[0]]["loss"]
    ), "Validation loss should decrease over epochs."
    new_metrics = trainer.eval_model(val_loader, "eval", epoch_idx=2)
    assert new_metrics is not None
    assert new_metrics["loss"] == final_metrics[epoch_keys[1]]["loss"], "Loss should be the same."
    assert new_metrics["l1_dist"] == final_metrics[epoch_keys[1]]["l1_dist"], "L1 distance should be the same."
