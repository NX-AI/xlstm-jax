import os

from xlstm_jax.distributed.xla_utils import simulate_CPU_devices

if os.environ["JAX_PLATFORMS"] == "cpu":
    NUM_DEVICES = 8
    simulate_CPU_devices(NUM_DEVICES)
else:
    NUM_DEVICES = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

import jax
import jax.numpy as jnp
import pytest

from xlstm_jax.distributed.single_gpu import Batch
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.trainer import TrainerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig

from ..helpers.mse_trainer import MSETrainer, ToyModel


@pytest.mark.parametrize("tp_size,fsdp_size", [(1, 1), (2, 2), (1, 8), (8, 1)])
def test_mse_trainer(tp_size: int, fsdp_size: int):
    """Tests training a simple model with MSE loss under different mesh configs."""
    trainer = MSETrainer(
        TrainerConfig(),
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
                name="constant",
                lr=1e-3,
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

    train_loader = [data_gen_fn(idx) for idx in range(250)]
    val_loader = train_loader[:20]
    final_metrics = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=2,
    )
    assert final_metrics is not None
    assert "val_1" in final_metrics and "val_2" in final_metrics
    assert set(final_metrics["val_1"].keys()) == {"loss", "l1_dist", "epoch_time"} and set(
        final_metrics["val_2"].keys()
    ) == set(final_metrics["val_1"].keys())
    assert final_metrics["val_1"]["loss"] < 0.1, "Validation loss should be less than 0.1."
    assert final_metrics["val_2"]["loss"] < 0.1, "Validation loss should be less than 0.1."
    assert (
        final_metrics["val_2"]["loss"] < final_metrics["val_1"]["loss"]
    ), "Validation loss should decrease over epochs."
    new_metrics = trainer.eval_model(val_loader, "eval", epoch_idx=3)
    assert new_metrics is not None
    assert new_metrics["loss"] == final_metrics["val_2"]["loss"], "Loss should be the same."
    assert new_metrics["l1_dist"] == final_metrics["val_2"]["l1_dist"], "L1 distance should be the same."
