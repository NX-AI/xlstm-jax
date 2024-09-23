import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from xlstm_jax.dataset import Batch
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.trainer import TrainerConfig
from xlstm_jax.trainer.logger import LoggerConfig, TensorBoardLoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig

from ..helpers.mse_trainer import MSETrainer, ToyModel

LOGGER = logging.getLogger(__name__)


@pytest.mark.parametrize("tb_dir", ["tensorboard", "tb_out", "tb_logs"])
def test_tensorboard_logging_mse_trainer(tmp_path: Path, tb_dir: str):
    """Tests logging for example trainer."""
    log_path = tmp_path / "logs"
    trainer = MSETrainer(
        TrainerConfig(
            check_val_every_n_epoch=1,
            check_val_every_n_steps=100,
            logger=LoggerConfig(
                log_every_n_steps=20,
                log_path=log_path,
                log_tools=[TensorBoardLoggerConfig(log_dir=tb_dir, tb_flush_secs=1)],
            ),
        ),
        ModelConfig(
            model_class=ToyModel,
            parallel=ParallelConfig(
                data_axis_size=-1,
                model_axis_size=2,
                fsdp_axis_size=2,
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
        targets = inputs[:, 0:1]
        return Batch(inputs=inputs, targets=targets)

    train_loader = [data_gen_fn(idx) for idx in range(250)]
    val_loader = train_loader[:20]
    test_loader = train_loader[20:40]
    _ = trainer.train_model(
        train_loader,
        val_loader,
        test_loader=test_loader,
        num_epochs=2,
    )
    assert log_path.exists()
    assert (log_path / "output.log").exists()
    assert (log_path / tb_dir).exists(), f"Expected TensorBoard logging directory {log_path / tb_dir} to exist"
    assert (
        len(list((log_path / tb_dir).glob("events*"))) >= 1
    ), f"Expected 1 event file in {log_path / tb_dir}, but was empty."
