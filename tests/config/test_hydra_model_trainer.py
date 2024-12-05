from pathlib import Path

import jax
import numpy as np
import pytest
from hydra import compose, initialize

from xlstm_jax.define_hydra_schemas import register_configs
from xlstm_jax.main_train import main_train


@pytest.mark.parametrize("model_name", ["llamaDebug", "mLSTMv1Debug"])
def test_hydra_trainer(tmp_path: Path, model_name: str):
    """Sets up the config via Hydra and calls regular main_train function."""

    register_configs()

    context_length = 32
    batch_size_per_device = 1
    global_batch_size = batch_size_per_device * pytest.num_devices
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=tiny_experiment_for_unit_testing",
                f"log_path={tmp_path}",
                f"data_train.ds1.global_batch_size={global_batch_size}",
                f"data_train.ds1.max_target_length={context_length}",
                f"model.name={model_name}",
            ],
            return_hydra_config=True,
        )

        # Call main train function. We need to set the output_dir, log_path and cmd_logging_name
        # manually because Hydra does not fill the hydra configuration when composing the config
        # compared to using the @hydra.main decorator. Is only relevant for the unit test.
        cfg.hydra.runtime.output_dir = tmp_path
        cfg.logger.log_path = tmp_path
        cfg.logger.cmd_logging_name = "unit_test"
        final_metrics = main_train(cfg=cfg)

    assert final_metrics is not None, "Output metrics were None."
    assert (
        "val_epoch_1" in final_metrics
    ), f"Missing validation epoch 1 key in final metrics, got instead {final_metrics}."
    assert not any(np.isnan(jax.tree.leaves(final_metrics))), f"Found NaNs in final metrics: {final_metrics}."
    val_acc = final_metrics["val_epoch_1"]["accuracy"]
    assert val_acc == 1.0, f"Expected accuracy to be 100%, but got {val_acc:.2%}."
