from pathlib import Path
from unittest.mock import patch

import pytest
from hydra import compose, initialize

from xlstm_jax.define_hydra_schemas import register_configs


def test_main_train_basic_exception(tmp_path: Path):
    """Test that exceptions in main_train are caught and logged."""
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
                f"data.global_batch_size={global_batch_size}",
                f"data.max_target_length={context_length}",
            ],
            return_hydra_config=True,
        )

        # Call main train function. We need to set the output_dir, log_path and cmd_logging_name
        # manually because Hydra does not fill the hydra configuration when composing the config
        # compared to using the @hydra.main decorator. Is only relevant for the unit test.
        cfg.logger.cmd_logging_name = "unit_test"
        cfg.hydra.runtime.output_dir = tmp_path
        cfg.logger.log_path = tmp_path

        # Mock the train function to raise an exception
        with patch("xlstm_jax.main_train.main_train") as mock_train:
            mock_train.side_effect = ValueError("Test error in training")

            # Test that the exception is caught.
            with pytest.raises(ValueError) as exc_info:
                mock_train(cfg=cfg)

            # Verify the error message
            assert str(mock_train.side_effect) in str(exc_info.value)
