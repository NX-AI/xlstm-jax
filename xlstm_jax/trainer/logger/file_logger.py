#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import json
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import yaml

from xlstm_jax.common_types import HostMetrics
from xlstm_jax.configs import ConfigDict

from .base_logger import Logger, LoggerTool, LoggerToolsConfig

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=False)
class FileLoggerConfig(LoggerToolsConfig):
    """
    Configuration for the file logger tool.

    Attributes:
        log_step_key: The key to use for the step in the logs.
        log_epoch_key: The key to use for the epoch in the logs.
        config_format: The format to use when logging the config.
        log_dir: The directory to use for the logs. Is added to the
            log_path in the logger.
    """

    log_step_key: str = "log_step"
    log_epoch_key: str = "log_epoch"
    config_format: str = "json"
    log_dir: str = "file_logs"

    def __post_init__(self):
        allowed_config_formats = ["json", "yaml", "pickle"]
        assert self.config_format in allowed_config_formats, f"config_format must be one of {allowed_config_formats}"

    def create(self, logger: Logger) -> "LoggerTool":
        """Creates the file logger tool."""
        return FileLogger(self, logger)


class FileLogger(LoggerTool):
    def __init__(self, config: FileLoggerConfig, logger: Logger):
        """
        File logger tool to log metrics to disk.

        Args:
            config: The config for the file logger.
            logger: The logger object.
        """
        self.config = config
        self.config_to_log = None
        self.logger = logger
        self.log_path = self.logger.log_path / self.config.log_dir
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.logs = defaultdict(list)

    def log_config(self, config: ConfigDict | dict[str, ConfigDict]):
        """
        Log the config to disk.

        Args:
            config: The config to log.
        """
        if self.config.config_format in ["json", "yaml"]:
            if isinstance(config, ConfigDict):
                config = config.to_dict()
            elif isinstance(config, dict):
                config = {k: v.to_dict() for k, v in config.items()}
        if self.config.config_format == "json":
            with open(self.log_path / "config.json", "w") as f:
                json.dump(config, f, indent=4)
        elif self.config.config_format == "yaml":
            with open(self.log_path / "config.yaml", "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        elif self.config.config_format == "pickle":
            with open(self.log_path / "config.pkl", "wb") as f:
                pickle.dump(config, f)

    def setup(self):
        """Set up the file logger."""
        self.logs = defaultdict(list)
        LOGGER.info(f"File logger set up to log at {self.log_path}.")

    def log_metrics(self, metrics: HostMetrics, step: int, epoch: int, mode: str):
        """
        Log a single metric dictionary in the file logger.

        The metrics are logged in a list and saved to disk at the end.

        Args:
            metrics: The metrics to log.
            step: The current step.
            epoch: The current epoch.
            mode: The mode of logging. Commonly "train", "val", or "test".
        """
        metrics = metrics.copy()
        metrics[self.config.log_step_key] = step
        metrics[self.config.log_epoch_key] = epoch
        self.logs[mode].append(metrics)

    def finalize(self, status: str):
        """
        Finalize the file logger.

        Writes out the logs to disk.

        Args:
            status: The status of the training run (e.g. success, failure).
        """
        del status
        LOGGER.info("Finishing file logging.")
        for key in self.logs:
            df = pd.DataFrame(self.logs[key])
            df.to_csv(self.log_path / f"metrics_{key}.csv", index=False)
