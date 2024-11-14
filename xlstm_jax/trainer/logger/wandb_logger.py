import logging
from dataclasses import dataclass, field
from typing import Any

import wandb

from xlstm_jax.configs import ConfigDict
from xlstm_jax.trainer.metrics import HostMetrics
from xlstm_jax.utils import flatten_dict

from .base_logger import Logger, LoggerTool, LoggerToolsConfig

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=False)
class WandBLoggerConfig(LoggerToolsConfig):
    """
    Configuration for the WandB logger tool.

    Attributes:
        wb_entity: The WandB entity to log to.
        wb_project: The WandB project to log to.
        wb_host: The WandB host to log to.
        wb_key: The WandB API key to use. If None, the key will be read from the environment.
        wb_name: The name of the run.
        wb_notes: Notes to add to the run.
        wb_settings: Settings to pass to the WandB run.
        wb_tags: Tags to add to the run.
        log_dir: The directory to log to.
    """

    wb_entity: str = "xlstm"
    wb_project: str = "xlstm_nxai"
    wb_host: str = "https://api.wandb.ai"
    wb_key: str | None = None
    wb_name: str | None = None
    wb_notes: str | None = None
    wb_settings: dict[str, Any] = field(default_factory=lambda: {"start_method": "fork"})
    wb_tags: list[str] = field(default_factory=list)
    wb_resume_id: str | None = None
    log_dir: str = "wandb"

    def create(self, logger: Logger) -> "LoggerTool":
        """Create a WandB logger tool."""
        return WandBLogger(self, logger)


class WandBLogger(LoggerTool):
    def __init__(self, config: WandBLoggerConfig, logger: Logger):
        """
        WandB logger tool to log metrics to the cloud.

        Args:
            config: The config for the WandB logger.
            logger: The logger object.
        """
        self.config = config
        self.config_to_log = None
        self.logger = logger
        self.log_path = self.logger.log_path / self.config.log_dir
        self.wandb_run = None

    def log_config(self, config: ConfigDict | dict[str, ConfigDict]):
        """
        Log the config to WandB.

        If the run is not set up, the config will be saved and logged
        when the run is set up.

        Args:
            config: The config to log.
        """
        if isinstance(config, ConfigDict):
            config = config.to_dict()
        elif isinstance(config, dict):
            config = {k: v.to_dict() for k, v in config.items()}
        config = flatten_dict(config)
        self.config_to_log = config
        if self.wandb_run is not None:
            self.wandb_run.config.update(config)

    def setup(self):
        """
        Set up the WandB logger.

        If the run is already set up, this function skips the setup.
        """
        if self.wandb_run is not None:
            return

        LOGGER.info("Setting up WandB logging.")
        self.log_path.mkdir(parents=True, exist_ok=True)
        wandb.login(host=self.config.wb_host, key=self.config.wb_key)
        if self.config.wb_resume_id is None:
            self.wandb_run = wandb.init(
                entity=self.config.wb_entity,
                project=self.config.wb_project,
                name=self.config.wb_name,
                tags=self.config.wb_tags,
                notes=self.config.wb_notes,
                dir=self.log_path,
                config=self.config_to_log,
                settings=wandb.Settings(**self.config.wb_settings),
            )
        else:
            LOGGER.info(f"WandB: Resuming experiment {self.config.wb_resume_id}")
            self.wandb_run = wandb.init(
                entity=self.config.wb_entity,
                project=self.config.wb_project,
                id=self.config.wb_resume_id,
                resume="allow",
                settings=wandb.Settings(**self.config.wb_settings),
            )
        LOGGER.info(f"WandB mode: {self.wandb_run.settings.mode}")

    def log_metrics(self, metrics: HostMetrics, step: int, epoch: int, mode: str):
        """
        Log a single metric dictionary in the WandB logger.

        Args:
            metrics: The metrics to log.
            step: The current step.
            epoch: The current epoch. Currently unused.
            mode: The current mode. Will be used as a prefix for the metrics.
        """
        del epoch
        self.wandb_run.log({f"{mode}/": metrics}, step=step)

    def finalize(self, status: str):
        """
        Closes the WandB logger.

        Args:
            status: The status of the training run (e.g. success, failure).
        """
        LOGGER.info("Finishing WandB logging.")
        self.wandb_run.finish(exit_code=int(status != "success"))
