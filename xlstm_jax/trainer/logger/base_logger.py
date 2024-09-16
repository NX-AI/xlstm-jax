import time
from dataclasses import dataclass
from pathlib import Path

from xlstm_jax.configs import ConfigDict
from xlstm_jax.trainer.metrics import HostMetrics, Metrics, get_metrics


@dataclass(kw_only=True, frozen=True)
class LoggerConfig(ConfigDict):
    log_every_n_steps: int = 1
    log_path: Path | None = None

    @property
    def log_dir(self) -> str:
        return self.log_path.as_posix()


class Logger:
    """Logger class to log metrics, images, etc."""

    def __init__(self, config: LoggerConfig):
        """Base Logger Class.

        Args:
            config (ConfigDict): The logger config.
        """
        self.config = config
        self.epoch_start_time = 0.0
        self.log_path = config.log_path

    def start_epoch(self, epoch: int, step: int, mode: str = "train"):
        """Starts a new epoch.

        Args:
            epoch (int): The index of the epoch.
            step (int): The index of the global training step.
            mode (str, optional): The logging mode. Should be in ["train", "val", "test"]. Defaults to "train".
        """
        self.epoch_start_time = time.time()

    def log_step(self, metrics: Metrics) -> Metrics:
        """Log metrics for a single step.

        Args:
            metrics: The metrics to log. Should follow the structure of the metrics in the metrics.py file.

        Returns:
            If the metrics are logged in this step, the metrics will be updated to reset all step-specific metrics.
            If the metrics are not logged in this step, the metrics will be returned unchanged.
        """
        # One can use `get_metrics` to put device metrics to host.
        return metrics

    def end_epoch(
        self,
        metrics: Metrics,
    ) -> tuple[Metrics, HostMetrics]:
        """Ends the current epoch and logs the epoch metrics.

        Args:
            metrics (Metrics): The metrics that should be logged in this epoch.

        Returns:
            The originally passed metric dict and potentially any other metrics that should be passed
            to callbacks later on.
        """
        epoch_time = time.time() - self.epoch_start_time
        metrics, host_metrics = get_metrics(metrics, reset_metrics=False)
        host_metrics["epoch_time"] = epoch_time
        return metrics, host_metrics

    def finalize(self, status: str):
        """Closes the logger.

        Args:
            status (str): The status of the training run (e.g. success, failure).
        """
