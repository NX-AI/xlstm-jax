import logging
from dataclasses import dataclass

from torch.utils.tensorboard import SummaryWriter

from xlstm_jax.configs import ConfigDict
from xlstm_jax.trainer.base.param_utils import flatten_dict
from xlstm_jax.trainer.metrics import HostMetrics

from .base_logger import Logger, LoggerTool, LoggerToolsConfig

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=False)
class TensorBoardLoggerConfig(LoggerToolsConfig):
    """
    Configuration for the TensorBoard logger tool.

    Attributes:
        tb_flush_secs: The frequency at which to flush the tensorboard logs.
        tb_max_queue: The maximum number of items to queue before flushing.
        tb_new_style: Whether to use the new style of logging. See PyTorch
            SummaryWriter documentation for more information.
        log_dir: The directory to use for the logs. Is added to the
            log_path in the logger
    """

    tb_flush_secs: int = 120
    tb_max_queue: int = 10
    tb_new_style: bool = False
    log_dir: str = "tensorboard"

    def create(self, logger: Logger) -> "LoggerTool":
        """Creates the TensorBoard logger tool."""
        return TensorBoardLogger(self, logger)


class TensorBoardLogger(LoggerTool):
    def __init__(self, config: TensorBoardLoggerConfig, logger: Logger):
        """
        TensorBoard logger tool to log metrics to disk.

        Args:
            config: The config for the TensorBoard logger.
            logger: The logger object.
        """
        self.config = config
        self.config_to_log = None
        self.logger = logger
        self.writer: SummaryWriter = None

    def log_config(self, config: ConfigDict | dict[str, ConfigDict]):
        """
        Log the config to TensorBoard.

        If the writer is not set up, the config will be saved and logged
        when the writer is set up.

        Args:
            config: The config to log.
        """
        if isinstance(config, ConfigDict):
            config = config.to_dict()
        elif isinstance(config, dict):
            config = {k: v.to_dict() for k, v in config.items()}
        config = flatten_dict(config)
        config = {k: v if isinstance(v, (int, float, str, bool)) else str(v) for k, v in config.items()}
        self.config_to_log = config
        self._log_config()

    def _log_config(self):
        """Logs stored config to TensorBoard if writer is set up."""
        if self.writer is not None:
            self.writer.add_hparams(
                hparam_dict=self.config_to_log,
                metric_dict={},
                run_name="../tensorboard",
            )

    def setup(self):
        """
        Set up the TensorBoard logger.

        If the writer is already set up, this function skips the setup.
        """
        if self.writer is not None:
            return

        LOGGER.info("Setting up TensorBoard logging.")
        self.writer = SummaryWriter(
            log_dir=self.logger.log_path / self.config.log_dir,
            flush_secs=self.config.tb_flush_secs,
            max_queue=self.config.tb_max_queue,
        )
        self._log_config()

    def log_metrics(self, metrics: HostMetrics, step: int, epoch: int, mode: str):
        """
        Log a single metric dictionary in the TensorBoard logger.

        Args:
            metrics: The metrics to log.
            step: The current step.
            epoch: The current epoch. Currently unused.
            mode: The mode of logging. Commonly "train", "val", or "test". Is used as prefix for the metric keys.
        """
        del epoch
        for key, value in metrics.items():
            self.writer.add_scalar(f"{mode}/{key}", value, step, new_style=self.config.tb_new_style)

    def finalize(self, status: str):
        """
        Close the TensorBoard logger.

        Args:
            status: The status of the training run (e.g. success, failure).
        """
        del status
        LOGGER.info("Finishing TensorBoard logging.")
        self.writer.flush()
        self.writer.close()
