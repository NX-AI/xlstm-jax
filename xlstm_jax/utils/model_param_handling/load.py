import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax

from ...dataset import LLMBatch
from ...distributed.mesh_utils import initialize_mesh
from ...models import ModelConfig
from ...models.configs import ParallelConfig
from ...models.xlstm_parallel.xlstm_lm_model import xLSTMLMModel
from ...trainer.llm.trainer import LLMTrainer, LLMTrainerConfig
from ...trainer.optimizer import OptimizerConfig, SchedulerConfig

LOGGER = logging.getLogger(__name__)


def load_model_params_and_config_from_checkpoint(
    checkpoint_path: str | Path,
    return_config_as_dataclass: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load model parameters and config from a jax checkpoint.

    Args:
        checkpoint_path (str | Path): The path to the checkpoint file.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: The model parameters and the model config.
    """
    parallel = ParallelConfig(
        data_axis_name="dp",
        fsdp_axis_name="fsdp",
        model_axis_name="tp",
        pipeline_axis_name="pp",
        fsdp_modules=[],
        fsdp_gather_dtype="bfloat16",
        fsdp_min_weight_size=2**18,
        remat=[],
        fsdp_axis_size=1,
        model_axis_size=1,
        data_axis_size=1,
        tp_async_dense=False,
    )
    mesh = initialize_mesh(parallel_config=parallel)

    LOGGER.info("Mesh initialized.")
    LOGGER.info(f"Devices: {jax.devices()}")

    checkpoint_idx_path = Path(checkpoint_path)
    checkpoint_folder = checkpoint_idx_path.parent.parent
    checkpoint_idx = int(checkpoint_idx_path.name.split("_")[-1])

    path = Path(checkpoint_folder) / "checkpoints"
    metadata_path = (
        sorted(list(filter(lambda x: "checkpoint_" in str(x), path.iterdir())))[-1] / "metadata" / "metadata"
    )
    with open(metadata_path, encoding="utf8") as fp:
        metadata = fp.read()
    global_model_config = {"model_config": ModelConfig.from_metadata(metadata).model_config}
    LOGGER.info("Loading model config from metadata file.")
    parallel.fsdp_modules = global_model_config["model_config"].parallel.fsdp_modules
    parallel.remat = global_model_config["model_config"].parallel.remat

    xlstm_config = global_model_config["model_config"]
    xlstm_config.parallel = parallel
    xlstm_config.context_length = 128  # No specific reason for this value (should be multiple of 64)
    xlstm_config.__post_init__()

    # Create trainer with sub-configs.
    trainer = LLMTrainer(
        LLMTrainerConfig(
            callbacks=(),
            logger=None,
        ),
        ModelConfig(
            model_class=xLSTMLMModel,
            parallel=parallel,
            model_config=xlstm_config,
        ),
        # an optimizer is needed for the trainer (but actually not used), thus use one with minimal memory overhead
        OptimizerConfig(
            name="sgd",
            scheduler=SchedulerConfig(
                lr=1e-3,
            ),
        ),
        batch=LLMBatch.get_dtype_struct(1, 128),
        mesh=mesh,
    )

    LOGGER.info(f"Loading checkpoint from {checkpoint_folder}.")
    trainer.load_pretrained_model(
        Path(checkpoint_folder),
        step_idx=checkpoint_idx,
        load_best=False,
        load_optimizer=False,
        train_loader=None,
        val_loader=None,
    )

    params = trainer.state.params

    ret_cfg = xlstm_config if return_config_as_dataclass else asdict(xlstm_config)

    return params, ret_cfg
