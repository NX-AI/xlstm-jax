import argparse
import logging
from pathlib import Path
from typing import Any

import jax
import torch
from safetensors.torch import save_file

from xlstm_jax.dataset import LLMBatch
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModel
from xlstm_jax.trainer.llm.trainer import LLMTrainer, LLMTrainerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig
from xlstm_jax.utils.pytree_utils import flatten_pytree

LOGGER = logging.getLogger(__name__)

PyTree = Any


def params_to_safetensors(
    params: PyTree,
    output_path: Path,
    max_shard_size_bytes: int = 1 << 30,
    split_blocks: bool = True,
    blocks_layer_name: str = "blocks",
):
    """
    Convert an orbax checkpoint to one or more safetensors checkpoints, used in PyTorch / HuggingFace.

    Args:
        checkpoint_path: The Orbax checkpoint path.
        output_path: The resulting .safetensors file path. Should contain this file ending.
        max_shard_size_bytes: Size limit for a single .safetensors file in bytes, 0 means no limit.
        split_blocks: If to split a 'blocks' parameter as typically in PyTorch multi-block models are not merged.
        blocks_layer_name: Blocks layer name to split parameters by.

    Returns:
        None

    """

    # Convert JAX arrays to PyTorch tensors.
    params_flat = flatten_pytree(params)

    torch_state_dict = {key: torch.tensor(jax.device_get(value)) for key, value in params_flat.items()}

    if split_blocks:
        split_state_dict = {}
        for key, val in torch_state_dict.items():
            if "." + blocks_layer_name + "." not in key:
                split_state_dict[key] = val
            else:
                split_state_dict.update(
                    {
                        key.replace("." + blocks_layer_name + ".", "." + blocks_layer_name + f".{num}."): split_tensor
                        for num, split_tensor in enumerate(torch.unbind(val))
                    }
                )
        torch_state_dict = split_state_dict

    if max_shard_size_bytes is not None:
        keys = list(torch_state_dict)
        shard_size = 0
        idx = 0
        key_idx = 0
        torch_state_dict_shard = {}
        next_tensor = torch_state_dict[keys[key_idx]]
        next_size = int(next_tensor.dtype.itemsize * next_tensor.numel())
        while key_idx < len(list(keys)):
            while shard_size + next_size < max_shard_size_bytes and key_idx < len(list(keys)):
                torch_state_dict_shard[keys[key_idx]] = next_tensor
                shard_size += next_size
                key_idx += 1
                if key_idx < len(list(keys)):
                    next_tensor = torch_state_dict[keys[key_idx]]
                    next_size = int(next_tensor.dtype.itemsize * next_tensor.numel())
            if shard_size == 0:
                LOGGER.error("Single tensor is larger than maximal file size.")
                break
            save_file(torch_state_dict_shard, str(output_path).replace(".safetensors", f"_{idx}.safetensors"))
            torch_state_dict_shard = {}
            shard_size = 0
            idx += 1
    else:
        save_file(torch_state_dict, output_path)

    LOGGER.info(f"Checkpoint saved as {output_path}")


def main_convert_checkpoint(checkpoint_path: Path, output_path: Path, max_shard_size_bytes: int | None = 1 << 30):
    """
    Convert the checkpoint of a model, loading it first.

    Args:
        checkpoint_path: Path to the checkpoint (including step index).
        output_path: Path name for safetensors.
        max_shard_size_bytes: Maximal size for a safetensors shard. None means no sharding.

    Returns:
        None
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
    xlstm_config.context_length = 128
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

    params_to_safetensors(params, output_path, max_shard_size_bytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts a JAX orbax checkpoint to PyTorch safetensors. \n"
        'Use together with JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICE="" to not run out of memory.'
    )
    parser.add_argument("--checkpoint_dir", type=str, help="Directory of the orbax model checkpoint")
    parser.add_argument("--output_path", type=str, help="Output File for the safetensors checkpoint")
    parser.add_argument(
        "--max_shard_size",
        type=int,
        default=1 << 30,
        help="The maximal shard size of a single output file. If zero" "no sharding is applied.",
    )
    args = parser.parse_args()
    main_convert_checkpoint(
        checkpoint_path=Path(args.checkpoint_dir),
        output_path=Path(args.output_path),
        max_shard_size_bytes=args.max_shard_size,
    )
