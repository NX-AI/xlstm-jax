import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf

from mlstm_simple_torch.mlstm_simple.model import mLSTM, mLSTMConfig

from .convert_checkpoint import convert_orbax_checkpoint_to_torch_state_dict
from .convert_state_dict import apply_weight_transforms_, move_safetensors_state_dict_params_, move_state_dict_params_
from .load import load_model_params_and_config_from_checkpoint
from .store import store_checkpoint_sharded

LOGGER = logging.getLogger(__name__)


def create_mlstm_simple_config_from_jax_config(
    model_config_jax: dict[str, Any],
    overrides: dict[str, Any] = None,
) -> mLSTMConfig:
    new_cfg = dict(
        embedding_dim=model_config_jax["embedding_dim"],
        num_heads=model_config_jax["mlstm_block"]["mlstm"]["num_heads"],
        num_blocks=model_config_jax["num_blocks"],
        vocab_size=model_config_jax["vocab_size"],
        use_bias=model_config_jax["bias"],
        norm_eps=model_config_jax["norm_eps"],
        qk_dim_factor=model_config_jax["mlstm_block"]["mlstm"]["qk_dim_factor"],
        v_dim_factor=model_config_jax["mlstm_block"]["mlstm"]["v_dim_factor"],
        ffn_proj_factor=model_config_jax["mlstm_block"]["feedforward"]["proj_factor"],
        gate_soft_cap=model_config_jax["mlstm_block"]["mlstm"]["mlstm_cell"]["gate_soft_cap"],
        output_logit_soft_cap=model_config_jax["logits_soft_cap"],
    )
    if overrides is not None:
        new_cfg.update(overrides)
    new_cfg = mLSTMConfig(**new_cfg)
    return new_cfg


def apply_mlstm_param_reshapes(state_dict: dict[str, Any]) -> dict[str, Any]:
    transpose_transform_to = [
        "lm_head.weight",
        "q.weight",
        "k.weight",
        "v.weight",
        "ogate_preact.weight",
        "fgate_preact.weight",
        "igate_preact.weight",
        "out_proj.weight",
        "ffn.proj_up.weight",
        "ffn.proj_down.weight",
        "ffn.proj_up_gate.weight",
    ]

    squeeze_transform_to = [
        "embedding.weight",
        "norm_ffn.weight",
        "norm_mlstm.weight",
    ]

    flatten_transform_to = [
        "multihead_norm.weight",
    ]

    transform_mapping = {
        "transpose": transpose_transform_to,
        "squeeze": squeeze_transform_to,
        "flatten": flatten_transform_to,
    }

    reshaped_state_dict = apply_weight_transforms_(state_dict, transform_mapping)
    return reshaped_state_dict


def move_mlstm_jax_state_dict_into_torch_state_dict(
    model_state_dict_torch: dict[str, Any],
    model_state_dict_jax_path: Path = None,
    model_state_dict_jax: dict[str, Any] = None,
) -> dict[str, Any]:
    """Move the mLSTM jax model state dict into the torch model.

    Either loads the jax model state dict from the model_state_dict_jax_path or uses the provided model_state_dict_jax.

    Args:
        model_torch (dict[str, Any]): The torch model.
        model_state_dict_jax_path (Path): The path to the jax model state dict. Defaults to None.
        model_state_dict_jax (dict[str, Any]): The jax model state dict. Defaults to None.

    Returns:
        mLSTM: The torch model with the jax model state dict loaded.

    """

    assert (
        model_state_dict_jax_path is not None or model_state_dict_jax is not None
    ), "Either model_state_dict_jax_path or model_state_dict_jax must be provided."

    match_dict = {
        "lm_head.out_dense.kernel": "lm_head.weight",
        "lm_head.out_norm.scale": "out_norm.weight",
        "embedding": "embedding.weight",
        ".ffn.proj_up.Dense_0.kernel": ".ffn.proj_up.weight",
        ".ffn.proj_down.Dense_0.kernel": ".ffn.proj_down.weight",
        ".ffn.proj_up_gate.Dense_0.kernel": ".ffn.proj_up_gate.weight",
        ".ffn_norm.sharded.scale": ".norm_ffn.weight",
        ".xlstm.dense_k.Dense_0.kernel": "mlstm_layer.k.weight",
        ".xlstm.dense_q.Dense_0.kernel": "mlstm_layer.q.weight",
        ".xlstm.dense_v.Dense_0.kernel": "mlstm_layer.v.weight",
        ".xlstm.dense_o.Dense_0.kernel": "mlstm_layer.ogate_preact.weight",
        ".xlstm.fgate.Dense_0.kernel": "mlstm_layer.fgate_preact.weight",
        ".xlstm.fgate.Dense_0.bias": "mlstm_layer.fgate_preact.bias",
        ".xlstm.igate.Dense_0.kernel": "mlstm_layer.igate_preact.weight",
        ".xlstm.igate.Dense_0.bias": "mlstm_layer.igate_preact.bias",
        ".xlstm.outnorm.sharded.scale": "mlstm_layer.multihead_norm.weight",
        ".xlstm.proj_down.Dense_0.kernel": "mlstm_layer.out_proj.weight",
        ".xlstm_norm.sharded.scale": ".norm_mlstm.weight",
    }
    if model_state_dict_jax_path is not None:
        model_state_dict_torch = move_safetensors_state_dict_params_(
            from_state_dict_path=model_state_dict_jax_path,
            to_state_dict=model_state_dict_torch,
            match_dict=match_dict,
        )
    elif model_state_dict_jax is not None:
        model_state_dict_torch = move_state_dict_params_(
            from_state_dict=model_state_dict_jax, to_state_dict=model_state_dict_torch, match_dict=match_dict
        )
    else:
        raise ValueError("Either model_state_dict_jax_path or model_state_dict_jax must be provided.")

    return model_state_dict_torch


def pipeline_convert_mlstm_checkpoint_jax_to_torch_simple(
    jax_orbax_model_checkpoint: dict[str, Any],
    jax_model_config: dict[str, Any],
    torch_model_config_overrides: dict[str, Any] = None,
) -> mLSTM:
    LOGGER.info("Jax model checkpoint loaded.")
    LOGGER.info("Converting jax model checkpoint to torch model checkpoint.")
    torch_state_dict_from_jax = convert_orbax_checkpoint_to_torch_state_dict(
        orbax_pytree=jax_orbax_model_checkpoint, split_blocks=True, blocks_layer_name="blocks"
    )

    LOGGER.info("Creating torch mLSTM config.")
    mlstm_config = create_mlstm_simple_config_from_jax_config(jax_model_config, overrides=torch_model_config_overrides)
    LOGGER.info("Creating torch mLSTM model.")
    mlstm_model = mLSTM(mlstm_config)
    LOGGER.info("Moving jax model checkpoint parameters into torch model.")
    model_state_dict = mlstm_model.state_dict()
    mlstm_model_state_dict = move_mlstm_jax_state_dict_into_torch_state_dict(
        model_state_dict_jax=torch_state_dict_from_jax,
        model_state_dict_torch=model_state_dict,
    )
    LOGGER.info("Applying reshapes to torch model checkpoint.")
    mlstm_model_state_dict_reshaped = apply_mlstm_param_reshapes(mlstm_model_state_dict)

    LOGGER.info("Loading reshaped torch model checkpoint into torch model.")
    mlstm_model.load_state_dict(mlstm_model_state_dict_reshaped)
    return mlstm_model


def store_mlstm_simple_to_checkpoint(
    mlstm_model: mLSTM,
    store_torch_model_checkpoint_path: Path,
    checkpoint_type: Literal["plain", "huggingface"] = "plain",
    max_shard_size: int = 0,
):
    """Stores a mLSTM simple model into a checkpoint directory, using either the
    `huggingface` or `plain` format.

    Args:
        mlstm_model (mLSTM): The mLSTM simple model.
        store_torch_model_checkpoint_path; Torch checkpoint path to store into.
        checkpoint_type: Type of model checkpoint, either 'plain' or 'huggingface'.
        max_shard_size: Largest size of a checkpoint model shard. Zero means no sharding.
    """
    store_torch_model_checkpoint_path.mkdir(parents=True, exist_ok=True)

    mlstm_model_state_dict = mlstm_model.state_dict()

    if checkpoint_type == "huggingface":
        mlstm_model_state_dict["backbone.embeddings.weight"] = mlstm_model_state_dict.pop("embedding.weight")

    store_checkpoint_sharded(
        mlstm_model_state_dict,
        store_torch_model_checkpoint_path / "model.safetensors",
        metadata={"format": "pt"},
        max_shard_size=max_shard_size,
    )

    mlstm_config_dict = OmegaConf.create(asdict(mlstm_model.config))

    if checkpoint_type == "plain":
        OmegaConf.save(mlstm_config_dict, store_torch_model_checkpoint_path / "config.yaml")
    elif checkpoint_type == "huggingface":
        with open(store_torch_model_checkpoint_path / "config.json", "w", encoding="utf8") as fp:
            cfg_dict = asdict(mlstm_model.config)
            cfg_dict["model_type"] = "xlstm"
            json.dump(cfg_dict, fp)
    else:
        raise ValueError(f"Bad checkpoint type {checkpoint_type}")


def convert_mlstm_checkpoint_jax_to_torch_simple(
    load_jax_model_checkpoint_path: Path,
    store_torch_model_checkpoint_path: Path,
    checkpoint_type: Literal["plain", "huggingface"] = "plain",
    max_shard_size: int = 0,
) -> None:
    """Convert a jax mLSTM checkpoint to a torch mLSTM checkpoint.

    Loads the jax mLSTM checkpoint, creates a torch mLSTM model, and moves the jax checkpoint parameters into the
    torch model.

    The checkpoint for the torch model is then saved to the store_torch_model_checkpoint_path.

    The torch checkpoint is a directory containing the model params as .safetensors file(s) and a config.yaml file.

    Args:
        load_jax_model_checkpoint_path: Orbax checkpoint path.
        store_torch_model_checkpoint_path; Torch checkpoint path to store into.
        checkpoint_type: Type of model checkpoint, either 'plain' or 'huggingface'.
        max_shard_size: Largest size of a checkpoint model shard. Zero means no sharding

    """
    LOGGER.info(f"Loading jax model checkpoint from {load_jax_model_checkpoint_path}.")
    jax_orbax_model_checkpoint, jax_model_config = load_model_params_and_config_from_checkpoint(
        load_jax_model_checkpoint_path
    )

    mlstm_model = pipeline_convert_mlstm_checkpoint_jax_to_torch_simple(jax_orbax_model_checkpoint, jax_model_config)

    store_mlstm_simple_to_checkpoint(
        mlstm_model,
        store_torch_model_checkpoint_path=store_torch_model_checkpoint_path,
        checkpoint_type=checkpoint_type,
        max_shard_size=max_shard_size,
    )
