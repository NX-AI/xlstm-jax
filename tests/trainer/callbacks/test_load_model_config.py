#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from xlstm_jax.dataset import LLMBatch
from xlstm_jax.models.configs import ModelConfig, ParallelConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.cell import mLSTMCellConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from xlstm_jax.trainer.callbacks.checkpointing import ModelCheckpointConfig
from xlstm_jax.trainer.llm.trainer import LLMTrainer, LLMTrainerConfig
from xlstm_jax.trainer.logger import LoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig

metadata_example = (
    """
{
    "model": {
        "model_class": "xlstm_jax.models.xlstm_parallel.xlstm_lm_model.xLSTMLMModel",
        "model_config": {
            "_block_map": "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
            "add_embedding_dropout": false,
            "add_post_blocks_norm": true,
            "bias": false,
            "context_length": 2048,
            "dropout": 0.0,
            "dtype": "<class 'jax.numpy.bfloat16'>",
            "embedding_dim": 2048,
            "init_distribution_embed": "normal",
            "init_distribution_out": "normal",
            "lm_head_dtype": "<class 'jax.numpy.bfloat16'>",
            "logits_soft_cap": 30.0,
            "mlstm_block": {
                "_block_idx": "None",
                "_num_blocks": 24,
                "add_post_norm": false,
                "feedforward": {
                    "_num_blocks": 24,
                    "_proj_up_dim": 5504,
                    "act_fn": "swish",
                    "bias": false,
                    "dropout": 0.0,
                    "dtype": "<class 'jax.numpy.bfloat16'>",
                    "embedding_dim": 2048,
                    "ff_type": "ffn_gated",
                    "init_distribution": "normal",
                    "output_init_fn": "wang",
                    "parallel": "ParallelConfig(data_axis_size=-1, fsdp_axis_size=1, """
    """pipeline_axis_size=1, model_axis_size=1, data_axis_name='dp', fsdp_axis_name='fsdp', """
    """pipeline_axis_name='pp', model_axis_name='tp', remat=('xLSTMResBlock', 'FFNResBlock'), """
    """fsdp_modules=(), fsdp_min_weight_size=262144, fsdp_gather_dtype='bfloat16', """
    """fsdp_grad_scatter_dtype=None, tp_async_dense=False)",
                    "proj_factor": 2.6666666666666665,
                    "round_proj_up_dim_up": true,
                    "round_proj_up_to_multiple_of": 64
                },
                "mlstm": {
                    "_block_idx": "None",
                    "_inner_embedding_dim": 4096,
                    "_num_blocks": 24,
                    "_proj_up_dim": 4096,
                    "bias": false,
                    "context_length": 2048,
                    "conv1d_kernel_size": 4,
                    "debug_cell": false,
                    "dropout": 0.0,
                    "dtype": "<class 'jax.numpy.bfloat16'>",
                    "embedding_dim": 2048,
                    "gate_input": "qkv",
                    "init_distribution": "normal",
                    "layer_type": "mlstm_v1",
                    "mlstm_cell": {
                        "add_qk_norm": false,
                        "backend": {
                            "_registry": "{}",
                            "kwargs": "mLSTMBackendTritonConfig(autocast_dtype=None, """
    """chunk_size=64, reduce_slicing=True)",
                            "name": "triton_kernels"
                        },
                        "context_length": 2048,
                        "dtype": "<class 'jax.numpy.bfloat16'>",
                        "embedding_dim": 4096,
                        "fgate_bias_init_range": [
                            3.0,
                            6.0
                        ],
                        "gate_dtype": "<class 'jax.numpy.float32'>",
                        "gate_linear_headwise": false,
                        "gate_soft_cap": 15.0,
                        "igate_bias_init_range": -10.0,
                        "norm_eps": 1e-06,
                        "norm_type": "rmsnorm",
                        "num_heads": 4,
                        "parallel": "None",
                        "reset_at_document_boundaries": false,
                        "reset_fgate_value": -25.0
                    },
                    "norm_type": "rmsnorm",
                    "num_heads": 4,
                    "output_init_fn": "wang",
                    "parallel": "ParallelConfig(data_axis_size=-1, fsdp_axis_size=1, pipeline_axis_size=1, """
    """model_axis_size=1, data_axis_name='dp', fsdp_axis_name='fsdp', pipeline_axis_name='pp', model_axis_name='tp', """
    """remat=('xLSTMResBlock', 'FFNResBlock'), fsdp_modules=(), fsdp_min_weight_size=262144, """
    """fsdp_gather_dtype='bfloat16', fsdp_grad_scatter_dtype=None, tp_async_dense=False)",
                    "proj_factor": 2.0,
                    "qk_dim_factor": 0.5,
                    "qkv_proj_blocksize": 4,
                    "round_proj_up_dim_up": true,
                    "round_proj_up_to_multiple_of": 64,
                    "v_dim_factor": 1.0,
                    "vmap_qk": false
                },
                "parallel": "ParallelConfig(data_axis_size=-1, fsdp_axis_size=1, pipeline_axis_size=1, """
    """model_axis_size=1, data_axis_name='dp', fsdp_axis_name='fsdp', pipeline_axis_name='pp', """
    """model_axis_name='tp', remat=('xLSTMResBlock', 'FFNResBlock'), fsdp_modules=(), """
    """fsdp_min_weight_size=262144, fsdp_gather_dtype='bfloat16', fsdp_grad_scatter_dtype=None, """
    """tp_async_dense=False)"
            },
            "norm_eps": 1e-06,
            "norm_type": "rmsnorm",
            "num_blocks": 24,
            "parallel": "ParallelConfig(data_axis_size=-1, fsdp_axis_size=1, pipeline_axis_size=1, """
    """model_axis_size=1, data_axis_name='dp', fsdp_axis_name='fsdp', pipeline_axis_name='pp', """
    """model_axis_name='tp', remat=('xLSTMResBlock', 'FFNResBlock'), fsdp_modules=(), """
    """fsdp_min_weight_size=262144, fsdp_gather_dtype='bfloat16', fsdp_grad_scatter_dtype=None, """
    """tp_async_dense=False)",
            "scan_blocks": true,
            "slstm_at": [],
            "slstm_block": "None",
            "tie_weights": false,
            "vocab_size": 50304,
            "weight_decay_on_embedding": false
        },
        "parallel": "ParallelConfig(data_axis_size=-1, fsdp_axis_size=1, pipeline_axis_size=1, model_axis_size=1, """
    """data_axis_name='dp', fsdp_axis_name='fsdp', pipeline_axis_name='pp', model_axis_name='tp', """
    """remat=('xLSTMResBlock', 'FFNResBlock'), fsdp_modules=(), fsdp_min_weight_size=262144, """
    """fsdp_gather_dtype='bfloat16', fsdp_grad_scatter_dtype=None, tp_async_dense=False)"
    },
    "optimizer": {
        "alpha": 8.0,
        "beta1": 0.9,
        "beta2": 0.95,
        "beta3": 0.9999,
        "eps": 1e-08,
        "grad_clip_norm": 0.5,
        "grad_clip_value": "None",
        "name": "adamw",
        "nesterov": false,
        "scheduler": {
            "cooldown_lr": 0.0,
            "cooldown_steps": 2000,
            "decay_steps": 95000,
            "end_lr": "None",
            "end_lr_factor": 0.1,
            "lr": 0.0003,
            "name": "cosine_decay",
            "warmup_steps": 750
        },
        "use_sharded_clip_norm": true,
        "weight_decay": 0.1,
        "weight_decay_exclude": "None",
        "weight_decay_include": [
            ".*kernel"
        ]
    },
    "trainer": {
        "callbacks": [
            {
                "enable_async_checkpointing": true,
                "every_n_epochs": 1,
                "every_n_steps": -1,
                "log_path": "None",
                "main_process_only": false,
                "max_to_keep": 1,
                "mode": "min",
                "monitor": "perplexity",
                "save_dataloader_state": true,
                "save_optimizer_state": true
            },
            {
                "every_n_epochs": -1,
                "every_n_steps": 50,
                "log_lr_key": "optimizer/lr",
                "main_process_only": true
            },
            {
                "every_n_epochs": -1,
                "every_n_steps": -1,
                "main_process_only": true,
                "profile_every_n_minutes": 60,
                "profile_first_step": 10,
                "profile_log_dir": "tensorboard",
                "profile_n_steps": 5
            }
        ],
        "check_for_nan": true,
        "check_val_every_n_epoch": 1,
        "check_val_every_n_steps": 5000,
        "debug": false,
        "default_train_log_modes": [
            "mean",
            "std",
            "max"
        ],
        "donate_train_state": true,
        "enable_progress_bar": false,
        "gradient_accumulate_scan": false,
        "gradient_accumulate_steps": 1,
        "intermediates_log_modes": [
            "mean"
        ],
        "log_grad_norm": true,
        "log_grad_norm_per_param": true,
        "log_intermediates": true,
        "log_logit_stats": true,
        "log_param_norm": true,
        "log_param_norm_per_param": true,
        "logger": {
            "log_every_n_steps": 50,
            "log_path": "/nfs-gpu/xlstm/logs/outputs_plippe/outputs/xlstm-jax/slimpajama6b/version_256",
            "log_tools": [
                {
                    "config_format": "json",
                    "log_dir": "file_logs",
                    "log_epoch_key": "log_epoch",
                    "log_step_key": "log_step"
                },
                {
                    "log_dir": "tensorboard",
                    "tb_flush_secs": 10,
                    "tb_max_queue": 10,
                    "tb_new_style": false
                },
                {
                    "log_dir": "wandb",
                    "wb_entity": "xlstm",
                    "wb_host": "https://api.wandb.ai",
                    "wb_key": "None",
                    "wb_name": "slimpajama600B_1.3B_v1_gbs256_ctx2048_lr0.0003_triton_kernels",
                    "wb_notes": "None",
                    "wb_project": "xlstm_jax",
                    "wb_settings": "{'start_method': 'fork'}",
                    "wb_tags": [
                        "slimpajama600B",
                        "1.3B_v1",
                        "reproduction",
                        "triton_kernels"
                    ]
                }
            ]
        },
        "seed": 0,
        "seed_eval": 0
    }
}
"""
)


# TODO: Clean this once all models are trained with a stable config
def test_load_model_config_legacy():
    model_cfg = ModelConfig.from_metadata(metadata_content=metadata_example)

    mod = model_cfg.model_class(model_cfg.model_config)
    assert isinstance(mod, model_cfg.model_class)


# TODO: Integrate a test for OmegaConf based config stringification and parsing
# def test_load_model_config_legacy_omegaconf():
#     from omegaconf import OmegaConf
#     model_cfg = OmegaConf.merge(OmegaConf.structured(ModelConfig), OmegaConf.create(json.load(metadata_example)))
#     model_cfg_baseline = ModelConfig.from_metadata(metadata_content=metadata_example)
#     assert model_cfg == model_cfg_baseline


def test_checkpoint_reload(tmp_path: Path):
    """
    Tests checkpointing with ModelCheckpoint callback with per-step eval.

    The test trains a simple model with MSE loss under different mesh configs. We then check whether the checkpoints
    have been created as expected, load an older model, and reproduce the training and validation metrics.
    """
    tp_size, fsdp_size = 1, 1
    batch_size = 8
    context_length = 16
    log_path = tmp_path
    parallel = ParallelConfig(
        data_axis_name="dp",
        fsdp_axis_name="fsdp",
        model_axis_name="tp",
        pipeline_axis_name="pp",
        fsdp_modules=["Embed", "LMHead", "mLSTMBlock"],
        fsdp_min_weight_size=2**8,
        fsdp_axis_size=fsdp_size,
        model_axis_size=tp_size,
        data_axis_size=-1,
    )
    # Define model config as before.
    xlstm_config = xLSTMLMModelConfig(
        vocab_size=20,
        embedding_dim=128,
        num_blocks=2,
        context_length=context_length,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        parallel=parallel,
        dtype="float32",
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                proj_factor=2.0,
                conv1d_kernel_size=4,
                num_heads=4,
                dropout=0.2,
                embedding_dim=128,
                context_length=context_length,
                mlstm_cell=mLSTMCellConfig(
                    gate_linear_headwise=True,
                    gate_soft_cap=30.0,
                    reset_at_document_boundaries=True,
                ),
            )
        ),
    )

    trainer = LLMTrainer(
        LLMTrainerConfig(
            callbacks=(
                ModelCheckpointConfig(
                    monitor="perplexity",
                    max_to_keep=2,
                    save_optimizer_state=True,
                    enable_async_checkpointing=True,
                ),
            ),
            logger=LoggerConfig(log_path=tmp_path),
        ),
        ModelConfig(
            model_class=xLSTMLMModel,
            parallel=parallel,
            model_config=xlstm_config,
        ),
        OptimizerConfig(
            name="adam",
            scheduler=SchedulerConfig(
                name="constant",
                lr=1e-4,
            ),
        ),
        batch=LLMBatch.get_dtype_struct(batch_size, xlstm_config.context_length),
    )

    def data_gen_fn(idx: int) -> LLMBatch:
        inputs = jax.random.randint(
            jax.random.PRNGKey(idx),
            (batch_size, xlstm_config.context_length),
            minval=0,
            maxval=xlstm_config.vocab_size,
        )
        targets = jnp.mod(inputs + 1, xlstm_config.vocab_size)
        return LLMBatch.from_inputs(inputs=inputs, targets=targets)

    train_loader = [data_gen_fn(idx) for idx in range(100)]
    val_loader = train_loader[:10]
    _ = trainer.train_model(
        train_loader,
        val_loader,
        num_train_steps=100,
    )
    assert log_path.exists()
    checkpoint_path = log_path / "checkpoints"
    assert checkpoint_path.exists()
    assert (checkpoint_path / "checkpoint_100").exists()
    metadata_file_path = checkpoint_path / "checkpoint_100" / "metadata" / "metadata"

    assert metadata_file_path.exists()

    with open(str(metadata_file_path), encoding="utf8") as fp:
        metadata = fp.read()

    model_cfg = ModelConfig.from_metadata(metadata)
    mod = model_cfg.model_class(model_cfg.model_config)
    assert isinstance(mod, model_cfg.model_class)

    # check if loaded config matches old config
    assert model_cfg.model_class == trainer.model_config.model_class
    # check equivalence via json string
    cfg_str_new = json.dumps(model_cfg.model_config.to_dict(), sort_keys=True)
    cfg_str_old = json.dumps(trainer.model_config.model_config.to_dict(), sort_keys=True)
    assert cfg_str_new == cfg_str_old, f"Non matching cfg strings: {cfg_str_new} != {cfg_str_old}"
