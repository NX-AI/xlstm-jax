#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import argparse
import logging
from pathlib import Path

import jax

from xlstm_jax.dataset import GrainArrayRecordsDataConfig, LLMBatch, create_data_iterator
from xlstm_jax.distributed import set_XLA_flags
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.cell import mLSTMBackendNameAndKwargs
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMCellConfig, mLSTMLayerConfig
from xlstm_jax.models.xlstm_parallel.components.feedforward import FeedForwardConfig
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from xlstm_jax.trainer.callbacks import JaxProfilerConfig, LearningRateMonitorConfig, ModelCheckpointConfig
from xlstm_jax.trainer.llm.trainer import LLMTrainer, LLMTrainerConfig
from xlstm_jax.trainer.logger import FileLoggerConfig, LoggerConfig, TensorBoardLoggerConfig, WandBLoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig

set_XLA_flags()  # Must be executed before any JAX operation.

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

MODEL_CONFIGS = {
    "120M": {
        "model_config": lambda parallel, context_length: xLSTMLMModelConfig(
            vocab_size=50304,
            embedding_dim=768,
            num_blocks=12,
            context_length=context_length,
            tie_weights=False,
            add_embedding_dropout=False,
            add_post_blocks_norm=True,
            parallel=parallel,
            scan_blocks=False,
            dtype="bfloat16",
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    num_heads=4,
                    mlstm_cell=mLSTMCellConfig(
                        gate_dtype="float32", backend=mLSTMBackendNameAndKwargs(name="triton_kernels")
                    ),
                )
            ),
        ),
        "batch_size_per_device": 16,
        "gradient_accumulate_steps": 1,
        "fsdp_modules": (),
        "remat": (),
        "data_axis_size": -1,
        "fsdp_axis_size": 1,
        "lr": 1e-3,
    },
    "165M": {
        "model_config": lambda parallel, context_length: xLSTMLMModelConfig(
            vocab_size=50304,
            embedding_dim=768,
            num_blocks=24,
            context_length=context_length,
            tie_weights=False,
            add_embedding_dropout=False,
            add_post_blocks_norm=True,
            parallel=parallel,
            scan_blocks=True,
            dtype="bfloat16",
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    num_heads=4,
                    mlstm_cell=mLSTMCellConfig(
                        gate_dtype="float32",
                        backend=mLSTMBackendNameAndKwargs(name="triton_kernels"),
                    ),
                )
            ),
        ),
        "batch_size_per_device": 16,
        "gradient_accumulate_steps": 1,
        "fsdp_modules": (),
        "remat": ("mLSTMBlock",),
        "data_axis_size": -1,
        "fsdp_axis_size": 1,
        "lr": 1e-3,
    },
    "165M_v1": {
        "model_config": lambda parallel, context_length: xLSTMLMModelConfig(
            vocab_size=50304,
            embedding_dim=768,
            # Fewer blocks due to FFN being in mLSTM block.
            num_blocks=12,
            context_length=context_length,
            tie_weights=False,
            add_embedding_dropout=False,
            add_post_blocks_norm=True,
            parallel=parallel,
            scan_blocks=True,
            norm_eps=1e-6,
            norm_type="rmsnorm",
            dtype="bfloat16",
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    layer_type="mlstm_v1",
                    num_heads=4,
                    mlstm_cell=mLSTMCellConfig(
                        gate_dtype="float32",
                        backend=mLSTMBackendNameAndKwargs(name="triton_kernels"),
                        # Lowering the input bias init appears to stabilize training.
                        igate_bias_init_range=-10.0,
                        add_qk_norm=False,
                        norm_type="rmsnorm",
                        norm_eps=1e-6,
                        reset_at_document_boundaries=False,
                    ),
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=4.0,
                    act_fn="gelu",
                    ff_type="ffn",
                    dtype="bfloat16",
                ),
                add_post_norm=False,
            ),
        ),
        "batch_size_per_device": 16,
        "gradient_accumulate_steps": 1,
        "fsdp_modules": (),
        "remat": ("mLSTMBlock",),
        "data_axis_size": -1,
        "fsdp_axis_size": 1,
        "lr": 1e-3,
    },
    "1.3B": {
        "model_config": lambda parallel, context_length: xLSTMLMModelConfig(
            vocab_size=50304,
            embedding_dim=2048,
            num_blocks=48,
            context_length=context_length,
            tie_weights=False,
            add_embedding_dropout=False,
            add_post_blocks_norm=True,
            parallel=parallel,
            scan_blocks=True,
            dtype="bfloat16",
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    num_heads=4,
                    mlstm_cell=mLSTMCellConfig(
                        gate_dtype="float32",  # backend=mLSTMBackendNameAndKwargs(name="triton_kernels")
                    ),
                ),
            ),
        ),
        "batch_size_per_device": 8,
        "gradient_accumulate_steps": 1,
        "fsdp_modules": (),
        "remat": ("mLSTMBlock",),
        "data_axis_size": -1,
        "fsdp_axis_size": 1,
        "lr": 7e-4,
    },
    "1.3B_v1": {
        "model_config": lambda parallel, context_length: xLSTMLMModelConfig(
            vocab_size=50304,
            embedding_dim=2048,
            num_blocks=24,
            context_length=context_length,
            tie_weights=False,
            add_embedding_dropout=False,
            add_post_blocks_norm=True,
            parallel=parallel,
            scan_blocks=True,
            norm_eps=1e-6,
            norm_type="rmsnorm",
            dtype="bfloat16",
            lm_head_dtype="bfloat16",
            logits_soft_cap=30.0,
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    layer_type="mlstm_v1",
                    num_heads=4,
                    qk_dim_factor=0.5,
                    mlstm_cell=mLSTMCellConfig(
                        gate_dtype="float32",
                        backend=mLSTMBackendNameAndKwargs(name="triton_kernels"),
                        # Lowering the input bias init appears to stabilize training.
                        igate_bias_init_range=-10.0,
                        add_qk_norm=False,
                        norm_type="rmsnorm",
                        norm_eps=1e-6,
                        reset_at_document_boundaries=False,
                        gate_soft_cap=15.0,
                    ),
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=8.0 / 3.0,
                    act_fn="swish",
                    ff_type="ffn_gated",
                    dtype="bfloat16",
                ),
                add_post_norm=False,
            ),
        ),
        "batch_size_per_device": 16,
        "gradient_accumulate_steps": 1,
        "fsdp_modules": (),
        "remat": ("xLSTMResBlock", "FFNResBlock"),
        "data_axis_size": -1,
        "fsdp_axis_size": 1,
        "lr": 7e-4,
    },
    "7B": {
        "model_config": lambda parallel, context_length: xLSTMLMModelConfig(
            vocab_size=50304,
            embedding_dim=4096,
            num_blocks=64,
            context_length=context_length,
            tie_weights=False,
            add_embedding_dropout=False,
            add_post_blocks_norm=True,
            parallel=parallel,
            scan_blocks=True,
            dtype="bfloat16",
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    num_heads=8,
                    mlstm_cell=mLSTMCellConfig(
                        gate_dtype="float32",  # backend=mLSTMBackendNameAndKwargs(name="triton_kernels")
                    ),
                )
            ),
        ),
        "batch_size_per_device": 8,
        "gradient_accumulate_steps": 1,
        "fsdp_modules": ("Embed", "LMHead", "mLSTMBlock"),
        "remat": ("mLSTMBlock",),
        "data_axis_size": -1,
        "fsdp_axis_size": 8,
        "lr": 5e-4,
    },
    "7B_v1": {
        "model_config": lambda parallel, context_length: xLSTMLMModelConfig(
            vocab_size=50304,
            embedding_dim=4096,
            num_blocks=30,
            context_length=context_length,
            tie_weights=False,
            add_embedding_dropout=False,
            add_post_blocks_norm=True,
            parallel=parallel,
            scan_blocks=True,
            norm_eps=1e-6,
            norm_type="rmsnorm",
            lm_head_dtype="bfloat16",
            logits_soft_cap=30.0,
            dtype="bfloat16",
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    layer_type="mlstm_v1",
                    num_heads=8,
                    qk_dim_factor=0.5,
                    mlstm_cell=mLSTMCellConfig(
                        gate_dtype="float32",
                        backend=mLSTMBackendNameAndKwargs(name="triton_kernels"),
                        # Lowering the input bias init appears to stabilize training.
                        igate_bias_init_range=-10.0,
                        add_qk_norm=False,
                        norm_type="rmsnorm",
                        norm_eps=1e-6,
                        reset_at_document_boundaries=False,
                        gate_soft_cap=15.0,
                    ),
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=8.0 / 3.0,
                    act_fn="swish",
                    ff_type="ffn_gated",
                    dtype="bfloat16",
                ),
                add_post_norm=False,
            ),
        ),
        "batch_size_per_device": 16,
        "gradient_accumulate_steps": 1,
        "fsdp_modules": ("Embed", "LMHead", "mLSTMBlock"),
        "remat": ("xLSTMResBlock", "FFNResBlock"),
        "data_axis_size": -1,
        "fsdp_axis_size": 8,
        "lr": 5e-4,
    },
}


def log_info(msg: str):
    if jax.process_index() == 0:
        LOGGER.info(msg)


def main_train(args: argparse.Namespace):
    # Config
    global_model_config = MODEL_CONFIGS[args.model]
    # Create mesh. Needs to be done before any JAX operation due to distribute initialize.
    parallel = ParallelConfig(
        data_axis_name="dp",
        fsdp_axis_name="fsdp",
        model_axis_name="tp",
        pipeline_axis_name="pp",
        fsdp_modules=global_model_config["fsdp_modules"],
        fsdp_gather_dtype="bfloat16",
        fsdp_min_weight_size=2**18,
        remat=global_model_config["remat"],
        fsdp_axis_size=global_model_config["fsdp_axis_size"],
        model_axis_size=global_model_config.get("model_axis_size", 1),
        data_axis_size=global_model_config["data_axis_size"],
        tp_async_dense=False,
    )
    mesh = initialize_mesh(parallel_config=parallel)
    log_info("Mesh initialized.")
    assert len(jax.devices(backend="gpu")) > 0, "No devices found. This script should be run on GPU."
    log_info(f"Devices: {jax.devices()}")

    # General hyperparameters.
    batch_size_per_device = global_model_config["batch_size_per_device"]
    batch_size = batch_size_per_device * len(jax.devices())
    context_length = 2048
    num_train_steps = 95_000
    lr = global_model_config.get("lr", 1e-3)
    log_path = Path(args.log_dir)

    # Create data iterator.
    log_info("Creating data iterator.")
    data_name = "627B" if args.use_full_dataset else "6B"
    dataset_name = "cerebras/SlimPajama-627B" if args.use_full_dataset else "DKYoon/SlimPajama-6B"
    train_config, eval_config = GrainArrayRecordsDataConfig.create_train_eval_configs(
        train_kwargs={"grain_packing": True, "grain_packing_bin_count": batch_size_per_device * 8},
        eval_kwargs={"grain_packing": False},  # Packing is deactivated for eval to make it reproducible across epochs
        global_batch_size=batch_size,
        data_path=Path("/nfs-gpu/xlstm/data/array_records/") / dataset_name.replace("/", "_"),
        max_target_length=context_length,
        data_column="text",
        tokenizer_path=args.tokenizer,
        data_shuffle_seed=123,
        worker_buffer_size=8,
    )
    train_data_iterator = create_data_iterator(config=train_config, mesh=mesh)
    eval_data_iterator = create_data_iterator(config=eval_config, mesh=mesh)

    # Define model config.
    model_config = global_model_config["model_config"](parallel=parallel, context_length=context_length)
    backend_name = model_config.mlstm_block.mlstm.mlstm_cell.backend.name
    wb_name = f"xlstm_slimpajama{data_name}_{args.model}_gbs{int(batch_size)}_ctx{context_length}_lr{lr}_{backend_name}"

    # Create trainer with sub-configs.
    log_info("Creating trainer.")
    trainer = LLMTrainer(
        LLMTrainerConfig(
            callbacks=[
                ModelCheckpointConfig(
                    every_n_epochs=1,
                    monitor="perplexity",
                    max_to_keep=1,
                    save_optimizer_state=True,
                    enable_async_checkpointing=True,
                ),
                LearningRateMonitorConfig(
                    every_n_steps=50,
                    every_n_epochs=-1,
                    main_process_only=True,
                ),
                JaxProfilerConfig(
                    profile_every_n_minutes=60,
                ),
            ],
            logger=LoggerConfig(
                log_path=log_path,
                log_every_n_steps=50,
                log_tools=[
                    FileLoggerConfig(log_dir="file_logs", config_format="json"),
                    TensorBoardLoggerConfig(log_dir="tensorboard", tb_flush_secs=10),
                    WandBLoggerConfig(
                        wb_project="xlstm_jax",
                        wb_entity="xlstm",
                        wb_name=wb_name,
                        wb_tags=[f"slimpajama{data_name}", args.model, "reproduction", backend_name],
                    ),
                ],
            ),
            check_val_every_n_steps=5_000,
            enable_progress_bar=False,
            check_for_nan=True,
            log_grad_norm=True,
            log_grad_norm_per_param=True,
            log_param_norm=True,
            log_param_norm_per_param=True,
            default_train_log_modes=["mean", "std", "max"],
            log_logit_stats=True,
            log_intermediates=True,
        ),
        ModelConfig(
            model_class=xLSTMLMModel,
            parallel=parallel,
            model_config=model_config,
        ),
        OptimizerConfig(
            name="adamw",
            scheduler=SchedulerConfig(
                name="exponential_decay",
                lr=lr,
                decay_steps=num_train_steps,
                end_lr_factor=0.1,
                warmup_steps=750,
                cooldown_steps=2_000,
            ),
            grad_clip_norm=0.5,
            weight_decay=0.1,
            weight_decay_include=[r".*kernel"],
            beta2=0.95,
            eps=1e-8,
        ),
        batch=LLMBatch.get_dtype_struct(batch_size, context_length),
        mesh=mesh,
    )

    if len(args.load_checkpoint_from) > 0:
        log_info(f"Loading checkpoint from {args.load_checkpoint_from}.")
        trainer.load_pretrained_model(
            Path(args.load_checkpoint_from),
            step_idx=-1,
            load_best=False,
            train_loader=train_data_iterator,
            val_loader=eval_data_iterator,
        )

    log_info("Training model.")
    final_metrics = trainer.train_model(
        train_loader=train_data_iterator,
        val_loader=eval_data_iterator,
        num_train_steps=num_train_steps,
    )
    log_info(f"Final metrics: {final_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train xLSTM model on SlimPajama6B dataset.")
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--model", type=str, choices=MODEL_CONFIGS.keys(), default="165M")
    parser.add_argument("--use_full_dataset", action="store_true", help="Use the 627B dataset instead of 6B version.")
    parser.add_argument("--load_checkpoint_from", type=str, default="")
    parser.add_argument("--tokenizer", type=str, choices=["gpt2", "EleutherAI/gpt-neox-20b"], default="gpt2")
    args = parser.parse_args()
    main_train(args)
