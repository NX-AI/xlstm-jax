import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

# For now, we load MODEL_CONFIGS from this training script
# In the future the config should be loaded from the checkpoint
# or a configuration library.
from scripts.run_train_slimpajama import MODEL_CONFIGS

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from xlstm_jax.configs import ConfigDict
from xlstm_jax.dataset import LLMBatch
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModel
from xlstm_jax.trainer.eval.lmeval_extended_evaluation import LMEvalEvaluationConfig
from xlstm_jax.trainer.llm.trainer import LLMTrainer, LLMTrainerConfig
from xlstm_jax.trainer.logger import FileLoggerConfig, LoggerConfig, TensorBoardLoggerConfig, WandBLoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig

LOGGER = logging.getLogger(__name__)


def log_info(msg: str):
    if jax.process_index() == 0:
        LOGGER.info(msg)


@dataclass(kw_only=True, frozen=False)
class CheckpointConfig(ConfigDict):
    checkpoint_dir: str = ""
    checkpoint_step_idx: int = -1
    checkpoint_load_best: bool = True


def main_lmeval(args: argparse.Namespace):
    parallel = ParallelConfig(
        data_axis_name="dp",
        fsdp_axis_name="fsdp",
        model_axis_name="tp",
        pipeline_axis_name="pp",
        fsdp_modules=[],
        fsdp_gather_dtype="bfloat16",
        fsdp_min_weight_size=2**18,
        remat=[],
        fsdp_axis_size=args.fsdp_size,
        model_axis_size=args.model_tp_size,
        data_axis_size=1,
        tp_async_dense=False,
    )
    mesh = initialize_mesh(parallel_config=parallel)

    def global_sync():
        with jax.named_scope("global_sync"):
            _ = jax.lax.psum(
                jnp.array(1.0),
                axis_name=mesh.axis_names,
            )

    global_sync_fn = shard_map(global_sync, mesh, in_specs=(P(mesh.axis_names)), out_specs=P(), check_rep=False)

    log_info("Mesh initialized.")
    assert len(jax.devices(backend="gpu")) > 0, "No devices found. This script should be run on GPU."
    log_info(f"Devices: {jax.devices()}")

    checkpoint_folder = args.load_checkpoint_from
    checkpoint_idx = args.checkpoint_step_idx
    if args.load_checkpoint_from_subdir:
        checkpoint_idx_path = Path(args.load_checkpoint_from_subdir)
        checkpoint_folder = checkpoint_idx_path.parent.parent
        checkpoint_idx = int(checkpoint_idx_path.name.split("_")[-1])
    elif checkpoint_folder:
        checkpoint_folder = Path(checkpoint_folder)

    if checkpoint_folder:
        path = Path(checkpoint_folder) / "checkpoints"
        metadata_path = (
            sorted(list(filter(lambda x: "checkpoint_" in str(x), path.iterdir())))[-1] / "metadata" / "metadata"
        )
        with open(metadata_path, encoding="utf8") as fp:
            metadata = fp.read()
        model_config_base = MODEL_CONFIGS[args.model]
        del model_config_base["model_config"]
        global_model_config = {"model_config": ModelConfig.from_metadata(metadata).model_config, **model_config_base}
        LOGGER.info("Loading model config from metadata file.")
        parallel.fsdp_modules = global_model_config["model_config"].parallel.fsdp_modules
        parallel.remat = global_model_config["model_config"].parallel.remat
    else:
        # Config
        global_model_config = MODEL_CONFIGS[args.model]
        # Create mesh. Needs to be done before any JAX operation due to distribute initialize.
        parallel.fsdp_modules = global_model_config["fsdp_modules"]
        parallel.remat = global_model_config["remat"]

    log_path = Path(args.log_dir) / "version_0"
    version_idx = 0
    while log_path.exists():
        log_path = Path(args.log_dir) / f"version_{version_idx}"
        version_idx += 1

    # General hyperparameters.
    batch_size = (
        global_model_config["batch_size_per_device"]
        if args.batch_size_per_device is None
        else args.batch_size_per_device
    ) * len(jax.devices())
    context_length = args.context_length

    # Define model config - 120M parameters.
    if callable(global_model_config["model_config"]):
        xlstm_config = global_model_config["model_config"](parallel=parallel, context_length=context_length)
    else:
        xlstm_config = global_model_config["model_config"]
        xlstm_config.parallel = parallel
        xlstm_config.context_length = context_length
        xlstm_config.__post_init__()
    if args.generic_wandb_name or not checkpoint_folder:
        wb_name = "evaluation"
    else:
        # use the hydra sweep name as wandb name, remove trailing and double / after split by filter
        wb_name = "eval_" + checkpoint_folder.parent.name + f"_{checkpoint_idx}" if checkpoint_idx > 0 else ""

    # Create trainer with sub-configs.
    log_info("Creating trainer.")
    trainer = LLMTrainer(
        LLMTrainerConfig(
            callbacks=(
                LMEvalEvaluationConfig(
                    tokenizer_path=args.tokenizer,
                    evaluation_tasks=args.tasks,
                    limit_requests=args.limit_requests,
                    context_length=args.context_length,
                    num_fewshot=args.num_fewshot,
                    use_infinite_eval=(not args.no_infinite_eval),
                    infinite_eval_chunksize=128,
                ),
            ),
            logger=LoggerConfig(
                log_path=log_path,
                log_every_n_steps=20,
                log_tools=[
                    FileLoggerConfig(log_dir="file_logs", config_format="json"),
                ]
                + []
                if args.no_logging
                else [
                    TensorBoardLoggerConfig(log_dir="tensorboard", tb_flush_secs=10),
                    WandBLoggerConfig(
                        wb_project="xlstm_jax_eval",
                        wb_entity="xlstm",
                        wb_name=wb_name,
                        wb_tags=[args.model, "evaluation"],
                        wb_resume_id=args.append_to_wandb_id,
                    ),
                ],
            ),
            check_val_every_n_epoch=1,
            enable_progress_bar=True,
            check_for_nan=True,
            log_grad_norm=True,
            log_grad_norm_per_param=False,
            log_param_norm=True,
            log_param_norm_per_param=False,
            default_train_log_modes=("mean", "std", "max"),
            log_logit_stats=True,
            log_intermediates=True,
        ),
        ModelConfig(
            model_class=xLSTMLMModel,
            parallel=parallel,
            model_config=xlstm_config,
        ),
        # this is needed for the trainer, but not actually used
        # it it just the optimizer with zero additional memory overhead
        OptimizerConfig(
            name="sgd",
            scheduler=SchedulerConfig(
                lr=1e-3,
            ),
        ),
        batch=LLMBatch.get_dtype_struct(batch_size, context_length),
        mesh=mesh,
    )

    eval_callback = trainer.callbacks[0]
    eval_callback.replace_trainer_state_by_eval_only()

    if checkpoint_folder:
        log_info(f"Loading checkpoint from {checkpoint_folder}.")
        trainer.load_pretrained_model(
            Path(checkpoint_folder),
            step_idx=checkpoint_idx,
            load_best=args.checkpoint_load_best,
            load_optimizer=False,
            train_loader=None,
            val_loader=None,
        )
        # update log config with checkpoint info
        trainer.logger.log_config(
            {
                "trainer": trainer.trainer_config,
                "model": trainer.model_config,
                "optimizer": trainer.optimizer_config,
                "checkpoint_folder": CheckpointConfig(
                    checkpoint_dir=checkpoint_folder,
                    checkpoint_load_best=args.checkpoint_load_best,
                    checkpoint_step_idx=checkpoint_idx,
                ),
            }
        )

    trainer.logger.on_training_start()
    metrics = eval_callback.run_evaluate()

    # synchronize all GPUs
    with jax.named_scope("global_sync"):
        global_sync_fn()

    for task in sorted(metrics):
        trainer.logger.log_host_metrics(
            metrics[task],
            step=trainer.global_step,
            mode="leh_" + task + (f"_nfs{args.num_fewshot}" if args.num_fewshot else ""),
        )

    log_info(f"Final metrics: {metrics}")
    trainer.logger.finalize(status="success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate xLSTM model on LMEval harness.")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[
            "lambada_openai",
            "lambada_standard",
            "hellaswag",
            "piqa",
            "arc_easy",
            "arc_challenge",
            "winogrande",
            "anli",
            "arithmetic",
            "asdiv",
            "wsc",
            "crows_pairs_english",
            "crows_pairs_french",
            "mmlu",
            "hendrycks_ethics",
            "lambada_cloze",
            "logiqa",
            "mathqa",
            "mc_taco",
            "medmcqa",
            "medqa_4options",
            "openbookqa",
            "prost",
            "race",
            "sciq",
            "social_iqa",
            "swag",
            "freebase",
            "wikitext",
            "wmdp",
            "wsc273",
        ],
    )
    parser.add_argument("--model", type=str, choices=MODEL_CONFIGS.keys(), default="120M")
    parser.add_argument("--log_dir", type=str, default="/nfs-gpu/xlstm/logs/outputs_poeppel/eval_jax")
    parser.add_argument("--load_checkpoint_from", type=str, default="")
    parser.add_argument("--load_checkpoint_from_subdir", type=str, default=None)
    parser.add_argument("--checkpoint_step_idx", type=int, default=-1)
    parser.add_argument("--checkpoint_load_best", action="store_true")
    parser.add_argument("--limit_requests", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=8192)
    parser.add_argument("--batch_size_per_device", type=int, default=None)
    parser.add_argument("--fsdp_size", type=int, default=1)
    parser.add_argument("--model_tp_size", type=int, default=1)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--generic_wandb_name", action="store_true")
    parser.add_argument("--no_logging", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--no_infinite_eval", action="store_true")
    parser.add_argument("--append_to_wandb_id", type=str, default=None)

    args = parser.parse_args()

    if args.limit_requests is not None:
        LOGGER.info(f"Limit LM Eval samples to {args.limit_requests}")

    main_lmeval(args)
