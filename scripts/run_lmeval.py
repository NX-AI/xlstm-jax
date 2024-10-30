import argparse
import logging
from pathlib import Path

# For now, we load MODEL_CONFIGS from this training script
# In the future the config should be loaded from the checkpoint
# or a configuration library.
from scripts.run_train_slimpajama6b import MODEL_CONFIGS

import jax

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


def main_lmeval(args: argparse.Namespace):
    if args.load_checkpoint_from:
        path = Path(args.load_checkpoint_from) / "checkpoints"
        metadata_path = (
            sorted(list(filter(lambda x: "checkpoint_" in str(x), path.iterdir())))[-1] / "metadata" / "metadata"
        )
        with open(metadata_path, encoding="utf8") as fp:
            metadata = fp.read()
        model_config_base = MODEL_CONFIGS[args.model]
        del model_config_base["model_config"]
        global_model_config = {"model_config": ModelConfig.from_metadata(metadata).model_config, **model_config_base}
        LOGGER.info("Loading model config from metadata file.")
    else:
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
        fsdp_axis_size=args.fsdp_size,
        model_axis_size=args.model_tp_size,
        data_axis_size=1,
        tp_async_dense=False,
    )
    mesh = initialize_mesh(parallel_config=parallel)
    log_info("Mesh initialized.")
    assert len(jax.devices(backend="gpu")) > 0, "No devices found. This script should be run on GPU."
    log_info(f"Devices: {jax.devices()}")

    # General hyperparameters.
    # lr = global_model_config.get("lr", 1e-3)
    log_path = Path(args.log_dir)

    # General hyperparameters.
    batch_size = (
        global_model_config["batch_size_per_device"]
        if args.batch_size_per_device is None
        else args.batch_size_per_device
    ) * len(jax.devices())
    context_length = args.context_length
    log_path = Path(args.log_dir)

    # Define model config - 120M parameters.
    if callable(global_model_config["model_config"]):
        xlstm_config = global_model_config["model_config"](parallel=parallel, context_length=context_length)
    else:
        xlstm_config = global_model_config["model_config"]
        xlstm_config.parallel = parallel
        xlstm_config.context_length = context_length
        xlstm_config.__post_init__()
    backend_name = xlstm_config.mlstm_block.mlstm.mlstm_cell.backend.name
    wb_name = f"eval_{args.model}_gbs{int(batch_size)}_ctx{context_length}_{backend_name}"

    # Create trainer with sub-configs.
    log_info("Creating trainer.")
    trainer = LLMTrainer(
        LLMTrainerConfig(
            callbacks=(
                LMEvalEvaluationConfig(
                    tokenizer_path="gpt2",
                    evaluation_tasks=args.tasks.split(","),
                    limit_requests=args.limit_requests,
                    context_length=args.context_length,
                ),
            ),
            logger=LoggerConfig(
                log_path=log_path,
                log_every_n_steps=20,
                log_tools=[
                    FileLoggerConfig(log_dir="file_logs", config_format="json"),
                    TensorBoardLoggerConfig(log_dir="tensorboard", tb_flush_secs=10),
                    WandBLoggerConfig(
                        wb_project="xlstm_jax",
                        wb_entity="xlstm",
                        wb_name=wb_name,
                        wb_tags=[args.model, "evaluation"],
                    ),
                ],
            ),
            check_val_every_n_epoch=1,
            enable_progress_bar=False,
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

    if len(args.load_checkpoint_from) > 0:
        log_info(f"Loading checkpoint from {args.load_checkpoint_from}.")
        trainer.load_pretrained_model(
            Path(args.load_checkpoint_from),
            step_idx=args.checkpoint_step_idx,
            load_best=args.checkpoint_load_best,
            load_optimizer=False,
            train_loader=None,
            val_loader=None,
        )

    trainer.logger.on_training_start()
    metrics = eval_callback.run_evaluate()
    for task in metrics:
        trainer.logger.log_host_metrics(metrics[task], step=trainer.global_step, mode=task)

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
    parser.add_argument("--checkpoint_step_idx", type=int, default=-1)
    parser.add_argument("--checkpoint_load_best", action="store_true")
    parser.add_argument("--limit_requests", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=2048)
    parser.add_argument("--batch_size_per_device", type=int, default=None)
    parser.add_argument("--fsdp_size", type=int, default=1)
    parser.add_argument("--model_tp_size", type=int, default=1)

    args = parser.parse_args()

    log_info(f"Limit LM Eval samples to {args.limit_requests}")

    if args.load_checkpoint_from:
        LOGGER.warning(
            "Loading from a checkpoint currently does not entail using the same model "
            "configuration. You will have to adapt the configuration in run_train_slimpajama6b.py for the "
            "specific model for now. Later we want to load the model config from the checkpoint as well."
        )

    main_lmeval(args)
