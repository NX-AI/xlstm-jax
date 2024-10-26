from pathlib import Path
from typing import Any

import jax
import pandas as pd
import pytest

from xlstm_jax.dataset import LLMBatch
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.trainer.eval.lmeval_extended_evaluation import LMEvalEvaluationConfig
from xlstm_jax.trainer.llm.trainer import LLMTrainer, LLMTrainerConfig
from xlstm_jax.trainer.logger import FileLoggerConfig, LoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig


@pytest.mark.parametrize("tp_size,fsdp_size", [(1, 1), (2, 4)])
def test_lmeval_evaluation(llm_toy_model: Any, tmp_path: Path, tp_size: int, fsdp_size: int):
    """
    Tests evaluation on a simple model under different mesh configs.
    """
    LLMToyModel = llm_toy_model
    if pytest.num_devices < tp_size * fsdp_size:
        pytest.skip("Test requires more devices than available.")
    batch_size = 8
    context_length = 16
    log_path = tmp_path / "test_extended_evaluation"
    fl_dir = log_path / "file_logs"
    model_config = ModelConfig(
        model_class=LLMToyModel,
        parallel=ParallelConfig(
            data_axis_size=-1,
            model_axis_size=tp_size,
            fsdp_axis_size=fsdp_size,
            fsdp_min_weight_size=pytest.num_devices,
        ),
    )
    optimizer_config = OptimizerConfig(
        name="adam",
        scheduler=SchedulerConfig(
            name="constant",
            lr=1e-4,
        ),
    )

    trainer = LLMTrainer(
        LLMTrainerConfig(
            callbacks=(
                LMEvalEvaluationConfig(
                    tokenizer_path="google/byt5-small",
                    evaluation_tasks=["lambada"],
                    limit_requests=2,
                    cache_requests=True,
                    context_length=context_length,
                ),
            ),
            logger=LoggerConfig(
                log_path=log_path,
                log_tools=[
                    FileLoggerConfig(log_dir=fl_dir),
                ],
                log_every_n_steps=1,
            ),
            check_val_every_n_epoch=1,
        ),
        model_config,
        optimizer_config,
        batch=LLMBatch.get_dtype_struct(batch_size=batch_size, max_length=context_length),
    )

    res = trainer.callbacks[0].run_evaluate()

    assert "lambada_openai" in res
    assert "perplexity,none" in res["lambada_openai"]

    def data_gen_fn(idx: int) -> LLMBatch:
        inputs = jax.random.randint(jax.random.PRNGKey(idx), (batch_size, context_length), minval=0, maxval=50)
        labels = inputs
        return LLMBatch.from_inputs(inputs=inputs, targets=labels)

    train_loader = [data_gen_fn(idx) for idx in range(8)]
    val_loader = train_loader

    _ = trainer.train_model(
        train_loader,
        val_loader,
        num_train_steps=2,
    )

    assert log_path.exists()
    assert (log_path / "output.log").exists()
    assert (log_path / fl_dir).exists(), f"Expected file logging directory {log_path / fl_dir} to exist"
    assert (
        log_path / fl_dir / "metrics_lambada_openai.csv"
    ).exists(), f"Expected metrics file {log_path / fl_dir / 'metrics_lambada_openai.csv'} to exist"
    df = pd.read_csv(log_path / fl_dir / "metrics_lambada_openai.csv")

    assert "perplexity,none" in df.columns, f"Expected 'perplexity,none' column in DataFrame {df.columns}"
