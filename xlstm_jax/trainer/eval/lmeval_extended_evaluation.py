import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from lm_eval import tasks
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.evaluator import evaluate

from xlstm_jax.dataset.batch import LLMBatch, LLMIndexedBatch
from xlstm_jax.dataset.configs import DataConfig
from xlstm_jax.dataset.grain_iterator import make_grain_llm_iterator
from xlstm_jax.dataset.input_pipeline_interface import get_process_loading_real_data
from xlstm_jax.dataset.lmeval_dataset import HFTokenizeLogLikelihoodRolling
from xlstm_jax.distributed import split_array_over_mesh
from xlstm_jax.trainer.callbacks.extended_evaluation import ExtendedEvaluation, ExtendedEvaluationConfig
from xlstm_jax.trainer.data_module import DataloaderModule
from xlstm_jax.trainer.metrics import HostMetrics, Metrics

PyTree = Any

LOGGER = logging.getLogger(__name__)


def fuse_document_results(
    results_dict: dict[str, np.ndarray | tuple[np.ndarray, np.ndarray]],
) -> list[tuple[float, bool]]:
    """
     Fuse log-likelihood results of capped sequences (max_length) to document log-likelihoods.
    Aggregate results from (potentially) many batches of (potentially) many different tasks.
    All results must have matching document indices and aggregate log-likelihoods over multiple
    sequence indices weighted by their counts.

    If the exact sequence is the result of a greedy decoding (i.e. if all single token accuracies of non-masked parts
    are 1), also aggregate "greedy" accuracies.
    Args:
        resuts_dict: Dictionary of all results in concatenated form.
    Returns:
        Log-likelihoods and greedy (boolean) Accuracy for documents ordered by index.
    """
    assert "loglikelihood_single_noreduce_wcount" in results_dict
    assert "document_idx_single_noreduce" in results_dict
    assert "sequence_idx_single_noreduce" in results_dict
    assert "accuracies_single_noreduce_wcount" in results_dict
    didx = results_dict["document_idx_single_noreduce"]
    llhs = results_dict["loglikelihood_single_noreduce_wcount"][0]
    llhs_count = results_dict["loglikelihood_single_noreduce_wcount"][1]
    accs = results_dict["accuracies_single_noreduce_wcount"][0]
    accs_count = results_dict["accuracies_single_noreduce_wcount"][1]

    idxs = np.argsort(results_dict["document_idx_single_noreduce"])
    previous_didx = -1
    llh = 0.0
    acc = 0.0
    llh_count = 0
    acc_count = 0
    first = True
    results = []
    for idx in idxs:
        if didx[idx] != previous_didx and not first:
            results.append((llh, bool(acc == acc_count)))
            llh = 0.0
            acc = 0.0
            llh_count = 0
            acc_count = 0
        previous_didx = didx[idx]
        if didx[idx] == -1:
            continue
        first = False
        llh += llhs[idx]
        llh_count += llhs_count[idx]
        acc += accs[idx]
        acc_count += accs_count[idx]
        assert np.allclose(llh_count, acc_count)
    if not first:
        results.append((llh, bool(acc == acc_count)))
    return results


@dataclass(kw_only=True, frozen=False)
class LMEvalEvaluationConfig(ExtendedEvaluationConfig):
    tokenizer_path: str  # = "gpt2"
    """Tokenizer path"""  # TODO: Check if tokenizer can be seen from trainer / model_config, then re-use it
    evaluation_tasks: list[str]
    """List of evaluation task from LM Evaluation Harness"""
    cache_requests: bool = True
    """Whether to cache requests"""
    limit_requests: int | None = None
    """Whether to limit requests to a smaller number for debugging purposes"""
    write_out: bool = False
    """Whether to write out results"""
    context_length: int | None = None
    """ Override context_length of the model """
    batch_size: int | None = None
    """ Override batch_size of the trainer """
    worker_buffer_size: int = 1
    """ Worker buffer size for the grain loader """
    worker_count: int = 0
    """ Number of workers for the grain loading """
    debug: bool = False
    """ Scale ouputs such that metrics can be computed for a random testing model """

    def create(self, trainer: Any, data_module: DataloaderModule | None = None) -> "LMEvalEvaluation":
        """
        Args:
            trainer: Trainer
            data_module: DataloaderModule containing train/val/test - not used here

        Returns:
            LMEvalEvaluation object
        """
        return LMEvalEvaluation(config=self, trainer=trainer, data_module=data_module)


class LMEvalEvaluation(ExtendedEvaluation):
    """
    LMEvalEvaluation Callback
    """

    def __init__(
        self,
        config: LMEvalEvaluationConfig,
        trainer: Any,
        data_module: DataloaderModule | None = None,
    ):
        """
        Args:
            config: LMEvalEvaluationConfig
            trainer: Trainer
            data_module: DataloaderModule containing train/val/test - not used here
        """
        super().__init__(config=config, trainer=trainer, data_module=data_module)

        context_length = (
            self.config.context_length
            if self.config.context_length is not None
            else self.trainer.model_config.model_config.context_length
        )
        self.context_length = context_length
        batch_size = (
            self.config.batch_size if self.config.batch_size is not None else self.trainer.exmp_batch.inputs.shape[0]
        )
        self.batch_size = batch_size
        instance = self

        def _make_grain_iterator(dataset):
            # We have to use the sample distribution as in the training loop,
            # unless we want a different sharding, which probably doesn't make sense.
            # However the worker_count is 0 now by default, but should not be a bottleneck.
            process_indices = get_process_loading_real_data(
                DataConfig(global_batch_size=batch_size, max_target_length=context_length), mesh=trainer.mesh
            )
            it = make_grain_llm_iterator(
                dataloading_host_index=process_indices.index(jax.process_index()),
                dataloading_host_count=len(process_indices),
                global_mesh=trainer.mesh,
                dataset=dataset,
                global_batch_size=batch_size,
                max_target_length=2048,  # not actually used without padding
                worker_count=config.worker_count,
                shuffle=False,
                data_shuffle_seed=0,
                num_epochs=1,
                shift=False,
                batch_class=LLMIndexedBatch,
                reset_after_epoch=True,
                drop_remainder=False,
                apply_padding=False,
                grain_packing=False,
                worker_buffer_size=config.worker_buffer_size,
            )
            return it

        class _AdaptedLM(LM):
            """
            LM class for LMEval harness.
            """

            def __init__(self):
                super().__init__()
                self.mode = "test"

            def set_mode(self, mode):
                """
                Set the mode different to test for running the evaluation loop.
                """
                self.mode = mode

            def generate_until(self, requests: list[Instance]) -> list[str]:
                """
                This is a dummy currently to satisfy the requirements of LMEval for generation tasks.

                Args:
                    requests: List of LM Eval Instances

                Returns:
                    List of generated strings.
                """
                res = []

                for request in requests:
                    res.append("lol")
                    assert request.arguments[0].strip() != ""

                return res

            def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
                """
                Compute loglikelihood of typically shorter sequences with a prefix.

                Args:
                    requests: List of LM Eval Instances

                Returns:
                    List of loglikelihoods + greedy (boolean accuracy)
                """
                LOGGER.info("Start create dataset")
                dataset = HFTokenizeLogLikelihoodRolling(
                    tokenizer_path=config.tokenizer_path,
                    batch_size=batch_size,
                    max_length=context_length,
                ).map(requests)
                LOGGER.info("End create dataset")
                assert (
                    len(dataset) % batch_size == 0
                ), f"Dataset size no divisible by batch_size {len(dataset)} % {batch_size}"
                LOGGER.info("Start create data iterator")
                it = _make_grain_iterator(dataset)
                LOGGER.info("End create data iterator")

                LOGGER.info("Start inference")
                final_metrics = ExtendedEvaluation.eval_model(instance, it, mode=self.mode)
                LOGGER.info("End inference")
                LOGGER.info("Start results postprocessing")
                res = fuse_document_results(final_metrics)
                LOGGER.info("End results postprocessing")
                assert len(res) == len(requests), f"Mis-match of requests and results: {len(requests)} != {len(res)}"
                return res

            def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
                """
                Compute loglikelihood of longer sequences that might even be split in case they
                overflow the maximal context length of a model.

                Args:
                    requests: List of LM Eval Instances

                Returns:
                    List of loglikelihoods
                """
                res = self.loglikelihood(requests=requests)
                return [r[0] for r in res]

        self.lm = _AdaptedLM()

    def create_modified_exemplary_batch(self, exmp_batch: LLMBatch) -> LLMIndexedBatch:
        """
        Create an LLMIndexedBatch from a LLMBatch (for compilation purposes as example).

        Args:
            examp_batch: LLMBatch

        Returns:
            LLMIndexedBatch
        """
        return LLMIndexedBatch(
            inputs=exmp_batch.inputs,
            targets=exmp_batch.targets,
            inputs_position=exmp_batch.inputs_position,
            targets_position=exmp_batch.targets_position,
            inputs_segmentation=exmp_batch.inputs_segmentation,
            targets_segmentation=exmp_batch.targets_segmentation,
            document_idx=-jnp.ones([exmp_batch.inputs.shape[0]], dtype=jnp.int32),
            sequence_idx=jnp.zeros([exmp_batch.inputs.shape[0]], dtype=jnp.int32),
        )

    def run_evaluate(self) -> HostMetrics:
        """
        Runs the evaluation in LM Eval Harness. Does use external datasets.
        Might be called from callback functions to get metrics during training.

        Returns:
            Results from LMEval evaluation.
        """
        LOGGER.info("Running LMEval evaluation.")
        res = evaluate(
            self.lm,
            task_dict=tasks.get_task_dict(self.config.evaluation_tasks),
            limit=self.config.limit_requests,
            cache_requests=self.config.cache_requests,
            write_out=self.config.write_out,
        )["results"]
        for task in res:
            del res[task]["alias"]
            # convert "N/A" values to NaN values
            res[task] = {key: val if val != "N/A" else float("NaN") for key, val in res[task].items()}
        return res

    def get_metric_postprocess_fn(self) -> Callable[[HostMetrics], HostMetrics]:
        """
        Get function to post-process metrics with on host.

        Will be passed to logger. Adds perplexity to the metrics.

        Returns:
            The postprocess metric function.
        """

        def _postprocess_metrics(metrics: HostMetrics) -> HostMetrics:
            """Add perplexity to the metrics."""
            for key in ["loss", "loss_mean", "loss_single"]:
                if key in metrics:
                    metrics[f"perplexity{key[4:]}"] = np.exp(metrics[key])
            return metrics

        return _postprocess_metrics

    def eval_function(
        self, params: Any, apply_fn: Any, batch: LLMIndexedBatch, rng: jax.Array
    ) -> tuple[Metrics, PyTree]:
        """
        Function that passes the batch through the model and generates some extended metrics.

        Args:
            params: Model parameters.
            apply_fn: Model functions.
            batch: LLMIndexedBatch that is passed through the model.
            rng: RNG for potential dropout.

        Returns:
            Tuple with Metrics and MutableVariables.
        """
        _ = rng
        # Remaining computation is the same as before for single device.
        logits, mutable_variables = apply_fn(
            {"params": params},
            batch.inputs,
            train=False,
            mutable="intermediates",
        )
        # Select the targets per device.
        targets = batch.targets
        targets = split_array_over_mesh(targets, axis_name=self.trainer.pipeline_axis_name, split_axis=1)
        targets = split_array_over_mesh(targets, axis_name=self.trainer.model_axis_name, split_axis=1)
        assert (
            logits.shape[:-1] == targets.shape
        ), f"Logits and targets shapes do not match: {logits.shape} vs {targets.shape}"
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        correct_pred = jnp.equal(jnp.argmax(logits, axis=-1), targets)
        # Mask out padding tokens.
        targets_mask = batch.targets_segmentation != 0
        targets_mask = split_array_over_mesh(targets_mask, axis_name=self.trainer.pipeline_axis_name, split_axis=1)
        targets_mask = split_array_over_mesh(targets_mask, axis_name=self.trainer.model_axis_name, split_axis=1)
        loss = loss * targets_mask
        correct_pred = correct_pred * targets_mask
        num_targets = targets_mask.sum()
        # Collect metrics and return loss.
        step_metrics = {
            "loss": {"value": loss.sum(), "count": num_targets},
            "accuracy": {
                "value": correct_pred.sum(),
                "count": num_targets,
            },
            "accuracies": {
                "value": correct_pred.sum(axis=1),
                "count": targets_mask.sum(axis=1),
                "log_modes": ("single_noreduce_wcount",),
            },
            "loglikelihood": {
                # the factor prevents overflows in the debug case for a random model
                "value": -loss.sum(axis=1) * (0.0001 if self.config.debug else 1.0),
                "count": targets_mask.sum(axis=1),
                "log_modes": ("single_noreduce_wcount",),
            },
            "document_idx": {
                "value": batch.document_idx,
                "count": batch.document_idx.shape[0],
                "log_modes": ("single_noreduce",),
            },
            "sequence_idx": {
                "value": batch.sequence_idx,
                "count": batch.document_idx.shape[0],
                "log_modes": ("single_noreduce",),
            },
        }

        return step_metrics, mutable_variables

    def on_filtered_validation_epoch_start(self, epoch_idx: int, step_idx: int):
        """
        Runs evaluation on filtered validation epochs / steps.

        Args:
            epoch_idx: Epoch Index
            step_idx: Step Index
        """
        metrics = self.run_evaluate()
        for task in metrics:
            self.trainer.logger.log_host_metrics(metrics[task], step=step_idx, mode=task)

    def on_test_epoch_start(self, epoch_idx: int):
        """
        Runs evaluation on test_epoch.

        Args:
            epoch_idx: Epoch index
        """
        metrics = self.run_evaluate()
        for task in metrics:
            self.trainer.logger.log_host_metrics(metrics[task], step=self.trainer.global_step, mode="task")
