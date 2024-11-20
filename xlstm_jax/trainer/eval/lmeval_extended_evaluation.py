import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict, freeze
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.evaluator import simple_evaluate

from xlstm_jax.common_types import HostMetrics, ImmutableMetrics, Metrics, PyTree, TrainState
from xlstm_jax.dataset.batch import LLMBatch, LLMIndexedBatch
from xlstm_jax.dataset.configs import DataConfig
from xlstm_jax.dataset.grain_iterator import make_grain_llm_iterator
from xlstm_jax.dataset.input_pipeline_interface import get_process_loading_real_data
from xlstm_jax.dataset.lmeval_dataset import HFTokenizeLogLikelihoodRolling
from xlstm_jax.dataset.lmeval_pipeline import lmeval_preprocessing_pipeline
from xlstm_jax.distributed import split_array_over_mesh
from xlstm_jax.trainer.callbacks.extended_evaluation import (
    ExtendedEvaluation,
    ExtendedEvaluationConfig,
    device_metrics_aggregation,
)
from xlstm_jax.trainer.data_module import DataloaderModule
from xlstm_jax.trainer.llm.trainer import LLMTrainer
from xlstm_jax.trainer.metrics import (
    aggregate_metrics,
    get_metrics,
    update_metrics,
)

LOGGER = logging.getLogger(__name__)


def log_info(msg: str):
    """Logs an info message on the host device.

    Args:
        msg: Message to be logged.
    """
    if jax.process_index() == 0:
        LOGGER.info(msg)


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
    previous_didx = 0
    llh = 0.0
    acc = 0.0
    llh_count = 0
    acc_count = 0
    first = True
    results = []
    for idx in idxs:
        if didx[idx] != previous_didx and not first:
            if didx[idx] != previous_didx + 1:
                LOGGER.warning(f"Missing document index in between: {previous_didx} and {didx[idx]}")
            results.append((llh, bool(acc == acc_count)))
            llh = 0.0
            acc = 0.0
            llh_count = 0
            acc_count = 0
        previous_didx = didx[idx]
        # ignore documents with document_idx padding value
        if didx[idx] == 0:
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
    use_infinite_eval: bool = True
    """Whether to use the infinite eval"""
    infinite_eval_chunksize: int = 64
    """The chunk size for using the infinite eval"""
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
    system_instruction: str | None = None
    """ Additional system instruction """
    num_fewshot: int | None = None
    """ Define number of in-context samples for few-shot training. """
    bootstrap_iters: int = 1000
    """ Bootstrap iterations for calculating stderrs on metrics - LMEval standard is 100000, limit here for speed """
    apply_chat_template: bool | str = False
    """ Apply the LMEval chat template, or a custom template """

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
        trainer: LLMTrainer,
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
                if config.use_infinite_eval:
                    process_indices = get_process_loading_real_data(
                        DataConfig(global_batch_size=batch_size, max_target_length=context_length), mesh=trainer.mesh
                    )
                    it = lmeval_preprocessing_pipeline(
                        dataloading_host_index=process_indices.index(jax.process_index()),
                        dataloading_host_count=len(process_indices),
                        global_mesh=trainer.mesh,
                        dataset=requests,
                        global_batch_size=batch_size,
                        tokenizer_path=config.tokenizer_path,
                        worker_count=0,
                        worker_buffer_size=1,
                        padding_multiple=config.infinite_eval_chunksize,
                    )
                else:
                    dataset = HFTokenizeLogLikelihoodRolling(
                        tokenizer_path=config.tokenizer_path,
                        batch_size=batch_size,
                        max_length=context_length,
                    ).map(requests)
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

                log_info("Start inference")
                final_metrics = ExtendedEvaluation.eval_model(instance, it, mode=self.mode)
                log_info("End inference")

                log_info("Start results postprocessing")
                res = fuse_document_results(final_metrics)
                log_info("End results postprocessing")
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
        batch = LLMIndexedBatch(
            inputs=exmp_batch.inputs,
            targets=exmp_batch.targets,
            inputs_position=exmp_batch.inputs_position,
            targets_position=exmp_batch.targets_position,
            inputs_segmentation=exmp_batch.inputs_segmentation,
            targets_segmentation=exmp_batch.targets_segmentation,
            document_idx=-jnp.ones([exmp_batch.inputs.shape[0]], dtype=jnp.int32),
            sequence_idx=jnp.zeros([exmp_batch.inputs.shape[0]], dtype=jnp.int32),
        )
        if self.config.use_infinite_eval:
            if isinstance(exmp_batch, jax.ShapeDtypeStruct):
                return LLMIndexedBatch.get_dtype_struct(exmp_batch.inputs.shape[0], self.config.infinite_eval_chunksize)
            else:
                return LLMIndexedBatch.get_sample(exmp_batch.inputs.shape[0], self.config.infinite_eval_chunksize)
        else:
            return batch

    def run_evaluate(self) -> HostMetrics:
        """
        Runs the evaluation in LM Eval Harness. Does use external datasets.
        Might be called from callback functions to get metrics during training.

        Returns:
            Results from LMEval evaluation.
        """
        log_info("Running LMEval evaluation.")
        res = simple_evaluate(
            self.lm,
            tasks=self.config.evaluation_tasks,
            limit=self.config.limit_requests,
            cache_requests=self.config.cache_requests,
            write_out=self.config.write_out,
            num_fewshot=self.config.num_fewshot,
            apply_chat_template=self.config.apply_chat_template,
            system_instruction=self.config.system_instruction,
            bootstrap_iters=self.config.bootstrap_iters,
        )["results"]
        for task in res:
            del res[task]["alias"]
            # convert "N/A" values to NaN values
            res[task] = {key: val if val != "N/A" else float("NaN") for key, val in res[task].items()}
        log_info(f"LMEval results: {res}")
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

    def create_jitted_functions(self):
        """
        Create jitted version of the evaluation function.
        """
        if self.config.use_infinite_eval:
            eval_step, eval_single_step = self.create_recurrent_evaluation_step_function(
                chunk_size=self.config.infinite_eval_chunksize,
                exmp_batch=LLMIndexedBatch.get_dtype_struct(
                    self.trainer.exmp_batch.inputs.shape[0], self.config.infinite_eval_chunksize
                ),
            )
            super().init_eval_metrics(alternative_eval_step=eval_single_step)
            self.eval_step = eval_step
            self.eval_single_step = eval_single_step
        else:
            super().create_jitted_functions()

    def init_eval_metrics(self, batch: LLMIndexedBatch | None = None) -> FrozenDict | dict:
        """
        Override parent init_eval_metrics potentially for infinite eval.
        Then metrics are partly aggregated one level below (along the sequence) and aggregated fully
        (across batches) within `eval_model`.

        Args:
            batch: An input to the model with which the shapes are inferred. If None, the :attr:`exmp_batch` is used.

        Returns:
            A dictionary of metrics with the same shape as the eval metrics.
        """
        if self.config.use_infinite_eval:
            step_metrics = super().init_eval_metrics(alternative_eval_step=self.eval_single_step)
            return {"step_metrics": step_metrics}
        else:
            return super().init_eval_metrics()

    def aggregate_metrics(self, aggregated_metrics: HostMetrics, eval_metrics: HostMetrics) -> HostMetrics:
        """
        Aggregate metrics over multiple batches. This is an adaption of the parent class that ignores the
        passed "step_metrics" in the metrics dictionary. The "step_metrics" are single recurrent step
        metrics to be donated for a future evaluation step.

        Args:
            aggregated_metrics: Old aggregated metrics
            eval_metrics: Single batch metrics

        Returns:
            aggregated_metrics including the new batch

        """
        if self.config.use_infinite_eval:
            metrics = eval_metrics.copy()
            metrics.pop("step_metrics")
            return aggregate_metrics(aggregated_metrics, metrics)
        else:
            return super().aggregate_metrics(aggregated_metrics, eval_metrics)

    def create_recurrent_evaluation_step_function(
        self,
        chunk_size: int,
        exmp_batch: LLMIndexedBatch | None = None,
        cache_init_fn: Callable[[PyTree], PyTree] | None = None,
    ) -> Callable[[TrainState, LLMBatch, ImmutableMetrics | None], ImmutableMetrics]:
        """
        Create and return a recurrent function for the evaluation step. (see also llm/trainer.py).

        Compared to the `create_evaluation_step_function`, this evaluation supports much longer sequences by chunking
        the input and running the model recurrently over the chunks. This is useful for evaluation on long documents.
        This is enabled by keeping a cache, which is forwarded between evaluation steps.

        Note: do *not* jit this function if you want to support arbitrary input shapes. This function jit's the
        recurrent function for a single chunk, and adds a python loop around it to handle arbitrary length sequences.
        Thus, no outer jit is needed.

        Note: this function is explicitly meant for recurrent models like xLSTM. Using this function on a non-recurrent
        model will lead to unexpected, incorrect results.

        Args:
            chunk_size: Size of the chunks to split the input into. The slices are performed over the sequence length.
            exmp_batch: An example batch to determine the shape of the cache. Defaults to None, in which case the
                example batch from the trainer is used.
            cache_init_fn: A function to initialize the cache. If not provided, the cache is initialized with zeros.
                The function should take the shape dtype struct of the cache as input and return the initialized cache.

        Returns:
            The evaluation step function with support for arbitrary length sequences.
        """
        if cache_init_fn is None:
            cache_init_fn = partial(jax.tree.map, jnp.zeros_like)

        def rec_eval_step(
            state: TrainState, cache: PyTree, batch: LLMIndexedBatch, metrics: ImmutableMetrics | None
        ) -> tuple[PyTree, ImmutableMetrics]:
            """
            Recurrent evaluation step function.

            Args:
                state: Trainer state
                cache: Recurrent model cache (state)
                batch: Batch to be evaluated
                metrics: Old metrics to be updated / taken as basis for new metrics.

            Returns:
                New recurrent cache (state), new metrics
            """
            # In our multi-host setup, each local device will have a different batch.
            # So we first gather the batch across model and pipeline axes.
            batch = jax.lax.all_gather(
                batch, axis_name=(self.trainer.model_axis_name, self.trainer.pipeline_axis_name), axis=0, tiled=True
            )
            # Forward pass and compute metrics.
            step_metrics, mutable_variables = self.eval_function(
                state.params,
                state.apply_fn,
                batch,
                jax.random.PRNGKey(self.trainer.trainer_config.seed_eval),
                mutable_variables={"cache": cache},
            )
            cache = mutable_variables["cache"]
            with jax.named_scope("sync_metrics"):
                step_metrics = device_metrics_aggregation(trainer=self.trainer, metrics=step_metrics)
            metrics = update_metrics(metrics, step_metrics, default_log_modes=("mean_nopostfix",))
            return cache, metrics

        # Determine cache structure.
        cache_shape = self.trainer.get_cache_shape_dtype_struct(exmp_batch=exmp_batch)

        # Shard the single-step recurrent evaluation function.
        state_partition_specs = nn.get_partition_spec(self.trainer.state)
        cache_partition_specs = nn.get_partition_spec(cache_shape)
        rec_eval_step_fn = shard_map(
            rec_eval_step,
            self.trainer.mesh,
            in_specs=(state_partition_specs, cache_partition_specs, self.trainer.batch_partition_specs, P()),
            out_specs=P(),
            check_rep=False,
        )

        # Jit the step function. We donate also the cache to free up memory.
        if not self.trainer.trainer_config.debug:
            rec_eval_step_fn = jax.jit(
                rec_eval_step_fn,
                donate_argnames=["cache", "metrics"],
            )

        def single_eval_step_fn(
            state: TrainState, batch: LLMIndexedBatch, metrics: ImmutableMetrics | None
        ) -> ImmutableMetrics:
            """
            Creates a single evaluation step function (without loop) to enable the init_eval_metrics function
            as it is needed to compile the elementary step.
            The looped eval step however has variable metrics output shapes.
            """
            cache = cache_init_fn(cache_shape)
            cache, metrics = rec_eval_step_fn(state, cache, batch, metrics)
            return metrics

        # Create the looped evaluation step function.
        def looped_eval_step_fn(state: TrainState, batch: LLMIndexedBatch, metrics: HostMetrics | None) -> HostMetrics:
            """
            Evaluation step function that internally uses a loop over chunks.

            Args:
                state: The trainer/evaluator state.
                batch: The LLMIndexedBatch (of potentially very large sequence length)
                metrics: The old metrics containing the `step_metrics` entry for single (recurrent) step metrics.
                         These step_metrics contain the full validation metrics for "non-growing" metrics.
                         Other metrics (per batch, per sample - "noreduce" metrics) are ignored and
                         aggregated one level above. They are aggregated also internally starting from empty metrics.

            Returns:
                Update metrics after processing the batch.
            """
            # Create initial cache.
            cache = cache_init_fn(cache_shape)
            # Run the evaluation per chunk with forwarding the cache.
            num_chunks = (batch.inputs.shape[1] - 1) // chunk_size + 1
            # step_metrics are part of the global metrics dict here, as they need to be donated
            # for memory re-use. What is returns also contains the step metrics plus additional
            # "batch host metrics" aggregated over multiple "sequence-internal" steps
            # These are aggregated with the global "validation metrics" in `:func:eval_model`
            step_metrics = metrics["step_metrics"]

            # do not use outer aggregated metrics here, aggregate internally first
            # the aggregation with the outer metrics happens in the evaluation loop
            non_step_metrics = {}

            for i in range(num_chunks):
                # Get current chunk of batch.
                chunk_start = i * chunk_size
                chunk_end = min((i + 1) * chunk_size, batch.inputs.shape[1])
                chunk_batch = batch[:, chunk_start:chunk_end]
                if chunk_batch.inputs.shape[1] < chunk_size:
                    # Pad batch.
                    pad_size = chunk_size - chunk_batch.inputs.shape[1]
                    chunk_batch = jax.tree.map(
                        lambda x: jnp.pad(x, ((0, 0), (0, pad_size)), mode="constant") if x.ndim > 1 else x, chunk_batch
                    )
                # Run the evaluation step.
                cache, step_metrics = rec_eval_step_fn(state, cache, chunk_batch, step_metrics)
                non_step_metrics = aggregate_metrics(aggregated_metrics=non_step_metrics, batch_metrics=step_metrics)

            # Return final metrics, including step_metrics that are needed for re-donation.
            non_step_metrics["step_metrics"] = step_metrics
            metrics = non_step_metrics
            return freeze(metrics)

        return looped_eval_step_fn, single_eval_step_fn

    def finalize_metrics(self, aggregated_metrics: HostMetrics) -> HostMetrics:
        """
        Calculate final metrics from aggregated_metrics. (i,e, mean=sum/count)

        Args:
            aggregated_metrics: Aggregated metrics over the whole epoch

        Returns:
            Final metrics that are to be reported / logged.
        """
        if self.config.use_infinite_eval:
            aggregated_metrics = aggregated_metrics
            if "step_metrics" in aggregated_metrics:
                aggregated_metrics.pop("step_metrics")
        return get_metrics(aggregated_metrics)[1]

    def eval_function(
        self,
        params: Any,
        apply_fn: Any,
        batch: LLMIndexedBatch,
        rng: jax.Array | None = None,
        mutable_variables: dict[str, Any] | None = None,
    ) -> tuple[Metrics, PyTree]:
        """
        Function that passes the batch through the model and generates some extended metrics.

        Args:
            params: Model parameters.
            apply_fn: Model functions.
            batch: LLMIndexedBatch that is passed through the model.
            rng: RNG for potential dropout.
            mutable_variables: Mutable variables for the evaluation step function, e.g. the cache (recurrent state).

        Returns:
            Tuple with Metrics and MutableVariables.
        """
        _ = rng
        # Remaining computation is the same as before for single device.
        if mutable_variables is None:
            mutable_variables = {}
        logits, mutable_variables = apply_fn(
            {"params": params, **mutable_variables},
            batch.inputs,
            pos_idx=batch.inputs_position,
            document_borders=batch.get_document_borders(),
            train=False,
            mutable=["intermediates"] + list(mutable_variables.keys()),
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
