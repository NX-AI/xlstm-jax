# Standard libraries
import importlib
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from copy import copy
from dataclasses import dataclass
from functools import partial
from glob import glob
from typing import (
    Any,
    Dict,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

# ML collections for config
from xlstm_jax.configs import ConfigDict
from xlstm_jax.dataset import Batch
from xlstm_jax.distributed.common_types import PRNGKeyArray
from xlstm_jax.distributed.data_parallel import sync_gradients
from xlstm_jax.distributed.single_gpu import accumulate_gradients
from xlstm_jax.import_utils import resolve_import
from xlstm_jax.models import ModelConfig
from xlstm_jax.trainer import callbacks
from xlstm_jax.trainer.callbacks import CallbackConfig, ModelCheckpoint
from xlstm_jax.trainer.logger import Logger, LoggerConfig
from xlstm_jax.trainer.metrics import HostMetrics, ImmutableMetrics, Metrics, get_metrics, update_metrics
from xlstm_jax.trainer.optimizer import OptimizerConfig, build_optimizer

import flax

# JAX/Flax libraries
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from absl import flags, logging
from flax import linen as nn
from flax.core import FrozenDict, freeze, unfreeze
from flax.training import train_state
from jax import random
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
from tabulate import tabulate as python_tabulate
from tqdm.auto import tqdm

from .param_utils import tabulate_params
from .train_state import TrainState


@dataclass(kw_only=True, frozen=True)
class TrainerConfig(ConfigDict):
    seed: int = 0
    debug: bool = False
    donate_train_state: bool = True
    enable_progress_bar: bool = True
    gradient_accumulate_steps: int = 1
    check_val_every_n_epoch: int = 1
    logger: LoggerConfig = LoggerConfig()
    callbacks: Sequence[CallbackConfig] = ()
    seed_eval: int = 0


class TrainerModule:
    def __init__(
        self,
        trainer_config: TrainerConfig,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        batch: Batch,
    ):
        """A basic Trainer module summarizing most common training functionalities like logging,
        model initialization, training loop, etc.

        Args:
            trainer_config: A dictionary containing the trainer configuration.
            model_config: A dictionary containing the model configuration.
            optimizer_config: A dictionary containing the optimizer configuration.
            batch: An input to the model with which the shapes are inferred.
        """
        super().__init__()
        self.trainer_config = trainer_config
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.exmp_batch = batch
        # Setup parallel mesh
        self.init_mesh(model_config)
        # Create empty model. Note: no parameters yet
        self.build_model(model_config)
        # Init trainer parts
        self.init_optimizer(optimizer_config)
        self.init_model(batch)
        self.create_jitted_functions()
        self.init_logger(self.trainer_config.logger)
        self.init_callbacks(self.trainer_config.callbacks)

    def batch_to_input(self, batch: Batch) -> Any:
        """Convert a batch to the input format expected by the model.

        Needs to be implemented by the subclass if batch.inputs is not
        sufficient.

        Args:
            batch: A batch of data.

        Returns:
            The input to the model.
        """
        return batch.inputs

    def init_mesh(self, model_config: ConfigDict):
        """Initialize the mesh for parallel training.

        Args:
            model_config: A dictionary containing the model configuration, including the parallelization parameters.
        """
        if "SLURM_STEP_NODELIST" in os.environ:
            # Initializes one process per device, using the SLURM environment variables.
            # TODO: We may need to do this already before data loading, so very early in the run script.
            # To be checked once the framework is more mature.
            jax.distributed.initialize()
        # Save axis names to trainer for easier usage.
        self.data_axis_name = model_config.parallel.data_axis_name
        self.fsdp_axis_name = model_config.parallel.fsdp_axis_name
        self.pipeline_axis_name = model_config.parallel.pipeline_axis_name
        self.model_axis_name = model_config.parallel.model_axis_name
        # Setup device structure.
        device_array = np.array(jax.devices()).reshape(
            model_config.parallel.data_axis_size,
            model_config.parallel.fsdp_axis_size,
            model_config.parallel.pipeline_axis_size,
            model_config.parallel.model_axis_size,
        )
        # Initialize mesh.
        self.mesh = Mesh(
            device_array,
            (
                self.data_axis_name,
                self.fsdp_axis_name,
                self.pipeline_axis_name,
                self.model_axis_name,
            ),
        )
        if jax.process_index() == 0:
            logging.info(f"Initialized mesh with {self.mesh}.")

    def build_model(self, model_config: ConfigDict):
        """Create the model class from the model_config.

        Args:
            model_config: A dictionary containing the model configuration.
        """
        self.model = model_config.model_class(
            model_config if model_config.model_config is None else model_config.model_config
        )

    def init_logger(self, logger_config: ConfigDict):
        """Initialize a logger and creates a logging directory.

        Args:
            logger_params: A dictionary containing the specification of the logger.
        """
        self.logger = Logger(logger_config)
        self.log_dir = self.logger.log_dir

    def init_callbacks(self, callback_configs: Sequence[CallbackConfig]):
        """Initialize the callbacks defined in the trainer config."""
        self.callbacks = []
        for cb_config in callback_configs:
            logging.info(f"Initializing callback {cb_config.__class__.__name__}")
            if hasattr(callbacks, cb_config.class_name):
                callback_class = getattr(callbacks, cb_config.class_name)
            else:
                callback_class = resolve_import(cb_config.class_name)
            callback = callback_class(config=cb_config, trainer=self, data_module=None)
            self.callbacks.append(callback)

    def init_optimizer(self, optimizer_config: ConfigDict):
        """Initialize the optimizer.

        Args:
            optimizer_config: A dictionary containing the optimizer configuration.
        """
        self.optimizer, self.lr_scheduler = build_optimizer(optimizer_config)

    def init_model(self, exmp_input: Batch):
        """Create an initial training state with newly generated network parameters.

        This init function is parallelized over the mesh to initialize the per-device
        parameters. It also initializes the optimizer parameters. As a result, it sets
        the training state of the trainer with the initialized parameters.

        Args:
            exmp_input: An input to the model with which the shapes are inferred.
        """

        def _init_model(init_rng: PRNGKeyArray, batch: Batch) -> TrainState:
            param_rng, init_rng = jax.random.split(init_rng)
            variables = self.run_model_init(batch, param_rng)
            params = variables.pop("params")
            if isinstance(variables, FrozenDict):
                mutable_variables, params = variables.pop("params")
            else:
                params = variables.pop("params")
                mutable_variables = variables
            if len(mutable_variables) == 0:
                mutable_variables = None
            # Create train state.
            state = TrainState.create(
                apply_fn=self.model.apply,
                params=params,
                mutable_variables=mutable_variables,
                rng=init_rng,
                tx=self.optimizer,
            )
            return state

        # Prepare PRNG.
        init_rng = random.PRNGKey(self.trainer_config.seed)
        # First infer the output sharding to set up shard_map correctly.
        # This does not actually run the init, only evaluates the shapes.
        init_model_fn = jax.jit(
            shard_map(
                _init_model,
                self.mesh,
                in_specs=(P(), P((self.data_axis_name, self.fsdp_axis_name))),
                out_specs=P(),
                check_rep=False,
            ),
        )
        state_shapes = jax.eval_shape(init_model_fn, init_rng, exmp_input)
        state_specs = nn.get_partition_spec(state_shapes)
        # Run init model function again with correct output specs.
        init_model_fn = jax.jit(
            shard_map(
                _init_model,
                self.mesh,
                in_specs=(P(), P((self.data_axis_name, self.fsdp_axis_name))),
                out_specs=state_specs,
                check_rep=False,
            ),
        )
        self.state = init_model_fn(init_rng, exmp_input)

    def init_train_metrics(self, batch: Batch | None = None) -> FrozenDict:
        """Initialize the training metrics with zeros.

        We infer the training metric shape from the train_step function. This is done to
        prevent a double-compilation of the train_step function, where the first step has
        to be done with metrics None, and the next one with the metrics shape.

        Args:
            batch: An input to the model with which the shapes are inferred. If None,
            the exmp_input is used.

        Returns:
            A dictionary of metrics with the same shape as the train metrics.
        """
        if not hasattr(self, "train_metric_shapes"):
            self.train_metric_shapes = None
        if self.train_metric_shapes is None:
            if batch is None:
                batch = self.exmp_batch
            _, self.train_metric_shapes = jax.eval_shape(self.train_step, self.state, batch, None)
        return jax.tree.map(lambda x: jnp.zeros_like(x), self.train_metric_shapes)

    def init_eval_metrics(self, batch: Batch | None = None) -> FrozenDict:
        """Initialize the evaluation metrics with zeros.

        See init_train_metrics for more details.

        Args:
            batch: An input to the model with which the shapes are inferred. If None,
            the exmp_input is used.

        Returns:
            A dictionary of metrics with the same shape as the eval metrics.
        """
        if not hasattr(self, "eval_metric_shapes"):
            self.eval_metric_shapes = None
        if self.eval_metric_shapes is None:
            if batch is None:
                batch = self.exmp_batch
            self.eval_metric_shapes = jax.eval_shape(self.eval_step, self.state, batch, None)
        return jax.tree.map(lambda x: jnp.zeros_like(x), self.eval_metric_shapes)

    def set_dataset(self, dataset: Any):
        """Set the dataset for the trainer and the callbacks.

        Args:
            dataset: The dataset to set.
        """
        for callback in self.callbacks:
            callback.set_dataset(dataset)
        self.dataset = dataset

    def get_model_rng(self, rng: jax.Array) -> dict[str, random.PRNGKey]:
        """Return a dictionary of PRNGKey for init and tabulate.

        By default, adds a key for the parameters and one for dropout. If
        more keys are needed, this function should be overwritten.

        Args:
            rng: The current PRNGKey.

        Returns:
            Dict of PRNG Keys.
        """
        param_rng, dropout_rng = random.split(rng)
        return {"params": param_rng, "dropout": dropout_rng}

    def run_model_init(self, exmp_input: Batch, init_rng: jax.Array) -> dict:
        """The model initialization call.

        Args:
            exmp_input: An input to the model with which the shapes are inferred.
            init_rng: A jax.random.PRNGKey.

        Returns:
            The initialized variable dictionary.
        """
        rngs = self.get_model_rng(init_rng)
        exmp_input = self.batch_to_input(exmp_input)
        # TODO: Discuss which default structure we want, i.e. `train` as argument or `deterministic`.
        variables = self.model.init(rngs, exmp_input, train=False)
        if not isinstance(variables, FrozenDict):
            variables = freeze(variables)
        return variables

    def tabulate_params(self) -> str:
        """Return a string summary of the parameters represented as table.

        Returns:
            A string representation of the parameters.
        """
        return tabulate_params(self.state)

    def create_jitted_functions(self):
        """Create jitted versions of the training and evaluation functions.

        If self.trainer_config.debug is True, not jitting is applied.
        """
        train_step = self.create_training_step_function()
        eval_step = self.create_evaluation_step_function()
        if self.trainer_config.debug:  # Skip jitting
            logging.info("Skipping jitting due to debug=True")
            self.train_step = train_step
            self.eval_step = eval_step
        else:  # Jit
            train_donate_argnames = ["metrics"]  # Donate metrics to avoid copying.
            if self.trainer_config.donate_train_state:
                train_donate_argnames.append("state")
            self.train_step = jax.jit(
                train_step,
                donate_argnames=train_donate_argnames,
            )
            self.eval_step = jax.jit(
                eval_step,
                donate_argnames=["metrics"],  # Donate metrics to avoid copying.
            )

    def loss_function(
        self, params: Any, apply_fn: Any, batch: Batch, rng: jax.Array, train: bool = True
    ) -> tuple[jax.Array, Metrics]:
        """The loss function that is used for training.

        This function needs to be overwritten by a subclass.

        Args:
            params: The model parameters.
            apply_fn: The apply function of the state.
            batch: The current batch.
            rng: The random number generator.
            train: Whether the model is in training mode.

        Returns:
            The loss and a dictionary of metrics.
        """
        del params, apply_fn, batch, rng, train
        raise NotImplementedError
        # return loss, metrics

    def create_training_step_function(
        self,
    ) -> Callable[[TrainState, Batch, ImmutableMetrics | None], tuple[TrainState, ImmutableMetrics]]:
        """Create and return a function for the training step.

        The function takes as input the training state and a batch from the train loader. The
        function is expected to return a dictionary of logging metrics, and a new train state.
        """

        def train_step(
            state: TrainState, batch: Batch, metrics: ImmutableMetrics | None
        ) -> tuple[TrainState, ImmutableMetrics]:
            next_rng, step_rng = jax.random.split(state.rng)
            grads, step_metrics = accumulate_gradients(
                state,
                batch,
                step_rng,
                self.trainer_config.gradient_accumulate_steps,
                loss_fn=partial(self.loss_function, train=True),
            )
            # Update parameters. We need to sync the gradients across devices before updating.
            with jax.named_scope("sync_gradients"):
                grads = sync_gradients(
                    grads, (self.data_axis_name, self.fsdp_axis_name, self.pipeline_axis_name, self.model_axis_name)
                )
            new_state = state.apply_gradients(grads=grads, rng=next_rng)
            # Sum metrics across replicas. Alternatively, we could keep the metrics separate
            # and only synchronize them before logging. For simplicity, we sum them here.
            with jax.named_scope("sync_metrics"):
                step_metrics = jax.tree.map(
                    lambda x: jax.lax.psum(
                        x,
                        axis_name=(
                            self.data_axis_name,
                            self.fsdp_axis_name,
                            self.pipeline_axis_name,
                            self.model_axis_name,
                        ),
                    ),
                    step_metrics,
                )
            metrics = update_metrics(metrics, step_metrics, train=True)
            return new_state, metrics

        state_specs = nn.get_partition_spec(self.state)
        train_step_fn = shard_map(
            train_step,
            self.mesh,
            in_specs=(state_specs, P((self.data_axis_name, self.fsdp_axis_name)), P()),
            out_specs=(state_specs, P()),
            check_rep=False,
        )
        return train_step_fn

    def create_evaluation_step_function(
        self,
    ) -> Callable[[TrainState, Batch, ImmutableMetrics | None], ImmutableMetrics]:
        """Create and return a function for the evaluation step.

        The function takes as input the training state and a batch from the val/test loader. The
        function is expected to return a dictionary of logging metrics, and a new train state.
        """

        def eval_step(state: TrainState, batch: Batch, metrics: ImmutableMetrics | None) -> ImmutableMetrics:
            _, step_metrics = self.loss_function(
                state.params,
                state.apply_fn,
                batch,
                random.PRNGKey(self.trainer_config.seed_eval),
                train=False,
            )
            with jax.named_scope("sync_metrics"):
                step_metrics = jax.tree.map(
                    lambda x: jax.lax.psum(
                        x,
                        axis_name=(
                            self.data_axis_name,
                            self.fsdp_axis_name,
                            self.pipeline_axis_name,
                            self.model_axis_name,
                        ),
                    ),
                    step_metrics,
                )
            metrics = update_metrics(metrics, step_metrics, train=False)
            return metrics

        state_specs = nn.get_partition_spec(self.state)
        eval_step_fn = shard_map(
            eval_step,
            self.mesh,
            in_specs=(state_specs, P((self.data_axis_name, self.fsdp_axis_name)), P()),
            out_specs=P(),
            check_rep=False,
        )
        return eval_step_fn

    def train_model(
        self,
        train_loader: Iterator,
        val_loader: Iterator,
        test_loader: Iterator | None = None,
        num_epochs: int = 500,
    ) -> dict[str, Any]:
        """Start a training loop for the given number of epochs.

        Args:
            train_loader: Data loader of the training set.
            val_loader: Data loader of the validation set.
            test_loader: If given, best model will be evaluated on the test set.
            num_epochs: Number of epochs for which to train the model.

        Returns:
            A dictionary of the train, validation and evt. test metrics for the
            best model on the validation set.
        """
        self.global_step = jax.device_get(self.state.step).item()
        # Prepare training loop
        self.on_training_start()
        self.test_eval_function(val_loader)
        all_eval_metrics = {}
        train_metrics = None
        for epoch_idx in self.tracker(range(1, num_epochs + 1), desc="Epochs"):
            self.on_training_epoch_start(epoch_idx)
            train_metrics, epoch_metrics = self.train_epoch(
                train_loader, epoch_idx=epoch_idx, train_metrics=train_metrics
            )
            self.on_training_epoch_end(epoch_metrics, epoch_idx)
            # Validation every N epochs
            if (
                self.trainer_config.check_val_every_n_epoch > 0
                and epoch_idx % self.trainer_config.check_val_every_n_epoch == 0
            ):
                self.on_validation_epoch_start(epoch_idx)
                eval_metrics = self.eval_model(val_loader, mode="val", epoch_idx=epoch_idx)
                _, all_eval_metrics[f"val_{epoch_idx}"] = get_metrics(eval_metrics, reset_metrics=False)
                self.on_validation_epoch_end(eval_metrics, epoch_idx)
        self.on_training_end()
        # Test best model if possible
        if test_loader is not None:
            self.load_model(raise_if_not_found=False)
            self.on_test_epoch_start(epoch_idx)
            test_metrics = self.eval_model(test_loader, mode="test", epoch_idx=epoch_idx)
            _, all_eval_metrics["test"] = get_metrics(test_metrics, reset_metrics=False)
            self.on_test_epoch_end(test_metrics, epoch_idx)
        # Close logger
        self.logger.finalize("success")
        for callback in self.callbacks:
            callback.finalize("success")
        return all_eval_metrics

    def test_model(self, test_loader: Iterator, apply_callbacks: bool = False, epoch_idx: int = 0) -> dict[str, Any]:
        """Tests the model on the given test set.

        Args:
            test_loader: Data loader of the test set.
            apply_callbacks: If True, the callbacks will be applied.
            epoch_idx: The epoch index to use for the callbacks and logging.
        """
        test_metrics = self.eval_model(test_loader, mode="test", epoch_idx=epoch_idx)
        if apply_callbacks:
            self.on_test_epoch_end(test_metrics, epoch_idx=epoch_idx)
        return test_metrics

    def test_eval_function(self, val_loader: Iterator) -> None:
        """Test the evaluation function on a single batch.

        This is useful to check if the functions have the correct signature and return the correct
        values. This prevents annoying errors that occur at the first evaluation step.

        This function does not test the training function anymore. This is because the training
        function is already executed in the first epoch and we change its jit signature to donate
        the train state and metrics. Thus, executing a training step requires updating the train
        state, which we would not want to do here. The compilation time is logged during the very
        first training step.

        Args:
            val_loader: Data loader of the validation set.
        """
        print("Verifying evaluation function...")
        val_batch = next(iter(val_loader))
        eval_metrics = self.init_eval_metrics(val_batch)
        start_time = time.time()
        logging.info("Testing and compiling eval_step...")
        _ = self.eval_step(self.state, val_batch, eval_metrics)
        logging.info(f"Successfully completed in {time.time() - start_time:.2f} seconds.")

    def train_epoch(
        self, train_loader: Iterator, epoch_idx: int, train_metrics: ImmutableMetrics | None
    ) -> tuple[ImmutableMetrics, HostMetrics]:
        """Train a model for one epoch.

        Args:
            train_loader: Data loader of the training set.
            epoch_idx: Current epoch index.

        Returns:
            A dictionary of the average training metrics over all batches
            for logging.
        """
        # Train model for one epoch, and log avg loss and accuracy
        self.logger.start_epoch(epoch_idx, mode="train")
        for batch in self.tracker(train_loader, desc="Training", leave=False):
            if train_metrics is None:
                train_metrics = self.init_train_metrics(batch)
            if self.global_step == 0:
                # Log compilation and execution time of the first batch.
                logging.info("Compiling train_step...")
                start_time = time.time()
                self.state, train_metrics = self.train_step(self.state, batch, train_metrics)
                logging.info(
                    f"Successfully completed train_step compilation in {time.time() - start_time:.2f} seconds."
                )
            else:
                # Annotated with step number for TensorBoard profiling.
                with jax.profiler.StepTraceAnnotation(f"train_step_{self.global_step}"):
                    self.state, train_metrics = self.train_step(self.state, batch, train_metrics)
            for callback in self.callbacks:
                callback.on_training_step(train_metrics, epoch_idx, self.global_step)
            train_metrics = self.logger.log_step(train_metrics)
            self.global_step += 1
        train_metrics, epoch_metrics = self.logger.end_epoch(train_metrics)
        return train_metrics, epoch_metrics

    def eval_model(self, data_loader: Iterator, mode: str, epoch_idx: int) -> HostMetrics:
        """Evaluate the model on a dataset.

        Args:
            data_loader: Data loader of the dataset to evaluate on.
            mode: Whether 'val' or 'test'
            epoch_idx: Current epoch index.

        Returns:
            A dictionary of the evaluation metrics, averaged over data points
            in the dataset.
        """
        # Test model on all images of a data loader and return avg loss
        self.logger.start_epoch(epoch_idx, mode=mode)
        eval_metrics = self.init_eval_metrics()
        step_count = 0
        for batch in self.tracker(data_loader, desc=mode.capitalize(), leave=False):
            eval_metrics = self.eval_step(self.state, batch, eval_metrics)
            step_count += 1
        if step_count == 0:
            logging.warning(f"No batches in {mode} loader at epoch {epoch_idx}.")
        _, metrics = self.logger.end_epoch(eval_metrics)
        return metrics

    def tracker(self, iterator: Iterator, **kwargs) -> Iterator:
        """Wrap an iterator in a progress bar tracker (tqdm) if the progress bar is enabled.

        Args:
            iterator: Iterator to wrap in tqdm.
            kwargs: Additional arguments to tqdm.

        Returns:
            Wrapped iterator if progress bar is enabled, otherwise same iterator
            as input.
        """
        if self.trainer_config.enable_progress_bar and jax.process_index() == 0:
            return tqdm(iterator, **kwargs)
        else:
            return iterator

    def on_training_start(self):
        """Method called before training is started.

        Can be used for additional initialization operations etc.
        """
        logging.info("Starting training")
        for callback in self.callbacks:
            callback.on_training_start()

    def on_training_end(self):
        """Method called after training has finished.

        Can be used for additional logging or similar.
        """
        logging.info("Finished training")
        for callback in self.callbacks:
            callback.on_training_end()

    def on_training_epoch_start(self, epoch_idx: int):
        """Method called at the start of each training epoch. Can be used for additional logging or
        similar.

        Args:
            epoch_idx: Index of the training epoch that has started.
        """
        logging.info(f"Starting training epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_training_epoch_start(epoch_idx)

    def on_training_epoch_end(self, train_metrics: dict[str, Any], epoch_idx: int):
        """Method called at the end of each training epoch. Can be used for additional logging or
        similar.

        Args:
            epoch_idx: Index of the training epoch that has finished.
        """
        logging.info(f"Finished training epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_training_epoch_end(train_metrics, epoch_idx)

    def on_validation_epoch_start(self, epoch_idx: int):
        """Method called at the start of each validation epoch. Can be used for additional logging
        or similar.

        Args:
            epoch_idx: Index of the training epoch at which validation was started.
        """
        logging.info(f"Starting validation epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_validation_epoch_start(epoch_idx)

    def on_validation_epoch_end(self, eval_metrics: dict[str, Any], epoch_idx: int):
        """Method called at the end of each validation epoch. Can be used for additional logging
        and evaluation.

        Args:
            epoch_idx: Index of the training epoch at which validation was performed.
            eval_metrics: A dictionary of the validation metrics. New metrics added to
                this dictionary will be logged as well.
            val_loader: Data loader of the validation set, to support additional
                evaluation.
        """
        logging.info(f"Finished validation epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_validation_epoch_end(eval_metrics, epoch_idx)

    def on_test_epoch_start(self, epoch_idx: int):
        """Method called at the start of each test epoch. Can be used for additional logging or
        similar.

        Args:
            epoch_idx: Index of the training epoch at which testing was started.
        """
        logging.info(f"Starting test epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_test_epoch_start(epoch_idx)

    def on_test_epoch_end(self, test_metrics: dict[str, Any], epoch_idx: int):
        """Method called at the end of each test epoch. Can be used for additional logging and
        evaluation.

        Args:
            epoch_idx: Index of the training epoch at which testing was performed.
            test_metrics: A dictionary of the test metrics. New metrics added to
                this dictionary will be logged as well.
            test_loader: Data loader of the test set, to support additional
                evaluation.
        """
        logging.info(f"Finished test epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_test_epoch_end(test_metrics, epoch_idx)

    def load_model(self, epoch_idx: int = -1, raise_if_not_found: bool = True):
        """Load model parameters and batch statistics from the logging directory."""
        logging.info(f"Loading model from epoch {epoch_idx}")
        state_dict = None
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                state_dict = callback.load_model(epoch_idx)
                break
        if state_dict is None:
            if raise_if_not_found:
                raise ValueError("No model checkpoint callback found in callbacks.")
            else:
                logging.warning("No model checkpoint callback found in callbacks.")
        else:
            self.restore(state_dict)

    def restore(self, state_dict: dict[str, Any]):
        """Restore the state of the trainer from a state dictionary.

        Args:
            state_dict: State dictionary to restore from.
        """
        logging.info("Restoring trainer state with keys " + str(state_dict.keys()))
        if "metrics" in state_dict:
            state_dict.pop("metrics")
        if "metadata" in state_dict:
            state_dict.pop("metadata")
        state_dict = freeze(state_dict)
        self.state = TrainState(
            step=state_dict.get("step", 0),
            apply_fn=self.model.apply,
            params=state_dict["params"],
            tx=self.state.tx if self.state.tx else self.init_optimizer(self.optimizer_config),
            opt_state=state_dict.get("opt_state", None),
            mutable_variables=state_dict.get("mutable_variables", None),
            rng=state_dict.get("rng", self.state.rng),
        )

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint: str,
        exmp_input: Batch = None,
        exmp_input_file: str = None,
        batch_size: int = -1,
    ) -> Any:
        """Create a Trainer object with same hyperparameters and loaded model from a checkpoint
        directory.

        Args:
            checkpoint: Folder in which the checkpoint and hyperparameter file is stored.
            exmp_input: An input to the model for shape inference.

        Returns:
            A Trainer object with model loaded from the checkpoint folder.
        """
        # Load config
        metadata_file = os.path.join(checkpoint, "metadata/metadata")
        assert os.path.isfile(metadata_file), "Could not find metadata file"
        with open(metadata_file, "rb") as f:
            config = ConfigDict(json.load(f))
        # Adjust log dir to where its loaded from
        adjusted_checkpoint = checkpoint.split("/")
        if adjusted_checkpoint[-1] == "":
            adjusted_checkpoint = adjusted_checkpoint[:-1]
        if len(adjusted_checkpoint) < 2:
            raise ValueError("Checkpoint path must be at least two levels deep")
        config.trainer.logger.log_dir = os.path.join(*adjusted_checkpoint[:-2])
        # Load example input
        # TODO: We may want to load the example input from the checkpoint folder.
        assert exmp_input is not None, "Example input must be provided"
        if batch_size > 0:
            exmp_input = exmp_input[:batch_size]
        # Create trainer
        trainer = cls(
            exmp_input=exmp_input,
            trainer_config=config.trainer,
            model_config=config.model,
            optimizer_config=config.optimizer,
        )
        # Load model
        trainer.load_model()
        return trainer
