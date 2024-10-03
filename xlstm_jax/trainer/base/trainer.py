import json
import logging
import os
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict, freeze
from jax import random
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
from tqdm.auto import tqdm

from xlstm_jax.configs import ConfigDict
from xlstm_jax.dataset import Batch
from xlstm_jax.distributed import accumulate_gradients, sync_gradients
from xlstm_jax.distributed.common_types import PRNGKeyArray
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models import ModelConfig
from xlstm_jax.trainer.callbacks import CallbackConfig, ModelCheckpoint, load_pretrained_model
from xlstm_jax.trainer.data_module import DataloaderModule
from xlstm_jax.trainer.logger import Logger, LoggerConfig
from xlstm_jax.trainer.metrics import HostMetrics, ImmutableMetrics, Metrics, update_metrics
from xlstm_jax.trainer.optimizer import OptimizerConfig, build_optimizer

from .param_utils import flatten_dict, get_grad_norms, get_num_params, get_param_norms, tabulate_params
from .train_state import TrainState

LOGGER = logging.getLogger(__name__)

PyTree = Any


@dataclass(kw_only=True, frozen=True)
class TrainerConfig(ConfigDict):
    """Configuration for the Trainer module."""

    seed: int = 0
    """Random seed for reproducibility. To be used in the model init and training step."""
    debug: bool = False
    """Whether to run in debug mode. This disables jitting of the training and evaluation functions, which will slow
    down the training significantly but makes debugging easier."""
    donate_train_state: bool = True
    """Whether to donate the train state in the training step. This can reduce memory usage as the parameters and
    optimizer states are in-place updated in the training step. However, this prevents using the previous train state
    after calling the training step (not used in Trainer, but keep in mind for custom training loops and callbacks)."""
    enable_progress_bar: bool = True
    """Whether to enable the progress bar. For multiprocess training, only the main process will show the progress bar.
    """
    gradient_accumulate_steps: int = 1
    """Number of steps to accumulate gradients before updating the parameters."""
    gradient_accumulate_scan: bool = False
    """Whether to use scan for gradient accumulation. This can be more memory efficient and significantly faster to
    compile for large models, but can be slighlty slower due to memory slicing."""
    check_val_every_n_epoch: int = 1
    """Check validation every N training epochs. If -1, no validation is performed after an epoch. Note that this is
    not mutually exclusive with check_val_every_n_steps, and both can be used."""
    check_val_every_n_steps: int = -1
    """Check validation every N training steps. If -1, no validation is performed on a per-step basis. Note that this
    is not mutually exclusive with check_val_every_n_epoch, and both can be used."""
    check_for_nan: bool = True
    """Whether to check for NaN values in the loss during training. If NaNs are found, training will be stopped."""
    log_grad_norm: bool = True
    """Whether to log the gradient norm."""
    log_grad_norm_per_param: bool = False
    """Whether to log the gradient norm per parameter. If the model has many parameters, this can lead to a large log
    file."""
    log_param_norm: bool = True
    """Whether to log the parameter norm."""
    log_param_norm_per_param: bool = False
    """Whether to log the parameter norm per parameter. If the model has many parameters, this can lead to a large log
    file."""
    log_intermediates: bool = False
    """Whether to log intermediate values during training. This is useful for debugging, but can lead to a large log
    file and a bit of overhead during training, if intermediates are complex to compute. Intermediates can be recorded
    by using the ``self.sow("intermediates", "KEY", VALUE)`` method in the model. The intermediate values are
    automatically registered and logged. Note that the values should be scalars."""
    default_train_log_modes: Sequence[str] = ("mean",)
    """Default logging modes for training metrics. Can be `mean`, `mean_nopostfix`, `single`, `max`, or `std`. See
    metrics for more information. Each selected mode will be logged with the corresponding postfix. During validation,
    we only log the `mean` of the metrics."""
    intermediates_log_modes: Sequence[str] = ("mean",)
    """Logging modes for intermediate values. See `default_train_log_modes` for more information."""
    logger: LoggerConfig = LoggerConfig()
    """Configuration for the logger."""
    callbacks: Sequence[CallbackConfig] = ()
    """List of callbacks to apply."""
    seed_eval: int = 0
    """Random seed for evaluation, if the model uses randomness during evaluation. This is useful to ensure
    reproducibility of evaluation metrics."""


class TrainerModule:
    """
    A basic Trainer module summarizing most common training functionalities like logging, model initialization, training
    loop, etc..

    Args:
        trainer_config: A dictionary containing the trainer configuration.
        model_config: A dictionary containing the model configuration.
        optimizer_config: A dictionary containing the optimizer configuration.
        batch: An input to the model with which the shapes are inferred. Can be a :class:`jax.ShapeDtypeStruct` instead
            of actual full arrays for efficiency.
        mesh: A mesh object to use for parallel training. If `None`, a new mesh will be created.
    """

    def __init__(
        self,
        trainer_config: TrainerConfig,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        batch: Batch,
        mesh: Mesh | None = None,
    ):
        super().__init__()
        self.trainer_config = trainer_config
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.exmp_batch = batch
        # Init logger first
        self.init_logger(self.trainer_config.logger)
        # Setup parallel mesh
        self.init_mesh(model_config, mesh)
        # Create empty model. Note: no parameters yet
        self.build_model(model_config)
        # Init trainer parts
        self.init_optimizer(optimizer_config)
        self.init_model(batch)
        self.create_jitted_functions()
        self.init_callbacks(self.trainer_config.callbacks)
        # Set first step to True to log compilation time of the first step.
        self.first_step = True
        self.global_step = 0

    def batch_to_input(self, batch: Batch) -> Any:
        """
        Convert a batch to the input format expected by the model.

        Needs to be implemented by the subclass if `batch.inputs` is not sufficient.

        Args:
            batch: A batch of data.

        Returns:
            The input to the model.
        """
        return batch.inputs

    def init_mesh(self, model_config: ConfigDict, mesh: Mesh | None = None):
        """
        Initialize the mesh for parallel training if no mesh is supplied.

        Args:
            model_config: A dictionary containing the model configuration, including the parallelization parameters.
            mesh: A mesh object to use for parallel training. If `None`, a new mesh is created.
        """
        if mesh is None:
            self.mesh = initialize_mesh(parallel_config=model_config.parallel)
        else:
            self.mesh = mesh

        # Save axis names to trainer for easier usage.
        self.data_axis_name = model_config.parallel.data_axis_name
        self.fsdp_axis_name = model_config.parallel.fsdp_axis_name
        self.pipeline_axis_name = model_config.parallel.pipeline_axis_name
        self.model_axis_name = model_config.parallel.model_axis_name

        # Create batch specs for sharding.
        self.batch_partition_specs = P(self.mesh.axis_names)
        if jax.process_index() == 0:
            LOGGER.info(f"Initialized mesh with {self.mesh}.")

    def build_model(self, model_config: ConfigDict):
        """
        Create the model class from the model_config.

        Args:
            model_config: A dictionary containing the model configuration.
        """
        self.model: nn.Module = model_config.model_class(
            model_config if model_config.model_config is None else model_config.model_config
        )

    def init_logger(self, logger_config: ConfigDict):
        """
        Initialize a logger and creates a logging directory.

        Args:
            logger_params: A dictionary containing the specification of the logger.
        """
        self.logger: Logger = Logger(logger_config, metric_postprocess_fn=self.get_metric_postprocess_fn())
        self.logger.log_config(
            {"trainer": self.trainer_config, "model": self.model_config, "optimizer": self.optimizer_config}
        )
        self.log_path = self.logger.log_path

    def get_metric_postprocess_fn(self) -> Callable[[HostMetrics], HostMetrics]:
        """
        Get function to post-process metrics with on host.

        Will be passed to logger. Default implementation returns the identity function.
        Can be overwritten by subclasses.

        Returns:
            Callable[[HostMetrics], HostMetrics]: The postprocess metric function.
        """
        return lambda x: x

    def init_callbacks(self, callback_configs: Sequence[CallbackConfig]):
        """Initialize the callbacks defined in the trainer config."""
        self.callbacks = []
        for cb_config in callback_configs:
            LOGGER.info(f"Initializing callback {cb_config.__class__.__name__}")
            callback = cb_config.create(trainer=self, data_module=None)
            self.callbacks.append(callback)

    def init_optimizer(self, optimizer_config: ConfigDict):
        """
        Initialize the optimizer.

        Args:
            optimizer_config: A dictionary containing the optimizer configuration.
        """
        self.optimizer, self.lr_scheduler = build_optimizer(optimizer_config)

    def init_model(self, exmp_input: Batch):
        """
        Create an initial training state with newly generated network parameters.

        This function is parallelized over the mesh to initialize the per-device parameters. It also initializes the
        optimizer parameters. As a result, it sets the training state of the trainer with the initialized parameters.

        Args:
            exmp_input: An input to the model with which the shapes are inferred.
        """

        def _init_model(init_rng: PRNGKeyArray, batch: Batch) -> TrainState:
            param_rng, init_rng = jax.random.split(init_rng)
            # Initialize parameters.
            variables = self.run_model_init(batch, param_rng)
            assert isinstance(variables, FrozenDict), "Model init must return a FrozenDict."
            mutable_variables, params = variables.pop("params")
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
                in_specs=(P(), self.batch_partition_specs),
                out_specs=P(),
                check_rep=False,
            ),
        )
        state_shapes = jax.eval_shape(init_model_fn, init_rng, exmp_input)
        state_partition_specs = nn.get_partition_spec(state_shapes)
        # Run init model function again with correct output specs.
        init_model_fn = jax.jit(
            shard_map(
                _init_model,
                self.mesh,
                in_specs=(P(), self.batch_partition_specs),
                out_specs=state_partition_specs,
                check_rep=False,
            ),
        )
        self.state = init_model_fn(init_rng, exmp_input)
        LOGGER.info("Model initialized.")
        LOGGER.info(
            tabulate_params(
                self.state,
                show_weight_decay=self.optimizer_config.weight_decay > 0,
                weight_decay_include=self.optimizer_config.weight_decay_include,
                weight_decay_exclude=self.optimizer_config.weight_decay_exclude,
            )
        )

    def init_train_metrics(self, batch: Batch | None = None) -> FrozenDict:
        """
        Initialize the training metrics with zeros.

        We infer the training metric shape from the train_step function. This is done to prevent a double-compilation of
        the train_step function, where the first step has to be done with metrics None, and the next one with the
        metrics shape.

        Args:
            batch: An input to the model with which the shapes are inferred. If None, the :attr:`exmp_batch` is used.

        Returns:
            A dictionary of metrics with the same shape as the train metrics.
        """
        if not hasattr(self, "train_metric_shapes"):
            self.train_metric_shapes = None
        if self.train_metric_shapes is None:
            if batch is None:
                batch = self.exmp_batch
            _, self.train_metric_shapes = jax.eval_shape(self.train_step, self.state, batch, None)
            LOGGER.info(f"Initialized train metrics with keys {self.train_metric_shapes.keys()}.")
        return jax.tree.map(lambda x: jnp.zeros_like(x), self.train_metric_shapes)

    def init_eval_metrics(self, batch: Batch | None = None) -> FrozenDict:
        """
        Initialize the evaluation metrics with zeros.

        See init_train_metrics for more details.

        Args:
            batch: An input to the model with which the shapes are inferred. If None, the :attr:`exmp_batch` is used.

        Returns:
            A dictionary of metrics with the same shape as the eval metrics.
        """
        if not hasattr(self, "eval_metric_shapes"):
            self.eval_metric_shapes = None
        if self.eval_metric_shapes is None:
            if batch is None:
                batch = self.exmp_batch
            self.eval_metric_shapes = jax.eval_shape(self.eval_step, self.state, batch, None)
            LOGGER.info(f"Initialized eval metrics with keys {self.eval_metric_shapes.keys()}.")
        return jax.tree.map(lambda x: jnp.zeros_like(x), self.eval_metric_shapes)

    def set_dataset(self, dataset: Any):
        """
        Set the dataset for the trainer and the callbacks.

        Args:
            dataset: The dataset to set.
        """
        for callback in self.callbacks:
            callback.set_dataset(dataset)
        self.dataset = dataset

    def get_model_rng(self, rng: jax.Array) -> dict[str, random.PRNGKey]:
        """
        Return a dictionary of PRNGKey for init and tabulate.

        By default, adds a key for the parameters and one for dropout. If more keys are needed, this function should be
        overwritten.

        Args:
            rng: The current PRNGKey.

        Returns:
            Dict of PRNG Keys.
        """
        param_rng, dropout_rng = random.split(rng)
        return {"params": param_rng, "dropout": dropout_rng}

    def run_model_init(self, exmp_input: Batch, init_rng: jax.Array) -> FrozenDict:
        """
        The model initialization call.

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
        """
        Return a string summary of the parameters represented as table.

        Returns:
            A string representation of the parameters.
        """
        return tabulate_params(self.state)

    def create_jitted_functions(self):
        """
        Create jitted versions of the training and evaluation functions.

        If self.trainer_config.debug is True, not jitting is applied.
        """
        train_step = self.create_training_step_function()
        eval_step = self.create_evaluation_step_function()
        if self.trainer_config.debug:  # Skip jitting
            LOGGER.info("Skipping jitting due to debug=True")
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
    ) -> tuple[jax.Array, tuple[Metrics, PyTree]]:
        """
        The loss function that is used for training.

        This function needs to be overwritten by a subclass.

        Args:
            params: The model parameters.
            apply_fn: The apply function of the state.
            batch: The current batch.
            rng: The random number generator.
            train: Whether the model is in training mode.

        Returns:
            The loss and a tuple of metrics and mutable variables.
        """
        del params, apply_fn, batch, rng, train
        raise NotImplementedError
        # return loss, metrics

    def create_training_step_function(
        self,
    ) -> Callable[[TrainState, Batch, ImmutableMetrics | None], tuple[TrainState, ImmutableMetrics]]:
        """
        Create and return a function for the training step.

        The function takes as input the training state and a batch from the train loader. The function is expected to
        return a dictionary of logging metrics, and a new train state.
        """

        def train_step(
            state: TrainState, batch: Batch, metrics: ImmutableMetrics | None
        ) -> tuple[TrainState, ImmutableMetrics]:
            # In our multi-host setup, each local device will have a different batch.
            # So we first gather the batch across model and pipeline axes.
            batch = jax.lax.all_gather(
                batch, axis_name=(self.model_axis_name, self.pipeline_axis_name), axis=0, tiled=True
            )
            # Split the random key for the current step.
            next_rng, step_rng = jax.random.split(state.rng)
            # Forward and backward with gradient accumulation.
            grads, step_metrics, mutable_variables = accumulate_gradients(
                state,
                batch,
                step_rng,
                self.trainer_config.gradient_accumulate_steps,
                loss_fn=partial(self.loss_function, train=True),
                use_scan=self.trainer_config.gradient_accumulate_scan,
            )
            # If we have intermediates in mutable variables, pop them and add to metrics.
            if mutable_variables is not None and "intermediates" in mutable_variables:
                if not isinstance(mutable_variables, FrozenDict):
                    mutable_variables = freeze(mutable_variables)
                mutable_variables, intermediates = mutable_variables.pop("intermediates")
                intermediates = flatten_dict(intermediates, flatten_sequences=True)
                inter_keys = list(intermediates.keys())
                for key in inter_keys:
                    if intermediates[key].ndim == 1 or (intermediates[key].ndim == 2 and 1 in intermediates[key].shape):
                        # For scanned intermediates, we need to flatten them and add them individually.
                        val = intermediates.pop(key)
                        val = val.reshape(-1)
                        for i in range(val.shape[0]):
                            intermediates[f"{key}_{i}"] = val[i]
                    else:
                        # For larger intermediates, we take the mean.
                        intermediates[key] = intermediates[key].mean()
                assert all(v.size == 1 for v in intermediates.values()), "Only scalar intermediates supported."
                # Add intermediates to metrics.
                if self.trainer_config.log_intermediates:
                    intermediate_metrics = {
                        k: {"value": v, "count": 1, "log_modes": self.trainer_config.intermediates_log_modes}
                        for k, v in intermediates.items()
                    }
                    step_metrics.update(intermediate_metrics)
            # If no mutable variables, set to None for state.
            if mutable_variables is not None and len(mutable_variables) == 0:
                mutable_variables = None
            # Update parameters. We need to sync the gradients across devices before updating.
            with jax.named_scope("sync_gradients"):
                grads = sync_gradients(
                    grads, (self.data_axis_name, self.fsdp_axis_name, self.pipeline_axis_name, self.model_axis_name)
                )
            new_state = state.apply_gradients(grads=grads, rng=next_rng, mutable_variables=mutable_variables)
            # Sum metrics across replicas. Communication negligible and can be done async to backward.
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
                    )
                    if not isinstance(x, str)
                    else x,
                    step_metrics,
                )

            # Add logging of gradient norm.
            if self.trainer_config.log_grad_norm:
                step_metrics.update(
                    get_grad_norms(grads=grads, return_per_param=self.trainer_config.log_grad_norm_per_param)
                )
            # Add logging of parameter norm.
            if self.trainer_config.log_param_norm:
                param_norms = get_param_norms(
                    params=new_state.params, return_per_param=self.trainer_config.log_param_norm_per_param
                )
                # We only log the mean of the parameter norms, as they don't change quickly during training and
                # logging all of them would be too verbose.
                param_norms = {
                    key: {"value": value, "log_modes": ("mean_nopostfix",)} for key, value in param_norms.items()
                }
                step_metrics.update(param_norms)

            # Update global training metrics.
            metrics = update_metrics(
                metrics, step_metrics, default_log_modes=self.trainer_config.default_train_log_modes
            )
            return new_state, metrics

        # Shard the training function.
        state_partition_specs = nn.get_partition_spec(self.state)
        train_step_fn = shard_map(
            train_step,
            self.mesh,
            in_specs=(state_partition_specs, self.batch_partition_specs, P()),
            out_specs=(state_partition_specs, P()),
            check_rep=False,
        )
        return train_step_fn

    def create_evaluation_step_function(
        self,
    ) -> Callable[[TrainState, Batch, ImmutableMetrics | None], ImmutableMetrics]:
        """
        Create and return a function for the evaluation step.

        The function takes as input the training state and a batch from the val/test loader. The function is expected to
        return a dictionary of logging metrics, and a new train state.
        """

        def eval_step(state: TrainState, batch: Batch, metrics: ImmutableMetrics | None) -> ImmutableMetrics:
            # In our multi-host setup, each local device will have a different batch.
            # So we first gather the batch across model and pipeline axes.
            batch = jax.lax.all_gather(
                batch, axis_name=(self.model_axis_name, self.pipeline_axis_name), axis=0, tiled=True
            )
            # Forward pass and compute metrics.
            _, (step_metrics, _) = self.loss_function(
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
                    )
                    if not isinstance(x, str)
                    else x,
                    step_metrics,
                )
            metrics = update_metrics(metrics, step_metrics, default_log_modes=("mean_nopostfix",))
            return metrics

        # Shard the evaluation function.
        state_partition_specs = nn.get_partition_spec(self.state)
        eval_step_fn = shard_map(
            eval_step,
            self.mesh,
            in_specs=(state_partition_specs, self.batch_partition_specs, P()),
            out_specs=P(),
            check_rep=False,
        )
        return eval_step_fn

    def train_model(
        self,
        train_loader: Iterator,
        val_loader: Iterator,
        test_loader: Iterator | None = None,
        num_epochs: int | None = None,
        num_train_steps: int | None = None,
        steps_per_epoch: int | None = None,
    ) -> dict[str, Any]:
        """
        Start a training loop for the given number of epochs.

        Inside the training loop, we use an epoch index and a global step index. Both indices are starting to count at 1
        (i.e. first epoch is "epoch 1", not "epoch 0").

        Args:
            train_loader: Data loader of the training set.
            val_loader: Data loader of the validation set.
            test_loader: If given, best model will be evaluated on the test set.
            num_epochs: Number of epochs for which to train the model. If None, will use num_train_steps.
            num_train_steps: Number of training steps for which to train the model. If None, will use num_epochs.
            steps_per_epoch: Number of steps per epoch. If None, will use the length of the train_loader.

        Returns:
            A dictionary of the train, validation and evt. test metrics for the
            best model on the validation set.
        """
        # Verify input arguments.
        self.global_step = jax.device_get(self.state.step).item()
        if num_epochs is not None and num_train_steps is not None:
            raise ValueError("Only one of num_epochs and num_train_steps can be set.")
        if num_epochs is None and num_train_steps is None:
            raise ValueError("Either num_epochs or num_train_steps must be set.")
        if steps_per_epoch is None and hasattr(train_loader, "__len__"):
            steps_per_epoch = len(train_loader)
        if num_epochs is not None:
            assert (
                steps_per_epoch is not None
            ), "train_loader must have a __len__ method or specify the steps_per_epoch if num_epochs is set."
            num_train_steps = steps_per_epoch * num_epochs

        # Prepare training loop.
        self.logger.on_training_start()
        self.log_training_info(num_epochs, num_train_steps, steps_per_epoch, train_loader, val_loader, test_loader)
        self.on_training_start()
        self.test_eval_function(val_loader)
        all_eval_metrics = {}
        train_metrics = None
        epoch_idx = 0

        # Share data loaders with callbacks.
        data_module = DataloaderModule(
            train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=test_loader
        )
        for callback in self.callbacks:
            callback.set_dataset(data_module)

        # Main training loop.
        stop_training_with_error = False
        while self.global_step < num_train_steps and not stop_training_with_error:
            if steps_per_epoch:
                epoch_idx = self.global_step // steps_per_epoch + 1
            else:
                LOGGER.warning(
                    "Steps per epoch could not be inferred by the training loader. Epoch index will be inferred by "
                    "breaks of iterator, but likely incorrect if you loaded a pre-trained model."
                )
                epoch_idx = epoch_idx + 1
            self.on_training_epoch_start(epoch_idx)
            self.logger.start_epoch(epoch=epoch_idx, step=self.global_step, mode="train")

            # Train epoch loop.
            for batch in self.tracker(train_loader, desc="Training", leave=False):
                self.global_step += 1
                if train_metrics is None:
                    train_metrics = self.init_train_metrics(batch)

                if self.first_step:
                    # Log compilation and execution time of the first batch.
                    LOGGER.info("Compiling train_step...")
                    start_time = time.time()
                    self.state, train_metrics = self.train_step(self.state, batch, train_metrics)
                    LOGGER.info(
                        f"Successfully completed train_step compilation in {time.time() - start_time:.2f} seconds."
                    )
                    self.first_step = False
                else:
                    # Annotated with step number for TensorBoard profiling.
                    with jax.profiler.StepTraceAnnotation(f"train_step_{self.global_step}"):
                        self.state, train_metrics = self.train_step(self.state, batch, train_metrics)

                # Callbacks and logging.
                for callback in self.callbacks:
                    callback.on_training_step(train_metrics, epoch_idx, self.global_step)
                train_metrics = self.logger.log_step(train_metrics, step=self.global_step)

                # Validation every N steps.
                if (
                    self.trainer_config.check_val_every_n_steps > 0
                    and self.global_step % self.trainer_config.check_val_every_n_steps == 0
                    and not (self.trainer_config.check_for_nan and self.logger.found_nans)
                ):
                    all_eval_metrics[f"val_step_{self.global_step}"] = self._eval_model_in_train_loop(
                        val_loader, epoch_idx
                    )

                # Check for NaN values.
                if self.trainer_config.check_for_nan and self.logger.found_nans:
                    stop_training_with_error = True
                    break

                # Stop training if we reached the desired number of steps.
                if self.global_step >= num_train_steps:
                    break

            # Finalize epoch.
            train_metrics, epoch_metrics = self.logger.end_epoch(train_metrics, step=self.global_step)
            self.on_training_epoch_end(epoch_metrics, epoch_idx)

            # Check for NaN values.
            if stop_training_with_error or (self.trainer_config.check_for_nan and self.logger.found_nans):
                stop_training_with_error = True
                break

            # Validation every N epochs.
            if (
                self.trainer_config.check_val_every_n_epoch > 0
                and epoch_idx % self.trainer_config.check_val_every_n_epoch == 0
                and not stop_training_with_error
            ):
                if f"val_step_{self.global_step}" in all_eval_metrics:
                    LOGGER.warning(
                        f"Skipping validation at epoch {epoch_idx} since already validated at step {self.global_step}."
                    )
                    all_eval_metrics[f"val_epoch_{epoch_idx}"] = all_eval_metrics[f"val_step_{self.global_step}"]
                else:
                    all_eval_metrics[f"val_epoch_{epoch_idx}"] = self._eval_model_in_train_loop(val_loader, epoch_idx)

        # Finalize training.
        self.on_training_end()

        # Test evaluation.
        if not stop_training_with_error and test_loader is not None:
            self.load_model(raise_if_not_found=False)
            self.on_test_epoch_start(epoch_idx)
            test_metrics = self.eval_model(test_loader, mode="test", epoch_idx=epoch_idx)
            self.on_test_epoch_end(test_metrics, epoch_idx)
            all_eval_metrics["test"] = test_metrics

        # Summarize status.
        if not stop_training_with_error:
            finalize_status = "success"
            LOGGER.info("Training finished successfully.")
        elif self.trainer_config.check_for_nan and self.logger.found_nans:
            finalize_status = "nan"
            LOGGER.error("Training stopped due to NaN values in the loss.")
        else:
            finalize_status = "error"
            LOGGER.error("Training stopped due to an error.")

        # Close logger and callbacks.
        self.logger.finalize(status=finalize_status)
        for callback in self.callbacks:
            callback.finalize(status=finalize_status)

        return all_eval_metrics

    def _eval_model_in_train_loop(self, val_loader: Iterator, epoch_idx: int) -> HostMetrics:
        """
        Evaluate the model on the validation set during the training loop.

        Args:
            val_loader: Data loader of the validation set.
            epoch_idx: Current epoch index.

        Returns:
            A dictionary of the evaluation metrics.
        """
        self.on_validation_epoch_start(epoch_idx, self.global_step)
        eval_metrics = self.eval_model(val_loader, mode="val", epoch_idx=epoch_idx)
        self.on_validation_epoch_end(eval_metrics, epoch_idx, self.global_step)
        return eval_metrics

    def test_model(self, test_loader: Iterator, apply_callbacks: bool = False, epoch_idx: int = 0) -> dict[str, Any]:
        """
        Tests the model on the given test set.

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
        """
        Test the evaluation function on a single batch.

        This is useful to check if the functions have the correct signature and return the correct values. This prevents
        annoying errors that occur at the first evaluation step.

        This function does not test the training function anymore. This is because the training function is already
        executed in the first epoch, and we change its jit signature to donate the train state and metrics. Thus,
        executing a training step requires updating the train state, which we would not want to do here. The compilation
        time is logged during the very first training step.

        Args:
            val_loader: Data loader of the validation set.
        """
        LOGGER.info("Verifying evaluation function...")
        val_batch = next(iter(val_loader))
        eval_metrics = self.init_eval_metrics(val_batch)
        start_time = time.time()
        LOGGER.info("Testing and compiling eval_step...")
        _ = self.eval_step(self.state, val_batch, eval_metrics)
        LOGGER.info(f"Successfully completed in {time.time() - start_time:.2f} seconds.")

    def eval_model(self, data_loader: Iterator, mode: str, epoch_idx: int) -> HostMetrics:
        """
        Evaluate the model on a dataset.

        Args:
            data_loader: Data loader of the dataset to evaluate on.
            mode: Whether 'val' or 'test'
            epoch_idx: Current epoch index.

        Returns:
            A dictionary of the evaluation metrics, averaged over data points in the dataset.
        """
        # Test model on all batches of a data loader and return avg loss
        self.logger.start_epoch(epoch=epoch_idx, step=self.global_step, mode=mode)
        eval_metrics = self.init_eval_metrics()
        step_count = 0
        for batch in self.tracker(data_loader, desc=mode.capitalize(), leave=False):
            eval_metrics = self.eval_step(self.state, batch, eval_metrics)
            step_count += 1
        if step_count == 0:
            LOGGER.warning(f"No batches in {mode} loader at epoch {epoch_idx}.")
        _, metrics = self.logger.end_epoch(eval_metrics, step=self.global_step)
        return metrics

    def tracker(self, iterator: Iterator, **kwargs) -> Iterator:
        """
        Wrap an iterator in a progress bar tracker (tqdm) if the progress bar is enabled.

        Args:
            iterator: Iterator to wrap in tqdm.
            kwargs: Additional arguments to tqdm.

        Returns:
            Wrapped iterator if progress bar is enabled, otherwise same iterator as input.
        """
        if self.trainer_config.enable_progress_bar and jax.process_index() == 0:
            return tqdm(iterator, **kwargs)
        else:
            return iterator

    def log_training_info(
        self,
        num_epochs: int | None,
        num_train_steps: int,
        steps_per_epoch: int | None,
        train_loader: Iterator,
        val_loader: Iterator,
        test_loader: Iterator | None,
    ):
        """
        Log the general training information.

        Args:
            num_epochs: Number of epochs for which to train the model.
            num_train_steps: Number of training steps for which to train the model.
            steps_per_epoch: Number of steps per epoch.
            train_loader: Data loader of the training set.
            val_loader: Data loader of the validation set.
            test_loader: Data loader of the test set.
        """
        # Log model metrics.
        model_metrics = {
            "num_params": get_num_params(self.state.params),
            "num_optimizer_params": get_num_params(self.state.opt_state),
            "start_step": jax.device_get(self.state.step).item(),
        }
        self.logger.log_host_metrics(model_metrics, step=self.global_step, mode="model")

        # Log dataset metrics.
        combined_dp_size = self.mesh.shape[self.data_axis_name] * self.mesh.shape[self.fsdp_axis_name]
        dataset_metrics = {
            "global_batch_size": self.exmp_batch.inputs.shape[0],
            "local_batch_size": self.exmp_batch.inputs.shape[0] // combined_dp_size,
            "train_steps_per_epoch": len(train_loader) if hasattr(train_loader, "__len__") else steps_per_epoch,
            "num_epochs": num_epochs,
            "num_train_steps": num_train_steps,
            "train_dataset_size": getattr(train_loader, "dataset_size", None),
            "val_dataset_size": getattr(val_loader, "dataset_size", None),
            "test_dataset_size": None if test_loader is None else getattr(test_loader, "dataset_size", None),
        }
        dataset_metrics = {k: v for k, v in dataset_metrics.items() if v is not None}
        self.logger.log_host_metrics(dataset_metrics, step=self.global_step, mode="dataset")

        # Log parallel configuration.
        parallel_metrics = {
            "dp_axis_size": self.mesh.shape[self.data_axis_name],
            "fsdp_axis_size": self.mesh.shape[self.fsdp_axis_name],
            "pipeline_axis_size": self.mesh.shape[self.pipeline_axis_name],
            "model_axis_size": self.mesh.shape[self.model_axis_name],
            "combined_dp_axis_size": combined_dp_size,
            "num_devices": jax.device_count(),
            "num_processes": jax.process_count(),
        }
        self.logger.log_host_metrics(parallel_metrics, step=self.global_step, mode="parallel")

    def on_training_start(self):
        """
        Method called before training is started.

        Can be used for additional initialization operations etc.
        """
        LOGGER.info("Starting training")
        for callback in self.callbacks:
            callback.on_training_start()

    def on_training_end(self):
        """
        Method called after training has finished.

        Can be used for additional logging or similar.
        """
        LOGGER.info("Finished training")
        for callback in self.callbacks:
            callback.on_training_end()

    def on_training_epoch_start(self, epoch_idx: int):
        """
        Method called at the start of each training epoch. Can be used for additional logging or similar.

        Args:
            epoch_idx: Index of the training epoch that has started.
        """
        LOGGER.info(f"Starting training epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_training_epoch_start(epoch_idx)

    def on_training_epoch_end(self, train_metrics: dict[str, Any], epoch_idx: int):
        """
        Method called at the end of each training epoch. Can be used for additional logging or similar.

        Args:
            epoch_idx: Index of the training epoch that has finished.
        """
        LOGGER.info(f"Finished training epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_training_epoch_end(train_metrics, epoch_idx)

    def on_validation_epoch_start(self, epoch_idx: int, step_idx: int):
        """
        Method called at the start of each validation epoch. Can be used for additional logging or similar.

        Args:
            epoch_idx: Index of the training epoch at which validation was started.
            step_idx: Index of the training step at which validation was started.
        """
        LOGGER.info(f"Starting validation at epoch {epoch_idx} and step {step_idx}")
        for callback in self.callbacks:
            callback.on_validation_epoch_start(epoch_idx=epoch_idx, step_idx=step_idx)

    def on_validation_epoch_end(self, eval_metrics: dict[str, Any], epoch_idx: int, step_idx: int):
        """
        Method called at the end of each validation epoch. Can be used for additional logging and evaluation.

        Args:
            eval_metrics: A dictionary of the validation metrics. New metrics added to this dictionary will be logged as
                well.
            epoch_idx: Index of the training epoch at which validation was performed.
            step_idx: Index of the training step at which validation was performed.
        """
        LOGGER.info(f"Finished validation epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_validation_epoch_end(eval_metrics, epoch_idx=epoch_idx, step_idx=step_idx)

    def on_test_epoch_start(self, epoch_idx: int):
        """
        Method called at the start of each test epoch. Can be used for additional logging or similar.

        Args:
            epoch_idx: Index of the training epoch at which testing was started.
        """
        LOGGER.info(f"Starting test epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_test_epoch_start(epoch_idx)

    def on_test_epoch_end(self, test_metrics: dict[str, Any], epoch_idx: int):
        """
        Method called at the end of each test epoch. Can be used for additional logging and evaluation.

        Args:
            epoch_idx: Index of the training epoch at which testing was performed.
            test_metrics: A dictionary of the test metrics. New metrics added to this dictionary will be logged as well.
        """
        LOGGER.info(f"Finished test epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_test_epoch_end(test_metrics, epoch_idx)

    def load_model(self, step_idx: int = -1, raise_if_not_found: bool = True):
        """
        Load model parameters and batch statistics from the logging directory.

        Args:
            step_idx: Step index to load the model from. If -1, the latest model is loaded.
            raise_if_not_found: If True, raises an error if no model is found. If False, logs a warning instead.
        """
        LOGGER.info(f"Loading model from step {step_idx}")
        state_dict = None

        # Find model checkpoint callback.
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                state_dict = callback.load_model(step_idx)
                break

        # Restore model from state dict if found.
        if state_dict is None:
            if raise_if_not_found:
                raise ValueError("No model checkpoint callback found in callbacks.")
            else:
                LOGGER.warning("No model checkpoint callback found in callbacks.")
        else:
            self.restore_model(state_dict)

    def load_data_loaders(
        self,
        step_idx: int = -1,
        train_loader: Iterator | None = None,
        val_loader: Iterator | None = None,
        test_loader: Iterator | None = None,
    ):
        """
        Load states of the data loaders from the logging directory.

        Args:
            step_idx: Step index to load the data loaders from. If -1, uses the global train step.
            train_loader: If given, the training data loader is set to this value.
            val_loader: If given, the validation data loader is set to this value.
            test_loader: If given, the test data loader is set to this value.
        """
        if step_idx == -1:
            step_idx = self.global_step
        LOGGER.info(f"Loading data loaders from step {step_idx}")
        state_dict = None

        # Find data loader checkpoint callback.
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                state_dict = callback.load_dataloader(step_idx)
                break

        # Restore data loaders from state dict if found.
        if state_dict is None:
            LOGGER.warning("No data loader checkpoint callback found in callbacks.")
        else:
            for key, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
                if key in state_dict and loader is not None and hasattr(loader, "set_state"):
                    loader.set_state(state_dict[key])

    def restore_model(self, state_dict: dict[str, Any] | FrozenDict[str, Any]):
        """
        Restore the state of the trainer from a state dictionary.

        Args:
            state_dict: State dictionary to restore from. Must contain the key "params" with the model parameters.
                Optional keys that overwrite the trainer state are "step", "opt_state", "mutable_variables", "rng".
        """
        LOGGER.info("Restoring trainer state with keys " + str(state_dict.keys()))
        assert "params" in state_dict, "State dictionary must contain the key 'params'."
        state_dict = freeze(state_dict)

        # Transfer state dict into train state.
        self.state = TrainState(
            step=state_dict.get("step", 0),
            apply_fn=self.model.apply,
            params=state_dict["params"],
            tx=self.state.tx if self.state.tx else self.init_optimizer(self.optimizer_config),
            opt_state=state_dict.get("opt_state", None),
            mutable_variables=state_dict.get("mutable_variables", None),
            rng=state_dict.get("rng", self.state.rng),
        )
        self.global_step = jax.device_get(self.state.step).item()

    def restore_data_loaders(
        self,
        state_dict: dict[str, Any],
        train_loader: Iterator | None = None,
        val_loader: Iterator | None = None,
        test_loader: Iterator | None = None,
    ):
        """
        Restore the state of the data loaders from a state dictionary.

        Args:
            state_dict: State dictionary to restore from. Should contain the keys "train", "val" and "test" with
                the data loader states.
            train_loader: If given, the training data loader is set to this value.
            val_loader: If given, the validation data loader is set to this value.
            test_loader: If given, the test data loader is set to this value.
        """
        LOGGER.info("Restoring data loaders with keys " + str(state_dict.keys()))
        # Restore data loaders from state dict.
        for key, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
            if key in state_dict:
                if loader is None:
                    LOGGER.warning(
                        f"Data loader {key} had saved state dict, but was not provided in the restoring function. "
                        "Skipping."
                    )
                    continue
                elif not hasattr(loader, "set_state"):
                    LOGGER.warning(f"Data loader {key} had saved state dict, but no set_state method. Skipping.")
                    continue
                else:
                    LOGGER.info(f"Restoring data loader {key}")
                    loader.set_state(state_dict[key])

    def load_pretrained_model(
        self,
        checkpoint_path: Path,
        step_idx: int = -1,
        load_best: bool = False,
        train_loader: Iterator | None = None,
        val_loader: Iterator | None = None,
        test_loader: Iterator | None = None,
    ):
        """
        Load a pretrained model from a checkpoint directory.

        Args:
            checkpoint_path: Path to the checkpoint directory.
            step_idx: Step index to load the model from. If -1, the latest model is loaded.
            load_best: If True, loads the best model instead of the latest model.
            raise_if_not_found: If True, raises an error if no model is found. If False, logs a warning instead.
            train_loader: If given, the training data loader is set to the state of the pretrained model.
            val_loader: If given, the validation data loader is set to the state of the pretrained model.
            test_loader: If given, the test data loader is set to the state of the pretrained model.
        """
        LOGGER.info(f"Loading pretrained model from {checkpoint_path}")
        state_dict, data_module_state = load_pretrained_model(
            checkpoint_path, trainer=self, step_idx=step_idx, load_best=load_best
        )
        assert len(state_dict) > 0, "No model checkpoint found in the directory."
        self.restore_model(state_dict)
        self.restore_data_loaders(
            data_module_state, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader
        )

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint: str,
        exmp_input: Batch = None,
        batch_size: int = -1,
    ) -> Any:
        """
        Create a Trainer object with same hyperparameters and loaded model from a checkpoint directory.

        Args:
            checkpoint: Folder in which the checkpoint and hyperparameter file is stored.
            exmp_input: An input to the model for shape inference.
            batch_size: Batch size to use for shape inference. If -1, the full exmp_input is used.

        Returns:
            A Trainer object with model loaded from the checkpoint folder.
        """
        # Load config.
        metadata_file = os.path.join(checkpoint, "metadata/metadata")
        assert os.path.isfile(metadata_file), "Could not find metadata file"
        with open(metadata_file, "rb") as f:
            config = ConfigDict(json.load(f))

        # Adjust log dir to where its loaded from.
        adjusted_checkpoint = checkpoint.split("/")
        if adjusted_checkpoint[-1] == "":
            adjusted_checkpoint = adjusted_checkpoint[:-1]
        if len(adjusted_checkpoint) < 2:
            raise ValueError("Checkpoint path must be at least two levels deep")
        config.trainer.logger.log_path = Path(os.path.join(*adjusted_checkpoint[:-2]))

        # Load example input.
        # TODO: We may want to load the example input from the checkpoint folder.
        assert exmp_input is not None, "Example input must be provided"
        if batch_size > 0:
            exmp_input = exmp_input[:batch_size]

        # Create trainer and load model.
        trainer = cls(
            exmp_input=exmp_input,
            trainer_config=config.trainer,
            model_config=config.model,
            optimizer_config=config.optimizer,
        )
        trainer.load_model()
        return trainer
