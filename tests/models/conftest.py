import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from xlstm_jax.dataset import LLMBatch
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.trainer.callbacks.checkpointing import ModelCheckpointConfig
from xlstm_jax.trainer.llm.trainer import LLMTrainer, LLMTrainerConfig
from xlstm_jax.trainer.logger import LoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig


class LLMTrainerHelper:
    @staticmethod
    def model_training_test(
        tmp_path: Path,
        model_config_generator: Callable[[ParallelConfig], ModelConfig],
        fsdp_size: int,
        batch_size: int = 8,
        context_length: int = 12,
        vocab_size: int = 50,
        check_saved_model: bool = True,
    ):
        """
        Tests training a model.
        """
        parallel = ParallelConfig(
            data_axis_size=-1,
            model_axis_size=1,
            fsdp_axis_size=fsdp_size,
            fsdp_min_weight_size=pytest.num_devices,
        )
        model_config = model_config_generator(parallel)
        optimizer_config = OptimizerConfig(
            name="adam",
            scheduler=SchedulerConfig(
                name="constant",
                lr=1e-4,
                decay_steps=150,
                warmup_steps=20,
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
                check_val_every_n_epoch=1,
            ),
            model_config,
            optimizer_config,
            batch=LLMBatch.get_dtype_struct(batch_size=batch_size, max_length=context_length),
        )

        def data_gen_fn(idx: int) -> LLMBatch:
            inputs = jax.random.randint(
                jax.random.PRNGKey(idx), (batch_size, context_length), minval=0, maxval=vocab_size
            )
            targets = jnp.mod(inputs + 1, vocab_size)
            return LLMBatch.from_inputs(inputs=inputs, targets=targets)

        train_loader = [data_gen_fn(idx) for idx in range(50)]
        val_loader = train_loader[:10]
        final_metrics = trainer.train_model(
            train_loader,
            val_loader,
            num_epochs=3,
        )
        assert final_metrics is not None
        assert all(f"val_epoch_{i}" in final_metrics for i in range(1, 4))
        assert "perplexity" in final_metrics["val_epoch_1"]
        assert not np.any(np.isnan(final_metrics["val_epoch_1"]["perplexity"]))
        assert (
            final_metrics["val_epoch_3"]["perplexity"] < final_metrics["val_epoch_2"]["perplexity"]
        ), "Validation perplexity should decrease over epochs."
        # Check that checkpoints have been created.
        assert os.path.exists(tmp_path)
        assert os.path.exists(tmp_path / "checkpoints")
        assert len(os.listdir(tmp_path / "checkpoints/")) == 2
        assert os.path.exists(tmp_path / "checkpoints" / "checkpoint_100"), (
            f"Checkpoint 100 should exist, but found {os.listdir(tmp_path / 'checkpoints/')} with "
            f"final metrics {final_metrics}."
        )
        if check_saved_model:
            # Check that validation performance can be reproduced with checkpoint.
            trainer.load_model(step_idx=100)
            new_metrics = trainer.eval_model(val_loader, "val", epoch_idx=2)
            assert "perplexity" in new_metrics
            assert "loss" in new_metrics
            assert (
                new_metrics["perplexity"] == final_metrics["val_epoch_2"]["perplexity"]
            ), f"Perplexity should match the loaded model: {new_metrics} versus {final_metrics['val_epoch_2']}"
            assert new_metrics["loss"] == final_metrics["val_epoch_2"]["loss"], "Loss should match the loaded model."
            # Check that we can continue training from this checkpoint as before.
            new_final_metrics = trainer.train_model(
                train_loader,
                val_loader,
                num_epochs=3,
            )
            assert new_final_metrics is not None
            assert (
                new_final_metrics["val_epoch_3"]["perplexity"] == final_metrics["val_epoch_3"]["perplexity"]
            ), "Perplexity should match the loaded model."

    @staticmethod
    def causal_masking_test(
        model_generator: Callable[[ParallelConfig], nn.Module],
        batch_size: int = 8,
        context_length: int = 32,
        vocab_size: int = 50,
        test_document_borders: bool = False,
    ):
        """
        Tests the causal masking in autoregressive language models.

        Args:
            model_generator: A function that generates the model given a parallel configuration.
            batch_size: The batch size.
            context_length: The context length. Should be larger than 8 to test causal masking.
            vocab_size: The vocabulary size.
            test_document_borders: Whether to test that documents are independent.
        """
        assert context_length > 8, "Context length should be larger than 8 to test causal masking."
        parallel = ParallelConfig(
            data_axis_size=-1,
            model_axis_size=1,
            fsdp_axis_size=1,
            fsdp_min_weight_size=pytest.num_devices,
        )
        model = model_generator(parallel)
        mesh = initialize_mesh(parallel)

        exmp_input = jax.random.randint(
            jax.random.PRNGKey(0), (batch_size, context_length), minval=0, maxval=vocab_size
        )

        def _init_model(init_rng: jax.Array, batch_input: jax.Array) -> Any:
            param_rng, dropout_rng = jax.random.split(init_rng)
            # Initialize parameters.
            variables = model.init({"params": param_rng, "dropout": dropout_rng}, batch_input)
            return variables

        # Prepare PRNG.
        init_rng = jax.random.PRNGKey(42)
        # First infer the output sharding to set up shard_map correctly.
        # This does not actually run the init, only evaluates the shapes.
        init_model_fn = jax.jit(
            shard_map(
                _init_model,
                mesh,
                in_specs=(P(), P()),
                out_specs=P(),
                check_rep=False,
            ),
        )
        variables_shapes = jax.eval_shape(init_model_fn, init_rng, exmp_input)
        variables_partition_specs = nn.get_partition_spec(variables_shapes)
        # Run init model function again with correct output specs.
        init_model_fn = jax.jit(
            shard_map(
                _init_model,
                mesh,
                in_specs=(P(), P()),
                out_specs=variables_partition_specs,
                check_rep=False,
            ),
        )
        variables = init_model_fn(init_rng, exmp_input)

        def _forward(
            batch_input: jax.Array, variables: Any, batch_position: jax.Array | None, batch_borders: jax.Array | None
        ) -> jax.Array:
            return model.apply(
                variables,
                batch_input,
                pos_idx=batch_position,
                document_borders=batch_borders,
                train=True,
                rngs={"dropout": jax.random.PRNGKey(42)},
            )

        forward_fn = jax.jit(
            shard_map(
                _forward,
                mesh,
                in_specs=(P(), variables_partition_specs, P(), P()),
                out_specs=P(),
                check_rep=False,
            ),
        )
        # Run forward function.
        logits = forward_fn(exmp_input, variables, None, None)
        logits = jax.device_get(logits)
        assert logits.shape == (batch_size, context_length, vocab_size), f"Logits shape: {logits.shape}"
        # Check that the model is causal.
        exmp_input_perturbed = exmp_input.at[0, 4].set((exmp_input[0, 4] + 1) % vocab_size)
        logits_perturbed = forward_fn(exmp_input_perturbed, variables, None, None)
        logits_perturbed = jax.device_get(logits_perturbed)
        np.testing.assert_array_equal(
            logits[1:],
            logits_perturbed[1:],
            err_msg="Perturbing one batch element should not affect other batch elements.",
        )
        np.testing.assert_array_equal(
            logits[0, :3],
            logits_perturbed[0, :3],
            err_msg="Causal masking failed, earlier tokens should not be affected.",
        )
        assert not np.allclose(
            logits[0, 4], logits_perturbed[0, 4]
        ), "Perturbing a token should change the output for the perturbed token."
        assert not np.allclose(
            logits[0, 5:], logits_perturbed[0, 5:]
        ), "Perturbing a token should change the output for future tokens."

        if test_document_borders:
            # Create document borders that are spaced over sequences.
            border_idx = jnp.linspace(1, context_length - 1, batch_size, dtype=jnp.int32)
            document_borders = jnp.arange(context_length)[None] == border_idx[:, None]
            document_borders = document_borders.at[:, 0].set(True)
            pos_idx = jnp.stack(
                [
                    np.concatenate([np.arange(border_idx[i]), np.arange(context_length - border_idx[i])])
                    for i in range(batch_size)
                ],
                axis=0,
            )
            # Run forward function with document borders.
            logits = forward_fn(exmp_input, variables, pos_idx, document_borders)
            logits = jax.device_get(logits)
            assert logits.shape == (batch_size, context_length, vocab_size), f"Logits shape: {logits.shape}"
            # Check that the model is causal.
            pert_idx = context_length // 2
            exmp_input_perturbed = exmp_input.at[:, pert_idx].set((exmp_input[:, pert_idx] + 1) % vocab_size)
            logits_perturbed = forward_fn(exmp_input_perturbed, variables, pos_idx, document_borders)
            logits_perturbed = jax.device_get(logits_perturbed)
            for i in range(batch_size):
                postfix = f" Failed for idx={i}, border={border_idx[i]}, pert={pert_idx}."
                np.testing.assert_array_equal(
                    logits[i, :pert_idx],
                    logits_perturbed[i, :pert_idx],
                    err_msg="Causal masking failed, earlier tokens should not be affected." + postfix,
                )
                assert not np.allclose(logits[i, pert_idx], logits_perturbed[i, pert_idx]), (
                    "Perturbing a token should change the output for the perturbed token." + postfix
                )
                if border_idx[i] <= pert_idx:
                    assert not np.allclose(logits[i, pert_idx + 1 :], logits_perturbed[i, pert_idx + 1 :]), (
                        "Perturbing a token should change the output for future tokens." + postfix
                    )
                else:
                    if pert_idx + 1 < border_idx[i] - 1:
                        assert not np.allclose(
                            logits[i, pert_idx + 1 : border_idx[i] - 1],
                            logits_perturbed[i, pert_idx + 1 : border_idx[i] - 1],
                        ), "Perturbing a token should change the output for future tokens." + postfix
                    np.testing.assert_allclose(
                        logits[i, border_idx[i] :],
                        logits_perturbed[i, border_idx[i] :],
                        atol=1e-6,
                        rtol=1e-6,
                        err_msg="Document border masking failed, tokens from future documents should not be affected."
                        + postfix,
                    )

            # Feed in the same batch twice, once with document order flipped. This should be equivalent.
            def flip_array(x: jax.Array):
                return jnp.stack(
                    [jnp.concatenate([x[i, border_idx[i] :], x[i, : border_idx[i]]]) for i in range(batch_size)], axis=0
                )

            exmp_input_flipped = flip_array(exmp_input)
            pos_idx_flipped = flip_array(pos_idx)
            document_borders_flipped = flip_array(document_borders)
            np.testing.assert_array_equal(
                document_borders_flipped[:, 0],
                document_borders[:, 0],
                err_msg="First document border should be the same.",
            )
            np.testing.assert_array_equal(
                document_borders_flipped[:, 1:],
                document_borders[:, 1:][:, ::-1],
                err_msg="Document borders should be exactly flipped up to the first token.",
            )
            logits_flipped = forward_fn(exmp_input_flipped, variables, pos_idx_flipped, document_borders_flipped)
            logits_flipped = jax.device_get(logits_flipped)
            np.testing.assert_allclose(
                flip_array(logits),
                logits_flipped,
                atol=1e-4,
                rtol=1e-4,
                err_msg="Flipping the input should not affect the output.",
            )


@pytest.fixture
def llm_trainer():
    return LLMTrainerHelper
