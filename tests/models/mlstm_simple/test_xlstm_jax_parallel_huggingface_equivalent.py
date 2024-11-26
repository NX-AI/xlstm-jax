from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
import torch
from flax import linen as nn
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend import mLSTMBackendNameAndKwargs
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.cell import mLSTMCellConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_jax.models.xlstm_parallel.components.feedforward import FeedForwardConfig
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig


@dataclass
class ModelConfig:
    batch_size: int = 1
    vocab_size: int = 16
    embedding_dim: int = 128
    num_blocks: int = 1
    num_heads: int = 2
    context_length: int = 64
    backend: str = "parallel_stabilized"
    backend_name: str = ""


@pytest.mark.skip(reason="This tests requires imports submodules, which does not work currently.")
@pytest.mark.skipif(
    not pytest.huggingface_xlstm_available, reason="HuggingFace transformers library installed without xLSTM."
)
@pytest.mark.parametrize(
    "batch_size, vocab_size, embedding_dim, num_blocks, num_heads, context_length, backend, backend_name, param_noise_std",  # noqa
    [
        (1, 16, 128, 1, 2, 64, "parallel_stabilized", "", 1.0),
        # (1, 256, 256, 4, 4, 64, "parallel_stabilized", "", 1.0), # commented them out to reduce the test time on CI
        # (1, 256, 256, 8, 4, 64, "parallel_stabilized", "", 1.0),
    ],
)
def test_mLSTMv1_jax_cpu(
    tmp_path: Path,
    batch_size,
    vocab_size,
    embedding_dim,
    num_blocks,
    num_heads,
    context_length,
    backend,
    backend_name,
    param_noise_std: float | None,
):
    """Test that JAX mLSTMv1 produce the same output as mlstm simple within HuggingFace."""
    from transformers import AutoModelForCausalLM, xLSTMForCausalLM  # noqa: C0415

    from xlstm_jax.utils.model_param_handling.handle_mlstm_simple import (
        pipeline_convert_mlstm_checkpoint_jax_to_torch_simple,
        store_mlstm_simple_to_checkpoint,
    )

    cfg = ModelConfig(
        batch_size=batch_size,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        context_length=context_length,
        backend=backend,
        backend_name=backend_name,
    )
    if jax.default_backend() != "cpu":
        pytest.skip("PyTorch backend can only be run on CPU so far, difference are not representative with GPU.")

    parallel = ParallelConfig(
        data_axis_name="dp",
        fsdp_axis_name="fsdp",
        model_axis_name="tp",
        pipeline_axis_name="pp",
        fsdp_modules=[],
        fsdp_gather_dtype="bfloat16",
        fsdp_min_weight_size=2**18,
        remat=[],
        fsdp_axis_size=1,
        model_axis_size=1,
        data_axis_size=1,
        tp_async_dense=False,
    )
    mesh = initialize_mesh(parallel_config=parallel, device_array=np.array(jax.devices())[0:1])

    dtype_str = "float32"
    xlstm_config = xLSTMLMModelConfig(
        vocab_size=cfg.vocab_size,
        embedding_dim=cfg.embedding_dim,
        num_blocks=cfg.num_blocks,
        context_length=cfg.context_length,
        tie_weights=False,
        add_embedding_dropout=False,
        add_post_blocks_norm=True,
        parallel=parallel,
        scan_blocks=True,
        norm_eps=1e-6,
        norm_type="rmsnorm",
        init_distribution_out="normal",
        init_distribution_embed="normal",
        logits_soft_cap=30.0,
        lm_head_dtype=dtype_str,
        dtype=dtype_str,
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                layer_type="mlstm_v1",
                num_heads=cfg.num_heads,
                init_distribution="normal",
                output_init_fn="wang",
                qk_dim_factor=0.5,
                v_dim_factor=1.0,
                mlstm_cell=mLSTMCellConfig(
                    gate_dtype="float32",
                    backend=mLSTMBackendNameAndKwargs(
                        name=cfg.backend,
                        kwargs=dict(backend_name=cfg.backend_name) if cfg.backend == "triton_kernels" else dict(),
                    ),
                    igate_bias_init_range=0.0,
                    add_qk_norm=False,
                    norm_type="rmsnorm",
                    norm_eps=1e-6,
                    gate_soft_cap=15,
                    reset_at_document_boundaries=False,
                ),
            ),
            feedforward=FeedForwardConfig(
                proj_factor=2.667,
                act_fn="swish",
                ff_type="ffn_gated",
                dtype=dtype_str,
                output_init_fn="wang",
                init_distribution="normal",
            ),
            add_post_norm=False,
        ),
    )

    xlstm_model_jax = xLSTMLMModel(xlstm_config)

    exmp_input = jax.random.randint(
        jax.random.PRNGKey(0), (cfg.batch_size, cfg.context_length), minval=0, maxval=cfg.vocab_size
    )

    def _init_model(rng: jax.Array, batch_input: jax.Array) -> Any:
        param_rng, dropout_rng = jax.random.split(rng)
        # Initialize parameters.
        variables = xlstm_model_jax.init({"params": param_rng, "dropout": dropout_rng}, batch_input)
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
    # pprint("Jax Config:")
    # pprint(asdict(xlstm_config))
    # print("Variables jax:\n", flatten_pytree(variables))

    def _add_noise_to_array(arr: jax.Array, std: float) -> jax.Array:
        return arr + jax.random.normal(jax.random.PRNGKey(42), arr.shape) * std

    if param_noise_std is not None:
        variables["params"] = jax.tree_map(lambda x: _add_noise_to_array(x, param_noise_std), variables["params"])

    # print("Variables jax with noise:\n", flatten_pytree(variables))

    def _forward(
        batch_input: jax.Array, variables: Any, batch_position: jax.Array | None, batch_borders: jax.Array | None
    ) -> jax.Array:
        return xlstm_model_jax.apply(
            variables,
            batch_input,
            pos_idx=None,
            document_borders=None,
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

    logits_jax = forward_fn(exmp_input, variables, None, None)

    model_torch = pipeline_convert_mlstm_checkpoint_jax_to_torch_simple(
        variables["params"],
        asdict(xlstm_config),
        torch_model_config_overrides={
            "forward_backend_name": "chunkwise--native_autograd",
            "step_backend_name": "native",
        },
    )
    # pprint("Torch config:")
    # pprint(asdict(model_torch.config))
    # print("Model torch:\n", model_torch.state_dict())

    logits_jax = jax.device_get(logits_jax)

    exmp_input_torch = torch.from_numpy(jax.device_get(exmp_input))

    logits_torch = model_torch(exmp_input_torch).detach().numpy()

    assert logits_torch.shape == logits_jax.shape, "Logit shapes are different."

    np.testing.assert_allclose(logits_torch, logits_jax, rtol=5e-4, atol=5e-4)

    store_mlstm_simple_to_checkpoint(model_torch, tmp_path / "huggingface_conv", checkpoint_type="huggingface")

    hf_model_xlstm = xLSTMForCausalLM.from_pretrained(tmp_path / "huggingface_conv")

    hf_model_xlstm.save_pretrained(tmp_path / "huggingface_automodel")

    with torch.no_grad():
        logits_hfconv = hf_model_xlstm(exmp_input_torch).logits.numpy()

    assert logits_hfconv.shape == logits_jax.shape, "Logit shapes are different."

    np.testing.assert_allclose(logits_hfconv, logits_jax, rtol=5e-4, atol=5e-4)

    hf_model_auto = AutoModelForCausalLM.from_pretrained(tmp_path / "huggingface_automodel")

    with torch.no_grad():
        logits_hfauto = hf_model_auto(exmp_input_torch).logits.numpy()

    assert logits_hfauto.shape == logits_jax.shape, "Logit shapes are different."

    np.testing.assert_allclose(logits_hfconv, logits_jax, rtol=5e-4, atol=5e-4)
