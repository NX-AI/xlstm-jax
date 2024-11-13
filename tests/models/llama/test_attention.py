import jax
import jax.numpy as jnp
import numpy as np

from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.llama.attention import SelfAttention, SelfAttentionConfig
from xlstm_jax.utils import flatten_pytree


def test_attention_backends():
    """
    Tests attention backends.
    """
    jax_device_backend = jax.default_backend()
    backends = ["xla"]
    if jax_device_backend in ["cpu", "gpu"]:
        backends.append("pallas_triton")
    if jax_device_backend == "gpu":
        backends.append("cudnn")

    if jax_device_backend == "cpu":
        batch_size = 2
        context_length = 4
        hidden_size = 16
        head_dim = 4
        dtype = "float32"
    else:
        batch_size = 8
        context_length = 256
        hidden_size = 512
        head_dim = 128
        dtype = "bfloat16"

    inp_rng, segm_rng, model_rng = jax.random.split(jax.random.PRNGKey(0), 3)
    x = jax.random.normal(inp_rng, (batch_size, context_length, hidden_size), dtype=dtype)
    segment_ids = jax.random.randint(segm_rng, (batch_size, context_length), 0, 2)

    # Initialize attention modules.
    outputs = {}
    params = {}
    grads = {}
    for attn_backend in backends:
        config = SelfAttentionConfig(
            head_dim=head_dim,
            causal=True,
            attention_backend=attn_backend,
            dtype=dtype,
            parallel=ParallelConfig(),
        )
        attn = SelfAttention(config)
        outputs[attn_backend], params[attn_backend] = jax.device_get(
            attn.init_with_output(model_rng, x, segment_ids=segment_ids)
        )

        def loss_fn(inputs):
            params, x = inputs["params"], inputs["x"]
            out = attn.apply(params, x, segment_ids=segment_ids)
            loss = ((out - x) ** 2).astype(jnp.float32).sum(axis=-1)
            return loss.mean()

        grads[attn_backend] = jax.grad(loss_fn)({"params": params[attn_backend], "x": x})

    # Select default backend.
    ref_backend = backends[0]
    ref_outputs_flattened = flatten_pytree(outputs[ref_backend])
    ref_params_flattened = flatten_pytree(params[ref_backend])
    ref_grads_flattened = flatten_pytree(grads[ref_backend])

    for attn_backend in backends:
        # Check parameters.
        attn_params_flattened = flatten_pytree(params[attn_backend])
        for key in ref_params_flattened:
            assert key in attn_params_flattened, f"Parameter key {key} missing for backend {attn_backend}."
            np.testing.assert_array_equal(
                ref_params_flattened[key],
                attn_params_flattened[key],
                err_msg=f"Parameter value mismatch for key {key} in backend {attn_backend} vs {ref_backend}.",
            )

        # Check outputs.
        attn_outputs_flattened = flatten_pytree(outputs[attn_backend])
        for key in ref_outputs_flattened:
            assert key in attn_outputs_flattened, f"Output key {key} missing for backend {attn_backend}."
            np.testing.assert_allclose(
                ref_outputs_flattened[key].astype(np.float32),
                attn_outputs_flattened[key].astype(np.float32),
                atol=1e-3,
                rtol=1e-3,
                err_msg=f"Output value mismatch for key {key} in backend {attn_backend} vs {ref_backend}.",
            )

        # Check gradients.
        attn_grads_flattened = flatten_pytree(grads[attn_backend])
        for key in ref_grads_flattened:
            assert key in attn_grads_flattened, f"Gradient key {key} missing for backend {attn_backend}."
            np.testing.assert_allclose(
                ref_grads_flattened[key].astype(np.float32),
                attn_grads_flattened[key].astype(np.float32),
                atol=4e-4,
                rtol=1e-3,
                err_msg=f"Gradient value mismatch for key {key} in backend {attn_backend} vs {ref_backend}.",
            )
