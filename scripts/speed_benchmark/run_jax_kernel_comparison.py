#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from collections import defaultdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tabulate

try:
    from xlstm_jax.kernels.mlstm_chunkwise.max_triton_fwbw_v3.triton_fwbw import mlstm_chunkwise_max_triton
except ImportError:
    mlstm_chunkwise_max_triton = None


def compare_jax_and_pytorch_kernels(
    file_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Compare JAX kernel to PyTorch kernel output.

    We load stored data from the PyTorch kernel and feed the same input data to JAX.
    The function returns a dictionary of the hyperparameters, the PyTorch data, and the JAX data.

    Args:
        file_path: Path to the PyTorch kernel output file.

    Returns:
        Tuple of hyperparameters, PyTorch data, and JAX data.
    """
    # Resolve hyperparameters.
    file_name = file_path.name.split(".")[0]
    int_keys = ["B", "NH", "S", "DH", "CHUNK", "seed"]
    hparams = {}
    for hparam_str in file_name.split("_"):
        if hparam_str.startswith("dtype"):
            hparams["dtype"] = {"fp32": jnp.float32, "fp16": jnp.float16, "bf16": jnp.bfloat16}[hparam_str[5:]]
        else:
            key = [k for k in int_keys if hparam_str.startswith(k)]
            assert len(key) > 0, f"Unknown hyperparameter: {hparam_str}"
            key = key[0]
            hparams[key] = int(hparam_str[len(key) :])

    # Load PyTorch data.
    data = np.load(file_path)
    data_torch = {k: data[k] for k in data}

    # Map input to JAX arrays.
    q = jnp.array(data_torch["q"]).astype(hparams["dtype"])
    k = jnp.array(data_torch["k"]).astype(hparams["dtype"])
    v = jnp.array(data_torch["v"]).astype(hparams["dtype"])
    igate_preact = jnp.array(data_torch["igate_preact"]).astype(hparams["dtype"])
    fgate_preact = jnp.array(data_torch["fgate_preact"]).astype(jnp.float32)

    # Implement JAX forward-backward path.
    def loss_fn(q, k, v, igate_preact, fgate_preact):
        h_out_triton = mlstm_chunkwise_max_triton(
            q, k, v, igate_preact, fgate_preact, chunk_size=hparams["CHUNK"], autocast_kernel_dtype=hparams["dtype"]
        )
        h_out_triton = h_out_triton.astype(jnp.float32)
        # We skip the normalization, as both JAX and PyTorch should have the same kernels without normalization.
        return h_out_triton.sum(), h_out_triton

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4), has_aux=True))
    (loss_jax, h_out_triton), (grad_q, grad_k, grad_v, grad_igate_preact, grad_fgate_preact) = grad_fn(
        q, k, v, igate_preact, fgate_preact
    )

    # Convert JAX outputs to numpy.
    loss_jax = jax.device_get(loss_jax)
    h_out_triton = jax.device_get(h_out_triton.astype(np.float32))
    grad_titon = [grad_q, grad_k, grad_v, grad_igate_preact, grad_fgate_preact]
    grad_titon = [jax.device_get(g.astype(jnp.float32)) for g in grad_titon]
    data_jax = {
        "out": h_out_triton,
        "loss": loss_jax,
        "grad_q": grad_q,
        "grad_k": grad_k,
        "grad_v": grad_v,
        "grad_igate_preact": grad_igate_preact,
        "grad_fgate_preact": grad_fgate_preact,
    }

    # Return hyperparameters, PyTorch data, and JAX data.
    return hparams, data_torch, data_jax


if __name__ == "__main__":
    assert mlstm_chunkwise_max_triton is not None, "Triton kernels are not available."

    # Default directory where we stored the PyTorch kernel outputs.
    pytorch_kernel_output_files = sorted(
        list(Path("/nfs-gpu/xlstm/shared_tests/mlstm_pytorch_kernel_outputs/chunkwise_fwbw").glob("*.npz"))
    )
    assert len(pytorch_kernel_output_files) > 0, "No PyTorch kernel outputs found."

    # Compare kernels for each file.
    for file_path in pytorch_kernel_output_files:
        print(f"Comparing {file_path.name}...")
        hparams, data_torch, data_jax = compare_jax_and_pytorch_kernels(file_path)

        # Print comparison metrics in a table.
        table = defaultdict(list)
        for key in data_jax:
            diffs = np.abs(data_torch[key] - data_jax[key])
            table["Param"].append(key)
            table["Max Abs Diff"].append(diffs.max())
            table["Max Rel Diff"].append((diffs / (np.abs(data_torch[key]) + 1e-8)).max())
            table["Mean Abs Diff"].append(diffs.mean())
            if diffs.ndim >= 3:
                table["Last-16 Diff"].append(diffs[:, :, -16:].mean())
            else:
                table["Last-16 Diff"].append("n/a")
            for q in [0.5, 0.9, 0.99]:
                table[f"Quant-{q} Diff"].append(np.quantile(diffs, q))

        print(
            "Setting:" + "\n" + tabulate.tabulate([[k, v] for k, v in hparams.items()], headers=["Key", "Value"]) + "\n"
        )
        print("\n" + tabulate.tabulate(table, headers="keys") + "\n")
        print("#" * 80 + "\n")
