from pathlib import Path

import numpy as np
import torch

from mlstm_kernels.mlstm_kernels.mlstm.chunkwise.max_triton_fwbw_v3.triton_fwbw import mlstm_chunkwise_fwbw

assert torch.cuda.is_available(), "This script needs to be run with the PyTorch environment and GPU support."


def run_pytorch_kernel(
    batch_size: int,
    num_heads: int,
    context_length: int,
    head_dimension: int,
    chunk_size: int,
    seed: int,
    autocast_kernel_dtype: torch.dtype = torch.float32,
    out_path: Path = Path("/nfs-gpu/xlstm/shared_tests/mlstm_pytorch_kernel_outputs/chunkwise_fwbw"),
) -> None:
    """
    Run the PyTorch kernels and store to disk.

    Args:
        batch_size: Batch size.
        num_heads: Number of heads.
        context_length: Context length.
        head_dimension: Head dimension.
        chunk_size: Chunk size.
        seed: Random seed.
        autocast_kernel_dtype: Data type for the kernel.
        out_path: Output path for the kernel data.
    """
    # Generate random inputs.
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(batch_size, num_heads, context_length, head_dimension)).astype(np.float32)
    k = rng.normal(size=(batch_size, num_heads, context_length, head_dimension)).astype(np.float32)
    v = rng.normal(size=(batch_size, num_heads, context_length, head_dimension)).astype(np.float32)
    igate_preact = rng.normal(size=(batch_size, num_heads, context_length)).astype(np.float32)
    fgate_preact = rng.normal(size=(batch_size, num_heads, context_length)).astype(np.float32)

    # Set device to GPU.
    device = torch.device("cuda")

    # Set up input data for the PyTorch backend.
    q_torch = torch.from_numpy(q).to(device=device, dtype=autocast_kernel_dtype).requires_grad_()
    k_torch = torch.from_numpy(k).to(device=device, dtype=autocast_kernel_dtype).requires_grad_()
    v_torch = torch.from_numpy(v).to(device=device, dtype=autocast_kernel_dtype).requires_grad_()
    igate_preact_torch = torch.from_numpy(igate_preact).to(device=device, dtype=autocast_kernel_dtype).requires_grad_()
    # Keep forget gate in float32, as it is upcasted in the kernel anyway.
    fgate_preact_torch = torch.from_numpy(fgate_preact).to(device=device, dtype=torch.float32).requires_grad_()

    # Run the PyTorch kernel.
    out_torch: torch.Tensor = mlstm_chunkwise_fwbw(
        q_torch,
        k_torch,
        v_torch,
        igate_preact_torch,
        fgate_preact_torch,
        CHUNK_SIZE=chunk_size,
        autocast_kernel_dtype=autocast_kernel_dtype,
    )
    out_torch = out_torch.to(torch.float32)
    loss = out_torch.sum()
    loss.backward()

    # Convert to numpy on CPU.
    loss = loss.detach().cpu().numpy()
    out_torch = out_torch.detach().cpu().numpy()
    grad_q = q_torch.grad.to(torch.float32).cpu().numpy()
    grad_k = k_torch.grad.to(torch.float32).cpu().numpy()
    grad_v = v_torch.grad.to(torch.float32).cpu().numpy()
    grad_igate_preact = igate_preact_torch.grad.to(torch.float32).cpu().numpy()
    grad_fgate_preact = fgate_preact_torch.grad.to(torch.float32).cpu().numpy()

    # Store hyperparameters in filename.
    dtype_name = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }[autocast_kernel_dtype]
    filename = (
        f"B{batch_size}_NH{num_heads}_S{context_length}_DH{head_dimension}"
        f"_CHUNK{chunk_size}_seed{seed}_dtype{dtype_name}.npz"
    )

    # Save to disk.
    np.savez_compressed(
        out_path / filename,
        q=q,
        k=k,
        v=v,
        igate_preact=igate_preact,
        fgate_preact=fgate_preact,
        out=out_torch,
        loss=loss,
        grad_q=grad_q,
        grad_k=grad_k,
        grad_v=grad_v,
        grad_igate_preact=grad_igate_preact,
        grad_fgate_preact=grad_fgate_preact,
    )


if __name__ == "__main__":
    COMBINATIONS = [
        # (B, NH, S, DH, CHUNK_SIZE)
        (1, 2, 128, 16, 16),
        (1, 2, 128, 64, 64),
        (1, 2, 128, 128, 128),
        (8, 4, 128, 64, 64),
        (1, 1, 4096, 16, 16),
        (1, 1, 4096, 384, 64),
    ]
    seed = 42
    for comb in COMBINATIONS:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            print(f"Generating: {comb}, seed={seed}, dtype={dtype}")
            run_pytorch_kernel(*comb, seed=seed, autocast_kernel_dtype=dtype)
            print("Finished generating.")
