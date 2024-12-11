# Distributed Training
xLSTM-jax supports different parallelization strategies (data, fsdp, tensor) and activation checkpointing (referred to as remat in JAX),
enabling efficient training on large-scale distributed systems with hundreds or thousands of GPUs.
For an in-depth tutorial on 3D parallelization in JAX, please see [Training Models At Scale](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/overview.html) by Phillip Lippe,
covering
- [Data Parallelism (DP)](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/data_parallel_fsdp.html#Data-Parallelism)
- [Fully-sharded Data Parallelism (FSDP)](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/data_parallel_fsdp.html#Parameter-Sharding)
- [Tensor Parallelism (TP)](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/tensor_parallel_simple.html)

Several default parallelization configurations are located in `configs/parallel` directory.
For model sizes of up to 7B parameters, trained on up to 256 H100 GPUs, we found the combination of FSDP and activation checkpointing (remat) to be the most performant on our cluster (different hardware setups may favor different setups).
Tensor parallelization is required for larger models.

## Distributed Computing in JAX

### JIT vs Shard Map

In JAX, distributed computing can be implemented by way of automatic parallelization with [`jax.jit`](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) or manual sharding with [`shard_map`](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html). In our initial experiments, we obtained significantly better performance in reducing communication bottlenecks with `shard_map` than with `jit` on our compute cluster. Hence, we use `shard_map` for our parallelization strategies, but note that for other hardware setups, `jit` may be more performant.

### Multi-Host Training

For multi-host training, one JAX process needs to be started per host (see [official documentation](https://jax.readthedocs.io/en/latest/multi_process.html)). On a SLURM cluster, we start by default one process per GPU by way of `jax.distributed.initialize()`, which is coordinated by the environment variables set by SLURM. On a single host, we also support running a single process for all GPUs (for example for quick debugging).

Each JAX process starts its own data loading pipeline, in which each process loads only its respective shard of the full dataset. The batch size is then divided by the number of processes. For model parallelization strategies (TP or PP), we first gather the batch over the model parallel dimension before applying the model forward pass. To support batch sizes smaller than 1 per device, one could integrate "fake" data loaders like in [MaxText](https://github.com/AI-Hypercomputer/maxtext) and slice the batch accordingly within the forward pass. However, as this is not needed for a 7B model scale on H100s, we are not yet supporting this.

## Parallelization Strategies

For a detailed explanation of the parallelization strategies with `shard_map`, please see the [Training Models At Scale](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/overview.html) overview. We create a distributed 4D mesh with the following dimensions:
- `dp` (data parallel)
- `fsdp` (fully sharded data parallel)
- `pp` (pipeline parallel)
- `tp` (tensor parallel)

For our 7B model training on 256 H100 GPUs, we used a mesh shape of `(32, 8, 1, 1)` (ie FSDP within a node, DP across nodes).

### Data Parallelism

In data parallelism, the model is replicated across all devices, and each device processes a different batch of data. The gradients are then averaged across all devices. While some gradient communications can be performed asynchronously, note that stacking parameters by way of the scan transformation over layers may force some communications to only happen after all layers have been processed. However, this did not significantly impact our training performance.

### Fully-Sharded Data Parallelism

Fully-sharded data parallelism (FSDP) extends DP by sharding the model parameters and optimizer state across all devices. Before the forward pass of a module, we gather the model parameters over the `fsdp` dimension. Similarly, during the backward pass, we scatter the gradients over the `fsdp` dimension. This strategy reduces the memory footprint of the model on each device, as only a fraction of the model is stored on each device.

Separating the DP and FSDP dimensions allows for a flexible combination of both strategies. For example, we can use FSDP within a node to take advantage of its fast communication, and DP across nodes to scale to a large number of devices. Note that most FSDP communications can be performed asynchronously, which can further reduce the communication overhead. To prevent minor tensors from being sharded (for example biases or input/forget gate), we only shard tensors with a size larger than a certain threshold (specified in config). Additionally, for faster communication, we support gather the weights in different precisions (for example `bfloat16`), if they are casted to the same precision before the forward pass.

### Pipeline Parallelism

Pipeline parallelism (PP) splits the model "vertically" (that is layer dimension) across the `pp` dimension. However, due to the max training size of 256 GPUs, we are not actively using and supporting PP in our current setup. For model parallelism, we recommend using TP instead.

### Tensor Parallelism

Tensor parallelism (TP) splits the model "horizontally" (that is feature dimension) across the `tp` dimension, which is useful for models that do not fit on a single device. As the xLSTM-7B model has a very similar architecture to a standard Transformer, we can apply the same tensor parallelization strategies: in the mLSTM block, we split the heads across the `tp` dimension, and in the feedforward block, we split the up-projected hidden dimension across the `tp` dimension. As TP can introduce some communication bottlenecks, we support the asynchronous TP execution of linear layers, similar to the [ViT-22b](https://arxiv.org/abs/2302.05442), but note that GPUs with fully connected NVLinks may not benefit from this due to the communication layout.

### Miscellaneous

- **Activation Checkpointing (Remat)**: We use activation checkpointing (remat) to reduce the memory footprint of the model and to reduce the communication overhead. This is especially important for large models like xLSTM-7B, as it allows us to trade off memory for computation. We use the `remat` transformation around the individual blocks of the xLSTM model.
- **Gradient Accumulation**: We support gradient accumulation across multiple steps, which can be useful for large batch sizes or for models that do not fit on a single device.
- **Mixed Precision Training**: We support mixed precision training with setting the JAX dtype in the model config (for example to `bfloat16`), which can reduce the memory footprint and increase the training speed.
- **Distributed Gradient Clipping**: We support gradient clipping to prevent exploding gradients, which computes the norm across all (distributed) gradients and clips them accordingly.
