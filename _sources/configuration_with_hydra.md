# Configuring Experiments with Hydra

This project uses [Hydra](https://hydra.cc/) for managing configurations. Hydra is a framework that simplifies the process of configuring complex applications by allowing you to compose configurations dynamically. If you
have never worked with Hydra it might seem daunting at first and
you might want to go through
the [Hydra tutorial](https://hydra.cc/docs/1.3/intro/) first.

We use structured configs and Python dataclasses as basis for our configuration. There is
an additional [Hydra tutorial](https://hydra.cc/docs/1.3/tutorials/structured_config/intro/) for
structured configs, which you should go through if you've never worked with those.

We will still give a very short introduction here s.t. you can get started by running your
own experiment quickly.

## Configuration Structure

### The Config Dataclasses

The first thing to be aware of is that we use Python dataclasses as foundational configuration objects.
Let us look at the example of the `ParallelConfig`, located at `xlstm_jax.models.configs.ParallelConfig`:

```python
@dataclass(kw_only=True, frozen=False)
class ParallelConfig:
    """Configuration for parallelism."""

    data_axis_size: int = -1
    """Size of the data axis. If -1, it will be inferred by the number of available devices."""
    fsdp_axis_size: int = 1
    """Size of the FSDP axis. If -1, it will be inferred by the number of available devices."""
    pipeline_axis_size: int = 1
    """Size of the pipeline axis. If -1, it will be inferred by the number of available devices."""
    model_axis_size: int = 1
    """Size of the model axis. If -1, it will be inferred by the number of available devices."""
    data_axis_name: str = "dp"
    """Name of the data axis."""
    fsdp_axis_name: str = "fsdp"
    """Name of the FSDP axis."""
    pipeline_axis_name: str = "pp"
    """Name of the pipeline axis."""
    model_axis_name: str = "tp"
    """Name of the model axis."""
    remat: list[str] = field(default_factory=lambda: [])
    """Module names on which we apply activation checkpointing / rematerialization."""
    fsdp_modules: list[str] = field(default_factory=lambda: [])
    """Module names on which we apply FSDP sharding."""
    fsdp_min_weight_size: int = 2**18
    """Minimum size of a parameter to be sharded with FSDP."""
    fsdp_gather_dtype: str | None = None
    """The dtype to cast the parameters to before gathering with FSDP. If `None`, no casting is performed and parameters
    are gathered in original precision (for example `float32`)."""
    fsdp_grad_scatter_dtype: str | None = None
    """The dtype to cast the gradients to before scattering. If `None`, the dtype of the parameters is used."""
    tp_async_dense: bool = False
    """Whether to use asynchronous tensor parallelism for dense layers. Default to `False`, as on local hardware,
    ppermute communication introduces large overhead."""

    def __post_init__(self):
        _allowed_fsdp_dtypes = ["float32", "bfloat16", "float16"]

        if self.fsdp_gather_dtype is not None:
            assert self.fsdp_gather_dtype in _allowed_fsdp_dtypes
        if self.fsdp_grad_scatter_dtype is not None:
            assert self.fsdp_grad_scatter_dtype in _allowed_fsdp_dtypes

```

Here, all attributes and default values of `ParallelConfig` can be seen. To tell Hydra which
attributes are available in the dataclass we have to add it to Hydra's config store. This is done
in `xlstm_jax/define_hydra_schemas.py` and looks like this for the `ParallelConfig`:

```python
cs.store(name="parallel_schema", group="parallel", node=ParallelConfig)
```

It tells Hydra that whenever I use the string "parallel_schema" in a configuration yaml file, it
uses `ParallelConfig` as basis and uses all attributes and default values from that class.

### The Config yaml files

To overwrite the default values from the dataclasses, Hydra
uses yaml files, which are
organized in the `configs` directory.
For now, the structure is as follows:


- `configs/`: Contains all configuration files.
  - `config.yaml`: The main configuration file. This is the most important file since the defaults are
   specified here.

  - The submodules are in subfolders:
    - `checkpointing`
    - `data`
    - `hydra`
    - `logger`
    - `lr_monitor`
    - `model`
    - `optimizer`
    - `parallel`
    - `profiling`
    - `scheduler`
    - `trainer`

  Experiment files are located in the subfolder
  - `experiment`


## How to Run Experiments

The principal entry point for hydra is the script `scripts/training/train_with_hydra.py`.

The `main_train()` function in that file is decorated with:

```python
@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main_train(cfg: DictConfig):
    ...
```

The decorator invokes Hydra and the following happens: Hydra looks in the folder `../configs` for a file named `config.yaml`. If it finds the file,
Hydra starts to compile the configuration from the defaults supplied in `configs/config.yaml`.
For example, if that file looks like this:

```yaml filename="configs/config.yaml"  #TODO: how to show this?
defaults:
  - config_schema
  - parallel: synthetic
  - model: mLSTM120M
  - _self_


# General hyperparameters. Will be put in their respective config modules once they are created.

# device. cpu or gpu
device: cpu
```

the following will happen:

1. The first line reads `- config_schema`. This is related to the use of
structured configs and tells Hydra which values and types are allowed to be set in
this file.
You can check `define_hydra_schemas.py` to see the definition for `config_schema`.

2. `- parallel: synthetic` means that Hydra will now look in the `parallel` subfolder for the `synthetic.yaml`
 file and append all parameters in that file to the compiled list of parameters under the key `parallel`.
3. `- model: mLSTM120M` means that Hydra will now look in the `model` subfolder for the `mLSTM120M.yaml`
 file and append all parameters in that file to the compiled list of parameters under the key `model`.
4. `_self_` means that all parameters in this file itself (that is, the `config.yaml`) will be inserted into the
compiled config.

Note that all values that come later in the defaults list **overwrite** potential values that have been in the
configuration before.

Once Hydra has processed this file, it changes the current working directory as specified in the configs
and executes
`main_train(cfg)` in that directory, where `cfg` is the compiled configuration.
All top level parameters are in `cfg`, as for example, `cfg.device` and the
others are in their respective fields, for example, `cfg.parallel.data_axis_name`.

**Remark**
It is possible to overwrite every parameter from the command line. So you can, for example, execute

```PYTHONPATH=. python scripts/training/train_with_hydra.py device=gpu model.num_heads=42```

to substitute `cpu`
from the `config.yaml` with `gpu` and to substitute 42 for whatever value for `num_heads` was given in `mLSTM120M.yaml` file. It is more convenient to use experiment files for overwriting
parameters though.

### Using Experiment Files
(see https://hydra.cc/docs/patterns/configuring_experiments/)

A more reproducible way to start experiments with Hydra is to use experiment files. In these files
you can specify all parameters and config groups that you want to change. Let us take the `experiment/synthetic_experiment.yaml` as an example.
The top of that file reads

```yaml
# @package _global_
defaults:
  - /data@data_train.ds1: synthetic
  - override /parallel: synthetic
  - override /model: mLSTMv1_165M
  - _self_

# specify the deltas from the defaults:
task_name: slurm_tests
batch_size_per_device: 2
context_length: 128
num_train_steps: 10
lr: 1e-3

logger:
  log_every_n_steps: 2
```

Hydra now checks the defaults list of the experiment file and replaces the respective fields from the general
`config.yaml` with the new values given here.
That is, no matter what was provided as `parallel` config in the `config.yaml`,
we override that with the `synthetic` configuration instead.

Similarly, no matter what was supplied as model in the
`config.yaml`, we overwrite that with the `mLSTMv1_165M`.


In addition, you can
overwrite any parameter with your own values. This goes for all parameters at
every level of the config hierarchy.  In this example, the value in
`cfg.logger.log_every_n_steps` is overwritten with 2.
By using
experiment files, it becomes easy to specify and reproduce experiments.

To run this experiment, you execute

```PYTHONPATH=. python scripts/training/train_with_hydra.py +experiment=synthetic_experiment```

Note the + before experiment, that's not a typo!

### Type Checking of the Configurations

One benefit of using the structured configs and dataclasses as basis for the configuration is that type checking
becomes available. This happens when compiling the `cfg`.

To test whether you have supplied correctly typed parameters in your experiment files, you can execute

```python scripts/check_config.py +experiment={YOUR_EXPERIMENT_FILE}```

This function just compiles and logs your compiled config. But it's a good way to check if you have supplied a
working experiment file before starting a job on a cluster, for example.


## How to run experiments on a SLURM cluster
To run your script on a SLURM cluster you can use the `submitit_launcher`, which should have been installed
for Hydra if you have used our conda env.
The
default configuration is provided in `configs/hydra/launcher/slurm_launcher.yaml`:

```yaml
# @package _global_
defaults:
  - override /hydra/launcher: submitit_slurm

hydra:
  launcher:
    submitit_folder: ${hydra.sweep.dir}/.submitit/%j
    timeout_min: 60
    cpus_per_task: 28
    gpus_per_task: 1   # each GPU must have its separate task for jax
    gres: gpu:${n_gpus}
    tasks_per_node: ${n_gpus}
    mem_gb: ~
    nodes: 1
    name: ${hydra.job.name}
    _target_: hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher
    partition: compute
    qos: null
    comment: testing_slurmit_launcher
    additional_parameters: {
      "gpu-bind": "closest",
      "wait-all-nodes": "1",
      "time": "7-00:00:00",
      "exclusive": "",
    }
    srun_args:
      - "--kill-on-bad-exit=1"
    setup:
      - export NCCL_CROSS_NIC=0
      - export NCCL_SOCKET_NTHREADS=16
      - export NCCL_DEBUG=WARN
      - export NCCL_CUMEM_ENABLE=0
      - export NCCL_IB_SPLIT_DATA_ON_QPS=0
      - export NCCL_IB_QPS_PER_CONNECTION=16
      - export NCCL_IB_GID_INDEX=3
      - export NCCL_IB_TC=41
      - export NCCL_IB_SL=0
      - export NCCL_IB_TIMEOUT=22
      - export NCCL_NET_PLUGIN=none
      - export NCCL_SOCKET_IFNAME=eth0
      - export NCCL_IGNORE_CPU_AFFINITY=1
      - export NCCL_IB_HCA="=mlx5_0,mlx5_1,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16,mlx5_17"
      - export HCOLL_ENABLE_MCAST_ALL=0
      - export coll_hcoll_enable=0
      - export UCX_TLS=tcp
      - export UCX_NET_DEVICES=eth0
      - export RX_QUEUE_LEN=8192
      - export IB_RX_QUEUE_LEN=8192
      - export OMPI_MCA_coll=^hcoll
      - export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nfs-gpu/xlstm/miniforge3/envs/python_3.11_jax_0.4.34_cuda_12.6/cuda-compat
      - export TOKENIZERS_PARALLELISM=0
      - export JAX_COMPILATION_CACHE_DIR=/nfs-gpu/xlstm/data/jax_compilation_cache

```

Note that the path to the `cuda-compat` package and the
`JAX_COMPILATION_CACHE_DIR` must be adjusted
by you!

Here, you can supply (or rather overwrite in an experment file!) the things you would normally put into a SLURM run
script. To use the `submitit_launcher` you have to execute the following with the conda env
activated that you want to use for the experiment:

```PYTHONPATH=. python scripts/training/train_with_hydra.py --multirun hydra/launcher=slurm_launcher +experiment={YOUR_EXPERIMENT_FILE}```

So the only thing you have to add is ```--multirun hydra/launcher=slurm_launcher``` (and to overwrite SLURM-specific parameters as the number of nodes, for example, in your experiment file).

Note that you can also add this to your experiment file. See,
for example, the experiment `synthetic_experiment_slurm.yaml`,
where the lines

```yaml
  - override /hydra/launcher: slurm_launcher

hydra:
  mode: MULTIRUN
```

will make Hydra use the `submitit_launcher`



## How to Resume an Experiment?

To resume an experiment you need to know the path to the output
folder of the experiment. You then execute

```python
python scripts/traininig/get_cli_command_to_resume_training.py --resume_from_folder=PATH_TO_RUN
```

If you want to use SLURM, set the flag `--use_slurm`.
Not that this is only required if
the original run was executed with SLURM by way of the CLI override
`--multirun hydra/launcher=slurm_launcher` and
not by way of experiment file. If it was specified in the experiment file, SLURM is used anyway.

If you want to use the latest checkpoint, you don't need to supply anything but if you want to use a
specific checkpoint, use `--checkpoint_step=X` to use checkpoint X.

New hydra overrides can be supplied by way of `--new_overrides=STRING_OF_OVERRIDES`.
Example of such a string for more training steps, a different learning rate and
a different logging frequency would be
`--new_overrides="num_train_steps=20000 lr=0.0001 logger.log_every_n_steps=10"`, that is,
the format is exactly as you would supply for CLI overrides.

Executing `get_cli_command_to_resume_training` will return a string of the command that you
have to execute to continue the training run.

A full example to call is

```python
python scripts/training/get_cli_command_to_resume_training.py --resume_from_folder=PATH_TO_RUN --use_slurm --checkpoint_step=95000 --new_overrides="num_train_steps=20000 lr=0.0001 logger.log_every_n_steps=10"
```

What the script does is to look in the specified folder and obtain the overrides (including the experiment file)
that were used to start the previous run. It then compiles a new command from these, which, for the current
example would look something like this:

`PYTHONPATH=. python scripts/training/resume_training_with_hydra.py        -m hydra/launcher=slurm_launcher +experiment=synthetic_experiment_slurm +resume_from_folder=PATH_TO_RUN        +checkpoint_step=95000 num_train_steps=20000 lr=0.0001 logger.log_every_n_steps=10`

This is the actual command you have to run to resume the experiment. Keep in mind that you can still
use any other CLI overrides at this point so using them in the previous step is not strictly necessary.
