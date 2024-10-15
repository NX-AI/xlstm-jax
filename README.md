# xLSTM

## Table of Contents

- [Configuring Experiments with Hydra](#configuring-experiments-with-hydra)
    - [Configuration Structure](#configuration-structure)
    - [How to run experiments](#how-to-run-experiments)

# Configuring Experiments with Hydra

This project uses Hydra for managing configurations. Hydra is a framework that simplifies the process of configuring complex applications by allowing you to compose configurations dynamically.

## Configuration Structure

The configuration files are organized in the `configs` directory. For now, the structure is as follows


- [`configs/`]: Contains all configuration files.
  - `config.yaml`: The main configuration file. In there, the default sub-configurations are specified by way of the
  `defaults` list. Additionaly, global variables are defined.
  The sub-modules are in subfolders:

  TODO: fill up once ready


## How to Run Experiments

The principal entry point for hydra is the script `scripts/train_with_hydra.py`. On the command line, you can start a run locally by executing

```python scripts/train_with_hydra.py```.

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
  - data: synthetic
  - model: mLSTM120M
  - _self_


# General hyperparameters. Will be put in their respective config modules once they are created.

# device. cpu or gpu
device: cpu
```

the following will happen:

1. The first line reads `- config_schema`. This is related to the use of
structured configs, which will be detailed elsewhere (TODO). As with all entries
in the defaults list, Hydra will look for parameters in the file and append it
to the compiled configuration.

2. `- parallel: synthetic` means that Hydra will now look in the `parallel` subfolder for the `synthetic.yaml`
 file and append all parameters in that file to the compiled list of parameters under the key `parallel`.
3. `- data: synthetic` means that Hydra will now look in the `data` subfolder for the `synthetic.yaml`
 file and append all parameters in that file to the compiled list of parameters under the key `data`.
4. `- model: mLSTM120M` means that Hydra will now look in the `model` subfolder for the `mLSTM120M.yaml`
 file and append all parameters in that file to the compiled list of parameters under the key `model`.
5. `_self_` means that all parameters in this file itself (that is, the `config.yaml`) will be inserted into the
compiled config.

Note that all values that come later in the defaults list **overwrite** potential values that have been in the
configuration before.

Once Hydra has processed this file, it changes the current working directory as specified in the configs
and executes
`main_train(cfg)` in that directory, where `cfg` is the compiled configuration.
All top level parameters are in `cfg`, as for example, `cfg.device` and the
others are in their respective fields, for example, `cfg.parallel.data_axis_name`.

This is where Hydra stops. Everything coming after in the `main_train` function is instantiated manually for now.


### Remarks
- It is possible to overwrite every parameter from the command line. So you can, for example, execute
```python scripts/train_with_hydra device=gpu model.num_heads=42```
to substitute `cpu`
from the `config.yaml` with `gpu` and to substitute 42 for whatever value for `num_heads` was given in `mLSTM120M.yaml` file. This is probably not the approach you want to use to start experiments though.


### Using Experiment Files
(see https://hydra.cc/docs/patterns/configuring_experiments/)

A nicer since much more reproducible way to start experiments with Hydra is to use experiment files. In these files
you can specify all parameters and config groups that you want to change. Let us take the `experiment/train_mLSTM7B_slimpajama6b.yaml` as an example. That file reads

```yaml
# @package _global_
defaults:
  - override /parallel: mLSTM7B
  - override /model: mLSTM7B
  - override /data: slimpajama_6B_local_ds
  - override /optimizer: adamw
  - _self_

# specify the deltas from the defaults:
task_name: mLSTM7B_slimpajama6b_example_experiment
batch_size_per_device: 8
context_length: 2048
num_epochs: 1000
num_train_steps: 95_000
lr: 5e-4

trainer:
  gradient_accumulate_steps: 1
```

To use this file, you execute the following:

```python scripts/train_with_hydra.py +experiment=train_mLSTM7B_slimpajama6b.yaml ```

Note the + before experiment, that's not a typo!
Hydra now checks the defaults list of the experiment file and replaces the respective fields from the general
`config.yaml`. That is, instead of looking for the `synthetic.yaml` in the data subfolder it now looks for the
`slimpajama_6B_local_ds.yaml` file and uses those paramters when compiling the configuration. In addition, you can
overwrite any parameter with your own values. This goes for all parameters at every level of the config hierarchy.
In this example, the value in `cfg.trainer.gradient_accumulation_steps`
 is overwritten with 1.
By using experiment files, it becomes easy to specify and reproduce experiments.

### Type Checking of the Configurations

One benefit of using the structured configs and dataclasses as basis for the configuration is that type checking
becomes available. This happens when compiling the `cfg`. See also the structured configs section (TODO).

To test whether you have supplied correctly typed parameters in your experiment files, you can execute

```python scripts/check_config.py +experiment={YOUR_EXPERIMENT_FILE}```

This function just compiles and logs your compiled config. But it's a good way to check if you have supplied a
working experiment file before starting a job on a cluster, for example.


## How to run experiments on a SLURM cluster
To run your script on a SLURM cluster you can use the `submitit_launcher`, which is installed for Hydra. The
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
    gpus_per_node: 8
    tasks_per_node: 8
    mem_gb: ~
    nodes: 1
    name: ${hydra.job.name}
    _target_: hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher
    partition: compute
    qos: null
    comment: testing_slurmit_launcher
    additional_parameters: {}
    setup:
      - gpu-bind=none
      - wait-all-nodes=1
      - time=2-00:00:00
      - output=${hydra.sweep.dir}/.submitit/%j.out
      - exclusive
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
```

Here, you can supply (or rather overwrite in an experment file!) the things you would normally put into a SLURM run
script. To use the `submitit_launcher` you have to execute the following with the conda env
activated that you want to use for the experiment:

```python scripts/train_with_hydra.py --multirun hydra/launcher=slurm_launcher +experiment={YOUR_EXPERIMENT_FILE}```

So the only thing you have to add is ```--multirun hydra/launcher=slurm_launcher``` (and to overwrite SLURM-specific parameters as the number of nodes, for example, in your experiment file).
