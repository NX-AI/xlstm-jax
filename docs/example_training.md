(training-LLMs)=
# Training large language models in xlstm-jax
Here we explain how to train large language models (LLMs) in `xlstm-jax`.
We provide examples for training xLSTM and LLama on the DCLM and Slimpajama-627B datasets,
using our trainer and distributed models implemented in jax.
The training scripts are located in the `scripts/training` directory. It is currently required to run the scripts from the root directory of the repository and adding `PYTHONPATH=.` to the run command.
While the recommended way to train models is using Hydra and Slurm, we will start with a simpler entry point and describe training with Hydra later.

## Training without Hydra
To train an xLSTM model without Hydra on the SlimPajama dataset, you need to specify the model configuration by way of the config dataclasses.
For example, you can use the following command:
```bash
PYTHONPATH=. python scripts/training/run_train_slimpajama.py --log_dir=<log_dir> --model=<model>
```
where
- `<log_dir>` is the directory where the logs and checkpoints will be saved. Note that the checkpoints including model weights are quite large, so make sure you have enough disk space and fast I/O.
- `<model>` indicates one of the default configurations provided in the beginning of the script, that is one of [`120M`, `165M`, `165M_v1`, `1.3B`, `1.3B_v1`, `7B`, `7B_v1` ]. The name indicates the number of parameters and whether the mLSTM from the original paper is used or the version named "v1" that we used for training our 7B parameter model.
- By default, the script uses a smaller subset [SlimPajama-6B](https://huggingface.co/datasets/DKYoon/SlimPajama-6B/tree/main).
Please use the flag `--use_full_dataset` for training on the full dataset [SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B).

Similarly, a LLama baseline can be trained with the following command:
```
PYTHONPATH=. python scripts/training/run_train_llama_slimpajama.py --log_dir=<log_dir> --model=<model>
```
where `<model>` should be `1.3B` or `165M`, indicating a 1.3 billion or 165 million parameter model configuration, defined in the script.


## Training with a Hydra configuration
Our recommended way to train models is by using Hydra for hyperparameter configuration.
Please see {ref}`configuring-experiments-with-hydra` for more details on the Hydra configuration.

### Default configurations
Many default configurations are already provided in the `configs` directory.
For example, the subfolder `configs/model` contains several xLSTM and Llama model configs,
and `configs/data` contains configs for the DCLM, Slimpajama and other datasets.
These default configurations can be left untouched if you want to train one of the default model architectures on existing datasets.

### Experiment configuration
The main config files you need to interact with are the experiment config files located in the `configs/experiment` directory.
To train a model with a specific experiment configuration, you can use the script `train_with_hydra.py` with one of the experiment configs.
For example, you can train a mLSTM-v1 model with 165M parameters on the Slimpajama-627B dataset with the following command:
```bash
PYTHONPATH=. python scripts/training/train_with_hydra.py +experiment=train_mLSTMv1_165M_slimpajama627b
```
This will use the configs for data, model, parallel, optimizer, etc. as specified by default and overridden in the `train_mLSTMv1_165M_slimpajama627b.yaml` file.
