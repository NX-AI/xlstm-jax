# Installation

## Repository Installation
To install the package, clone the repository and install the package using pip:
```bash
git clone --recurse-submodules git@github.com:NX-AI/xlstm-jax.git
pip install -e .
```

This repository relies on custom Triton kernels hosted in https://github.com/NX-AI/mlstm_kernels and included as a submodule.
In order to install or update the kernels manually run `git submodule init` or `git submodule update`.

## Conda environment
We use conda for managing dependencies.
You can create a new conda environment and install the required dependencies by running the following command:
```bash
conda env create -f envs/environment_jax_0.4.32_gpu_python_3.11.yaml
```
