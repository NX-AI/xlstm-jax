# Installation

## Clone repository
To clone the repository, execute
```bash
git clone https://github.com/NX-AI/xlstm-jax.git
cd xlstm-jax
```

## Conda environment
We use conda for managing dependencies.
You can create a new conda environment and install the required dependencies by running the following command:
```bash
conda env create -f envs/environment_python_3.11_jax_0.4.34_cuda_12.6.yml
```
or
```bash
mamba env create -f envs/environment_python_3.11_jax_0.4.34_cuda_12.6.yml
```
if you use mamba.

## Pip install
When creation of the conda env has finished pip install the repository with
```bash
pip install -e .
```

## Testing installation
You can run one of the tests to verify that your installation works correctly, for example,
```bash
pytest tests/config/test_equivalence_hydra_nonhydra.py
```
or to test it with a GPU
```bash
CUDA_VISIBLE_DEVICES=0 pytest tests/config/test_equivalence_hydra_nonhydra.py
```
