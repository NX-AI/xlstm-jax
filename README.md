# xLSTM-jax
[Paper](https://arxiv.org/abs/2405.04517) | [Model](https://huggingface.co/NX-AI/xLSTM-7b) | [Docs](https://xlstm-jax.readthedocs.io/) | [Citation](#citation)

The official repository for the xLSTM model and training code in JAX.


## About
This repository contains the code to train and evaluate xLSTM on language modelling using JAX.
xLSTM is a new Recurrent Neural Network architecture based on ideas of the original LSTM.
Through Exponential Gating with appropriate normalization and stabilization techniques and a new Matrix Memory it overcomes the limitations of the original LSTM and shows promising performance on Language Modeling when compared to Transformers or State Space Models.

This code base supports a 3D parallelization strategy and is optimized for training on large-scale distributed systems with hundreds or thousands of GPUs.
We developed performant [triton](https://triton-lang.org/main/index.html) kernels for xLSTM, resulting in much faster training and inference.
Our kernels are implemented in [this repository](https://github.com/NX-AI/mlstm_kernels) and included as a submodule.

## xLSTM-7B
We used xlstm-jax to train a 7B parameter xLSTM model on 256 H100 GPUs.
The xLSTM-7B shows competitive performance on common benchmarks compared to other 7B LLMs, while achieving much better token throughput for larger sequence lengths.
![xLSTM Figure](https://raw.githubusercontent.com/NX-AI/xlstm/refs/heads/main/res/xlstm_7b_poster.svg)


## Documentation
The documentation is available at [https://xlstm-jax.readthedocs.io/](https://xlstm-jax.readthedocs.io/), covering
- [Installation](https://xlstm-jax.readthedocs.io/en/latest/installation.html)
- [Dataset preparation](https://xlstm-jax.readthedocs.io/en/latest/dataset_preparation.html)
- [Training large language models](https://xlstm-jax.readthedocs.io/en/latest/example_training.html)
- [Parallelization strategies](https://xlstm-jax.readthedocs.io/en/latest/distributed_training.html)
- [Configuring experiments with Hydra](https://xlstm-jax.readthedocs.io/en/latest/configuration_with_hydra.html)


## Contributing
Contributions to this repository are welcome.
- If you find bugs or have suggestions for improvements, please [open an issue](https://github.com/NX-AI/xlstm-jax/issues) with a detailed description of the problem or suggestion.
- If you want to contribute, please [open a pull request](https://github.com/NX-AI/xlstm-jax/pulls) with a detailed description of the changes you made.
- More general questions and discussions can be posted in the [Discussions section](https://github.com/NX-AI/xlstm-jax/discussions).


## Citation
If you use this codebase, or otherwise find our work valuable, please cite the xLSTM paper and this repository:
```
@article{xlstm,
  title={xLSTM: Extended Long Short-Term Memory},
  author={Beck, Maximilian and P{\"o}ppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, G{\"u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
  journal={arXiv preprint arXiv:2405.04517},
  year={2024}
}
```

```
@misc{xlstm-jax,
  title={xLSTM-jax},
  author={NXAI GmbH},
  year={2024},
  url={https://github.com/NX-AI/xlstm-jax/},
}
```

## License
This project is licensed under the NXAI Community License, please see [LICENSE](https://github.com/NX-AI/xlstm-jax/blob/main/LICENSE).
