xlstm_jax.distributed.xla_utils
===============================

.. py:module:: xlstm_jax.distributed.xla_utils


Functions
---------

.. autoapisummary::

   xlstm_jax.distributed.xla_utils.simulate_CPU_devices
   xlstm_jax.distributed.xla_utils.set_XLA_flags


Module Contents
---------------

.. py:function:: simulate_CPU_devices(device_count = 8)

   Simulate a CPU with a given number of devices.

   :param device_count: The number of devices to simulate.


.. py:function:: set_XLA_flags()

   Set XLA flags for better performance.

   For performance flags, see https://jax.readthedocs.io/en/latest/gpu_performance_tips.html and
   https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/GPU_performance.md.


