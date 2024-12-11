xlstm_jax.distributed.mesh_utils
================================

.. py:module:: xlstm_jax.distributed.mesh_utils


Functions
---------

.. autoapisummary::

   xlstm_jax.distributed.mesh_utils.initialize_mesh


Module Contents
---------------

.. py:function:: initialize_mesh(parallel_config, device_array = None, init_distributed_on_slurm = True)

   Initialize the mesh for parallel training.

   :param parallel_config: A dictionary containing the parallelization parameters.
   :param device_array: A numpy array containing the device structure. If None, all global devices are used.
   :param init_distributed_on_slurm: Whether to initialize the JAX distributed system, i.e. multiprocess training,
                                     if SLURM environment variables are present. If False, the JAX distributed system is not initialized.

   :returns: The initialized mesh.


