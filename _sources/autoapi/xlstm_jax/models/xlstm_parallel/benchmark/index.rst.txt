xlstm_jax.models.xlstm_parallel.benchmark
=========================================

.. py:module:: xlstm_jax.models.xlstm_parallel.benchmark


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.benchmark.init_mesh
   xlstm_jax.models.xlstm_parallel.benchmark.create_batch
   xlstm_jax.models.xlstm_parallel.benchmark.benchmark_model


Module Contents
---------------

.. py:function:: init_mesh(data_axis_size = -1, fsdp_axis_size = 1, pipeline_axis_size = 1, model_axis_size = 1, data_axis_name = 'dp', fsdp_axis_name = 'fsdp', pipeline_axis_name = 'pp', model_axis_name = 'tp')

.. py:function:: create_batch(batch_size, context_length, vocab_size, rng, mesh, config)

.. py:function:: benchmark_model(config, data_axis_size = -1, fsdp_axis_size = 1, pipeline_axis_size = 1, model_axis_size = 1, seed = 42, gradient_accumulate_steps = 1, batch_size_per_device = 32, optimizer = None, log_dir = None, log_num_steps = 1, log_skip_steps = 5, num_steps = 100)

