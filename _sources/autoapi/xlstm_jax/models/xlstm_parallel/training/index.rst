xlstm_jax.models.xlstm_parallel.training
========================================

.. py:module:: xlstm_jax.models.xlstm_parallel.training


Functions
---------

.. autoapisummary::

   xlstm_jax.models.xlstm_parallel.training.print_metrics
   xlstm_jax.models.xlstm_parallel.training.get_num_params
   xlstm_jax.models.xlstm_parallel.training.loss_fn
   xlstm_jax.models.xlstm_parallel.training.train_step
   xlstm_jax.models.xlstm_parallel.training.get_train_step_fn
   xlstm_jax.models.xlstm_parallel.training.init_xlstm
   xlstm_jax.models.xlstm_parallel.training.flatten_dict
   xlstm_jax.models.xlstm_parallel.training.tabulate_params


Module Contents
---------------

.. py:function:: print_metrics(metrics, title = None)

   Prints metrics with an optional title.

   :param metrics: A dictionary with metric names as keys and a tuple of (sum, count) as values.
   :param title: An optional title for the metrics.


.. py:function:: get_num_params(state)

   Calculate the number of parameters in the model.

   :param state: The current training state.

   :returns: The number of parameters in the model.


.. py:function:: loss_fn(params, apply_fn, batch, rng, config)

.. py:function:: train_step(state, metrics, batch, config, gradient_accumulate_steps = 1)

.. py:function:: get_train_step_fn(state, batch, mesh, config, gradient_accumulate_steps = 1)

.. py:function:: init_xlstm(config, mesh, rng, input_array, optimizer)

.. py:function:: flatten_dict(d)

   Flattens a nested dictionary.


.. py:function:: tabulate_params(state)

   Prints a summary of the parameters represented as table.

   :param state: The current training state.


