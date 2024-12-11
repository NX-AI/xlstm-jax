xlstm_jax.main_train
====================

.. py:module:: xlstm_jax.main_train


Functions
---------

.. autoapisummary::

   xlstm_jax.main_train.main_train


Module Contents
---------------

.. py:function:: main_train(cfg, checkpoint_step = None, load_dataloaders = True, load_optimizer = True)

   The main training function. This function initializes the mesh, data iterators,
     model config, and trainer and then starts training. Can be optionally started
     from a checkpoint, in which case the training state is loaded from the checkpoint
     with the supplied step index.

   In order to see error logs in our custom logger, we use the with_error_handling
     decorator.

   :param cfg: The full configuration.
   :param checkpoint_step: Step index of checkpoint to be loaded.
                           Defaults to None, in which case training starts from scratch.
   :type checkpoint_step: optional
   :param load_dataloaders: Whether to load the data loaders. Defaults to True.
   :type load_dataloaders: optional
   :param load_optimizer: Whether to load the optimizer. Defaults to True.
   :type load_optimizer: optional

   :returns: The final metrics of the training.


