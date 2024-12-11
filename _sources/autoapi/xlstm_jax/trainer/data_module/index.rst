xlstm_jax.trainer.data_module
=============================

.. py:module:: xlstm_jax.trainer.data_module


Attributes
----------

.. autoapisummary::

   xlstm_jax.trainer.data_module.DataIterator


Classes
-------

.. autoapisummary::

   xlstm_jax.trainer.data_module.DataloaderModule


Module Contents
---------------

.. py:data:: DataIterator

.. py:class:: DataloaderModule

   .. py:attribute:: train_dataloader
      :type:  DataIterator | None
      :value: None



   .. py:attribute:: val_dataloader
      :type:  DataIterator | dict[str, DataIterator] | None
      :value: None



   .. py:attribute:: test_dataloader
      :type:  DataIterator | dict[str, DataIterator] | None
      :value: None



