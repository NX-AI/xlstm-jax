xlstm_jax.dataset.input_pipeline_interface
==========================================

.. py:module:: xlstm_jax.dataset.input_pipeline_interface

.. autoapi-nested-parse::

   Copyright 2023 Google LLC

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   ---

   Input pipeline



Attributes
----------

.. autoapisummary::

   xlstm_jax.dataset.input_pipeline_interface.LOGGER
   xlstm_jax.dataset.input_pipeline_interface.GRAIN_AVAILABLE
   xlstm_jax.dataset.input_pipeline_interface.DataIterator


Functions
---------

.. autoapisummary::

   xlstm_jax.dataset.input_pipeline_interface.get_process_loading_data
   xlstm_jax.dataset.input_pipeline_interface.create_data_iterator
   xlstm_jax.dataset.input_pipeline_interface.create_mixed_data_iterator


Module Contents
---------------

.. py:data:: LOGGER

.. py:data:: GRAIN_AVAILABLE
   :value: True


.. py:data:: DataIterator

.. py:function:: get_process_loading_data(config, mesh)

   Get list of processes loading data.

   :param config: Config of the dataset to load.
   :param mesh: Global device mesh for sharding.

   :returns: List of process indices that will load real data.


.. py:function:: create_data_iterator(config, mesh)

.. py:function:: create_mixed_data_iterator(configs, mesh, dataset_weights = None)

   Create a data iterator that mixes multiple datasets.

   Each individual dataset will be loaded, and the iterator will return batches where each batch element is from
   one of the datasets. The frequency of each dataset is determined by the dataset_weights.

   :param configs: List of DataConfig objects, determining the datasets to load.
   :param mesh: JAX mesh object. Used to distribute the data over multiple devices.
   :param dataset_weights: Mixing weights for the datasets. If None, all datasets will have equal weight.

   :returns: DataIterator object that can be used to iterate over the mixed dataset.


