xlstm_jax.dataset.synthetic_dataloading
=======================================

.. py:module:: xlstm_jax.dataset.synthetic_dataloading

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

   This file is a modified version of the file input_pipeline_interface.py from the maxtext project,
   see https://github.com/AI-Hypercomputer/maxtext/MaxText/input_pipeline/input_pipeline_interface.py.

   Synthetic Data Iterator



Classes
-------

.. autoapisummary::

   xlstm_jax.dataset.synthetic_dataloading.SyntheticDataIterator


Module Contents
---------------

.. py:class:: SyntheticDataIterator(config, mesh)

   Creates a synthetic data iterator for performance testing work.

   :param config: Configuration for the synthetic data.
   :param mesh: Global device mesh for sharding.


   .. py:attribute:: config


   .. py:attribute:: mesh


   .. py:attribute:: num_batches


   .. py:attribute:: data_generator


   .. py:attribute:: _step_counter
      :value: 0



   .. py:property:: dataset_size
      :type: int



   .. py:method:: raw_generate_synthetic_data(config)
      :staticmethod:


      Generates a single batch of synthetic data



