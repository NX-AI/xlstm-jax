xlstm_jax.models.shared.init
============================

.. py:module:: xlstm_jax.models.shared.init


Attributes
----------

.. autoapisummary::

   xlstm_jax.models.shared.init.InitDistribution
   xlstm_jax.models.shared.init.InitFnName


Functions
---------

.. autoapisummary::

   xlstm_jax.models.shared.init.small_init
   xlstm_jax.models.shared.init.wang_init
   xlstm_jax.models.shared.init.create_common_init_fn
   xlstm_jax.models.shared.init._dist_from_stddev
   xlstm_jax.models.shared.init.uniform_init


Module Contents
---------------

.. py:data:: InitDistribution

.. py:data:: InitFnName

.. py:function:: small_init(dim, distribution = 'normal')

   Create initializer of Nguyen et al. (2019).

   Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
   The initializer creates an array with values according to the method described in:
   "Transformers without Tears: Improving the Normalization of Self-Attention", Nguyen, T. & Salazar, J. (2019).
   The array values are sampled with a standard deviation of sqrt(2 / (5 * dim)).

   :param dim: Feature dimensionality to use in the initializer.
   :param distribution: The distribution to sample from. Supported are normal, truncated normal, and uniform.

   :returns: Initializer function following the above described method.


.. py:function:: wang_init(dim, num_blocks, distribution = 'normal')

   Create Wang initializer.

   Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
   Commonly used for the output layers of residual blocks. The array values are sampled with a standard deviation
   of 2 / num_blocks / sqrt(dim).

   :param dim: Feature dimensionality to use in the initializer.
   :param num_blocks: Number of layers / blocks in the model.
   :param distribution: The distribution to sample from. Supported are normal, truncated normal, and uniform.

   :returns: Initializer function of the wang init.


.. py:function:: create_common_init_fn(fn_name, dim, num_blocks, distribution = 'normal')

   Create common initializer function.

   Allows to create different types of initializers with a single function call.

   :param fn_name: Name of the initializer function to create. Supported are "small" (:func:`~small_init`),
                   "wang" (:func:`~wang_init`), "wang2" (:func:`~wang_init` with 2x block num), and
                   "zeros" (zero initializer).
   :param dim: Feature dimensionality to use in the initializer.
   :param num_blocks: Number of layers / blocks in the model.
   :param distribution: The distribution to sample from. Supported are normal, truncated normal, and uniform.

   :returns: Initializer function of the specified type.


.. py:function:: _dist_from_stddev(stddev, distribution)

   Create initializer with specified standard deviation and distribution.

   The distribution has a zero mean and specified standard deviation.

   :param stddev: The standard deviation of the distribution.
   :param distribution: The distribution to sample from. Supported are normal, truncated normal, and uniform.

   :returns: Initializer function that samples the array value from the specified distribution with the given standard
             deviation.


.. py:function:: uniform_init(min_val, max_val)

   Create uniform initializer.

   :param min_val: Minimum value of the uniform distribution.
   :param max_val: Maximum value of the uniform distribution.

   :returns: An initializer function which samples values randomly between min_val and max_val.


