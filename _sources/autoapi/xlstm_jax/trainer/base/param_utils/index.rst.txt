xlstm_jax.trainer.base.param_utils
==================================

.. py:module:: xlstm_jax.trainer.base.param_utils


Functions
---------

.. autoapisummary::

   xlstm_jax.trainer.base.param_utils.is_partitioned
   xlstm_jax.trainer.base.param_utils.get_num_params
   xlstm_jax.trainer.base.param_utils.tabulate_params
   xlstm_jax.trainer.base.param_utils.get_grad_norms
   xlstm_jax.trainer.base.param_utils.get_param_norms
   xlstm_jax.trainer.base.param_utils.get_sharded_norm_logits
   xlstm_jax.trainer.base.param_utils.get_sharded_global_norm
   xlstm_jax.trainer.base.param_utils.get_param_mask_fn


Module Contents
---------------

.. py:function:: is_partitioned(x)

   Check if an object is a Partitioned.

   Parameters that are sharded via FSDP, PP, or TP, are represented as Partitioned objects. Parameters that are
   replicated are represented as regular jax.Array objects. This function can be used in the context of PyTrees as
   is_leaf argument in a tree map to consider Partitioned objects as leaves instead of traversing them. Note that
   in that case, JAX Arrays of standard replicated parameters and all other normal leaves are still considered leaves.

   :param x: The object to check.

   :returns: Whether the object is a Partitioned.


.. py:function:: get_num_params(params)

   Calculates the number of parameters in a PyTree.


.. py:function:: tabulate_params(state, show_weight_decay = False, weight_decay_exclude = None, weight_decay_include = None)

   Prints a summary of the parameters represented as table.

   :param state: The TrainState or the parameters as a dictionary.
   :param show_weight_decay: Whether to show the weight decay mask.
   :param weight_decay_exclude: List of regex patterns to exclude from weight decay. See optimizer config for more
                                information.
   :param weight_decay_include: List of regex patterns to include in weight decay. See optimizer config for more
                                information.

   :returns: The summary table as a string.
   :rtype: str


.. py:function:: get_grad_norms(grads, return_per_param = False)

   Determine the gradient norms.

   :param grads: The gradients as a PyTree.
   :param return_per_param: Whether to return the gradient norms per parameter or only the global norm.

   :returns: A dictionary containing the gradient norms.
   :rtype: dict


.. py:function:: get_param_norms(params, return_per_param = False)

   Determine the parameter norms.

   :param params: The parameters as a PyTree.
   :param return_per_param: Whether to return the parameter norms per parameter or only the global norm.

   :returns: A dictionary containing the parameter norms.
   :rtype: dict


.. py:function:: get_sharded_norm_logits(x)

   Calculate the norm of a sharded parameter or gradient.

   :param x: The parameter or gradient.

   :returns: The norm logit, i.e. the squared norm.
   :rtype: jax.Array


.. py:function:: get_sharded_global_norm(x)

   Calculate the norm of a sharded PyTree.

   :param x: The PyTree. Each leaf should be a jax.Array or nn.Partitioned.

   :returns: The global norm and the norm per leaf.
   :rtype: tuple[jax.Array, PyTree]


.. py:function:: get_param_mask_fn(exclude, include = None)

   Returns a function that generates a mask, which can for instance be used for weight decay.

   :param exclude: List of strings to exclude.
   :type exclude: Sequence[str]
   :param include: List of strings to include. If None, all parameters except those in exclude are
                   included.
   :type include: Sequence[str]

   :returns: Function that generates a mask.
   :rtype: Callable[[PyTree], PyTree]


