xlstm_jax.trainer.optimizer.ademamix
====================================

.. py:module:: xlstm_jax.trainer.optimizer.ademamix

.. autoapi-nested-parse::

   Adapted from Apple's official AdeMAMix implementation:
   https://github.com/apple/ml-ademamix/blob/main/optax/ademamix.py

   TODO: Check license and add it here.



Classes
-------

.. autoapisummary::

   xlstm_jax.trainer.optimizer.ademamix.ScaleByAdemamixState


Functions
---------

.. autoapisummary::

   xlstm_jax.trainer.optimizer.ademamix.alpha_scheduler
   xlstm_jax.trainer.optimizer.ademamix.beta3_scheduler
   xlstm_jax.trainer.optimizer.ademamix.ademamix
   xlstm_jax.trainer.optimizer.ademamix.scale_by_ademamix
   xlstm_jax.trainer.optimizer.ademamix.tree_cast
   xlstm_jax.trainer.optimizer.ademamix.tree_zeros_like
   xlstm_jax.trainer.optimizer.ademamix.tree_update_moment
   xlstm_jax.trainer.optimizer.ademamix.tree_update_moment_per_elem_norm
   xlstm_jax.trainer.optimizer.ademamix.tree_bias_correction


Module Contents
---------------

.. py:function:: alpha_scheduler(alpha, alpha_start = 0.0, warmup = 0)

   Linear scheduler for the mixing coefficient alpha in AdEMAMix.

   :param alpha: Final value of alpha.
   :param alpha_start: Initial value of alpha.
   :param warmup: Number of steps for the warmup phase. Often set equal to the number of training steps.

   :returns: A scheduler function that takes a step and returns the value of alpha.


.. py:function:: beta3_scheduler(beta_end, beta_start = 0.0, warmup = 0)

   Linear scheduler for the EMA parameter beta3 in AdEMAMix.

   :param beta_end: Final value of beta3.
   :param beta_start: Initial value of beta3. Often set equal to beta1.
   :param warmup: Number of steps for the warmup phase. Often set equal to the number of training steps.

   :returns: A scheduler function that takes a step and returns the value of beta3.


.. py:class:: ScaleByAdemamixState

   Bases: :py:obj:`NamedTuple`


   State for the AdEMAMix algorithm.


   .. py:attribute:: count
      :type:  chex.Array

      Step counter for the first momentum and adaptive learning rate.


   .. py:attribute:: count_m2
      :type:  chex.Array

      Step counter for the slower momentum.


   .. py:attribute:: m1
      :type:  optax._src.base.Updates

      Fast EMA.


   .. py:attribute:: m2
      :type:  optax._src.base.Updates

      Slow EMA.


   .. py:attribute:: nu
      :type:  optax._src.base.Updates

      Second moment estimate.


.. py:function:: ademamix(lr, b1 = 0.9, b2 = 0.999, b3 = 0.9999, alpha = 5.0, b3_scheduler = None, alpha_scheduler = None, eps = 1e-08, eps_root = 0.0, weight_decay = 0.0, mu_dtype = None, mask = None)

   AdEMAMix.

   :param lr: A global scaling factor, either fixed or evolving along
              iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
   :param b1: Exponential decay rate to track the fast EMA.
   :param b2: Exponential decay rate to track the second moment of past gradients.
   :param b3: Exponential decay rate to track the slow EMA.
   :param alpha: Mixing coeficient use for the linear combination of the fast and slow EMAs.
   :param b3_scheduler: an optional scheduler function, given a timestep, returns the
                        value of b3. Use `beta3_scheduler(b3,b1,T_b3)` to follow the AdEMAMix paper.
   :param alpha_scheduler: an optional scheduler function, given a timestep, returns the
                           value of alpha. Use `alpha_scheduler(alpha,0,T_alpha)` to follow the
                           AdEMAMix paper.
   :param eps: A small constant applied to denominator outside the square root
               (as in the Adam paper) to avoid dividing by zero when rescaling.
   :param eps_root: A small constant applied to denominator inside the square root (as
                    in RMSProp), to avoid dividing by zero when rescaling. This is needed for
                    instance when computing (meta-)gradients through Adam.
   :param mu_dtype: Optional `dtype` to be used for the first order accumulator; if
                    `None` then the `dtype` is inferred from `params` and `updates`.
   :param weight_decay: Strength of the weight decay regularization. Note that this
                        weight decay is multiplied with the learning rate. This is consistent
                        with other frameworks such as PyTorch, but different from
                        (Loshchilov et al., 2019) where the weight decay is only multiplied with
                        the "schedule multiplier", but not the base learning rate.
   :param mask: A tree with same structure as (or a prefix of) the params PyTree,
                or a Callable that returns such a pytree given the params/updates.
                The leaves should be booleans, `True` for leaves/subtrees you want to
                apply the weight decay to, and `False` for those you want to skip. Note
                that the Adam gradient transformations are applied to all parameters.

   :returns: The corresponding `GradientTransformation`.


.. py:function:: scale_by_ademamix(b1, b2, b3, alpha, b3_scheduler, alpha_scheduler, eps = 1e-08, eps_root = 0.0, mu_dtype = None)

   Scales updates by the AdEMAMix algorithm.

   :param b1: Exponential decay rate to track the fast EMA.
   :param b2: Exponential decay rate to track the second moment of past gradients.
   :param b3: Exponential decay rate to track the slow EMA.
   :param alpha: Mixing coeficient use for the linear combination of the fast and slow EMAs.
   :param b3_scheduler: an optional scheduler function, given a timestep, returns the
                        value of b3. Use `beta3_scheduler(b3,b1,T_b3)` to follow the AdEMAMix paper.
   :param alpha_scheduler: an optional scheduler function, given a timestep, returns the
                           value of alpha. Use `alpha_scheduler(alpha,0,T_alpha)` to follow the
                           AdEMAMix paper.
   :param eps: A small constant applied to denominator outside the square root
               (as in the Adam paper) to avoid dividing by zero when rescaling.
   :param eps_root: A small constant applied to denominator inside the square root (as
                    in RMSProp), to avoid dividing by zero when rescaling. This is needed for
                    instance when computing (meta-)gradients through Adam.
   :param mu_dtype: Optional `dtype` to be used for the first order accumulator; if
                    `None` then the `dtype` is inferred from `params` and `updates`.

   :returns: The corresponding `GradientTransformation`.


.. py:function:: tree_cast(tree, dtype)

   Cast tree to given dtype, skip if None.


.. py:function:: tree_zeros_like(tree, dtype = None)

   Creates an all-zeros tree with the same structure.

   :param tree: pytree.
   :param dtype: optional dtype to use for the tree of zeros.

   :returns: an all-zeros tree with the same structure as ``tree``.


.. py:function:: tree_update_moment(updates, moments, decay, order)

   Compute the exponential moving average of the `order`-th moment.

   :param updates: Gradients.
   :param moments: Moments.
   :param decay: Decay rate.
   :param order: Order of the moment.

   :returns: The updated moments.


.. py:function:: tree_update_moment_per_elem_norm(updates, moments, decay, order)

   Compute the EMA of the `order`-th moment of the element-wise norm.

   :param updates: Gradients.
   :param moments: Moments.
   :param decay: Decay rate.
   :param order: Order of the moment.

   :returns: The updated moments.


.. py:function:: tree_bias_correction(moment, decay, count)

   Performs bias correction. It becomes a no-op as count goes to infinity.

   :param moment: Moments.
   :param decay: Decay rate.
   :param count: Step count.

   :returns: The bias-corrected moments.


