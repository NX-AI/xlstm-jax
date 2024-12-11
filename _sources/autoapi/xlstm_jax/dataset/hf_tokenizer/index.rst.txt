xlstm_jax.dataset.hf_tokenizer
==============================

.. py:module:: xlstm_jax.dataset.hf_tokenizer


Functions
---------

.. autoapisummary::

   xlstm_jax.dataset.hf_tokenizer.load_tokenizer


Module Contents
---------------

.. py:function:: load_tokenizer(tokenizer_path, add_bos, add_eos, hf_access_token = None, cache_dir = None)

   Loads the tokenizer.

   :param tokenizer_path: The path to the tokenizer.
   :param add_bos: Whether to add the beginning of sequence token.
   :param add_eos: Whether to add the end of sequence token.
   :param hf_access_token: The access token for HuggingFace.
   :param cache_dir: The cache directory for the tokenizer.

   :returns: The tokenizer.


