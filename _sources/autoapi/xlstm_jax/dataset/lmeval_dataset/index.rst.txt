xlstm_jax.dataset.lmeval_dataset
================================

.. py:module:: xlstm_jax.dataset.lmeval_dataset


Classes
-------

.. autoapisummary::

   xlstm_jax.dataset.lmeval_dataset.HFTokenizeLogLikelihoodRolling
   xlstm_jax.dataset.lmeval_dataset.HFTokenizeLogLikelihood


Module Contents
---------------

.. py:class:: HFTokenizeLogLikelihoodRolling(tokenizer_path, max_length, batch_size = 1, hf_access_token = '', tokenizer_cache_dir = None, add_bos_token = True, add_eos_token = False, bos_token_id = None, eos_token_id = None)

   Dataset that tokenizes (HuggingFace) and splits documents according to the structure of
   loglikelihood_rolling.

   Targets are shifted for next token prediction. It does not work on a test instance level as used in e.g. grain,
   as documents are split into multiple sequences to match the maximal sequence length. However, we employ the `.map()`
   paradigm converting a list of lm_eval Instances to training instances (dict).
   See: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/model.py

   Prefix tokens are handled here as well and masked out in the targets_segmentation.

   :param tokenizer_path: HuggingFace tokenizer name
   :param max_length: Maximal sequence length / context_length
   :param batch_size: Batch size to be used for filling up the last batch.
   :param hf_access_token: HuggingFace access token for other tokenizers.
   :param tokenizer_cache_dir: HuggingFace tokenizer cache dir
   :param add_bos_token: Whether to add a beginning of sequence / document token.
   :param add_eos_token: Whether to add an end of sequence / document token.
   :param bos_token_id: BOS token id if not taken from tokenizer.
   :param eos_token_id: EOS token id if not taken from tokenizer.


   .. py:attribute:: tokenizer


   .. py:attribute:: batch_size
      :value: 1



   .. py:attribute:: max_length


   .. py:attribute:: add_bos_token
      :value: True



   .. py:attribute:: add_eos_token
      :value: False



   .. py:attribute:: _mapped_data
      :value: None



   .. py:attribute:: bos_token_id
      :value: None



   .. py:attribute:: eos_token_id
      :value: None



   .. py:method:: _tokenize(example)

      Tokenize a string with the tokenizer.

      :param example: String to tokenize

      :returns: BatchEncoding in HF format with tokens.



   .. py:method:: simple_array(*, prefix_tokens, all_tokens, doc_idx, seq_idx)

      Creates a simple document instance with "standard" padding and masks.
      This is for documents not exceeding the max_length or all sequences
      except the last for a longer document.

      :param prefix_tokens: List of prefix tokens
      :param all_tokens: List of all tokens
      :param doc_idx: Document index
      :param seq_idx: Sequence index (in document)

      :returns: Data instance dictionary.



   .. py:method:: map(requests)

      Maps a list of lm_eval Instances to a (potentially longer) list of sequences
      for a language model evaluation. Generated instances are padded to max_length
      and contain position and segmentation information as well as document and sequnce
      indices.

      :param requests: List of lm_eval Instances / Requests.

      :returns: List of converted instances for lm processing.



.. py:class:: HFTokenizeLogLikelihood

   Dataset mapper modeling a simplified lm_eval dataset. Post-processing here could be done
   using the grain pipeline. However, instances are not split if they exceed the maximal
   sequence length as for LoglikelihoodRolling
   See: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/model.py


   .. py:method:: map(requests)
      :staticmethod:


      Maps a list of lm_eval Instances to a dictionary usable in grain transforms.

      :param requests: List of lm_eval Instances / Requests.

      :returns: List of converted instances for lm processing.



