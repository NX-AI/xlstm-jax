import numpy as np
from grain.python import MapDataset
from lm_eval.api.instance import Instance
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class HFTokenizeLogLikelihoodRolling:
    """
    Dataset that tokenizes (HuggingFace) and and splits documents according to the structure of
    loglikelihood_rolling. Targets are shifted for next token prediction.
    It does not work on a test instance level as used in e.g. grain, as documents are split into
    multiple sequences to match the maximal sequence length. However, we employ the .map()
    paradigm converting a list of lm_eval Instances to training instances (dict).
    See: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/model.py

    Prefix Tokens are handled here as well and masked out in the targets_segmentation.

    Args:
        tokenizer_path: HuggingFace tokenizer name
        max_length: Maximal sequence length / context_length
        batch_size: Batch size to be used for filling up the last batch.
        hf_access_token: HuggingFace access token for other tokenizers.
        tokenizer_cache_dir: HuggingFace tokenizer cache dir
        add_bos_token: If to add a beginning of sequence/docuemnt token.
        add_eos_token: If to add an end of sequence/document token.
        bos_token_id: BOS token id if not taken from tokenizer.
        eos_token_id: EOS token id if not taken from tokenizer.
    """

    def __init__(
        self,
        tokenizer_path: str,
        max_length: int,
        batch_size: int = 1,
        hf_access_token: str = "",
        tokenizer_cache_dir: str | None = None,
        add_bos_token: bool = True,
        add_eos_token: bool = False,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            clean_up_tokenization_spaces=False,  # See https://github.com/huggingface/transformers/issues/31884
            legacy=False,
            token=hf_access_token,
            use_fast=True,  # set this to true possibly with TOKENIZER_PARALLELISM=false
            add_bos=False,
            add_eos=False,
            cache_dir=tokenizer_cache_dir,
        )
        self.batch_size = batch_size
        self.max_length = max_length
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self._mapped_data = None
        self.bos_token_id = bos_token_id if bos_token_id is not None else self.tokenizer.eos_token_id
        self.eos_token_id = bos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id

    def _tokenize(self, example: str) -> BatchEncoding[str, list[int]]:
        """
        Tokenize a string with the tokenizer.

        Args:
            example: String to tokenize

        Returns:
            BatchEncoding in HF format with tokens.

        """
        return self.tokenizer(
            example,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
        )

    def simple_array(
        self, *, prefix_tokens: list[int], all_tokens: list[int], doc_idx: int, seq_idx: int
    ) -> MapDataset:
        """
        Creates a simple document instance with "standard" padding and masks.
        This is for documents not exceeding the max_length or all sequences
        except the last for a longer document.

        Args:
            prefix_tokens: List of prefix tokens
            all_tokens: List of all tokens
            doc_idx: Document index
            seq_idx: Sequence index (in document)

        Returns:
           Data instance dictionary.
        """
        inputs = np.zeros(self.max_length, dtype=np.int32)
        inputs_segmentation = np.zeros(self.max_length, dtype=np.int32)
        targets = np.zeros(self.max_length, dtype=np.int32)
        targets_segmentation = np.zeros(self.max_length, dtype=np.int32)
        if len(all_tokens) > 0:
            inputs[: min(len(all_tokens), self.max_length)] = np.asarray(all_tokens)[
                : min(len(all_tokens), self.max_length)
            ]

            inputs_segmentation[: min(len(all_tokens), self.max_length)] = 1
            targets[: len(all_tokens) - 1] = np.asarray(all_tokens)[1:]
            targets_segmentation[max(len(prefix_tokens) - 1, 0) : len(all_tokens) - 1] = 1

        return {
            "inputs": inputs,
            "inputs_segmentation": inputs_segmentation,
            "targets": targets,
            "targets_segmentation": targets_segmentation,
            "inputs_position": np.arange(self.max_length),
            "targets_position": np.arange(self.max_length),
            "document_idx": np.asarray(doc_idx, dtype=np.int32),
            "sequence_idx": np.asarray(seq_idx, dtype=np.int32),
        }

    def map(self, requests: list[Instance]) -> MapDataset:
        """
        Maps a list of lm_eval Instances to a (potentially longer) list of sequences
        for a language model evaluation. Generated instances are padded to max_length
        and contain position and segmentation information as well as document and sequnce
        indices.

        Args:
           requests: List of lm_eval Instances / Requests.

        Returns:
           List of converted instances for lm processing.
        """
        llm_instances = []

        for doc_idx, req in tqdm(enumerate(requests)):
            assert len(req.args) in [
                1,
                2,
            ], f"Expected `args` to be a tuple of length 1 or 2, got {len(req.args)}"
            prefix_tokens = [self.bos_token_id] if self.add_bos_token else []
            suffix_tokens = [self.eos_token_id] if self.add_eos_token else []

            if len(req.args) == 1:
                tokens = self._tokenize(req.args[0])["input_ids"] + suffix_tokens
            else:
                prefix_tokens = prefix_tokens + self._tokenize(req.args[0])["input_ids"]
                tokens = self._tokenize(req.args[1])["input_ids"] + suffix_tokens

            all_tokens = prefix_tokens + tokens

            if len(all_tokens) > self.max_length:
                offset = 0
                seq_idx = 0
                while offset + self.max_length < len(all_tokens):
                    llm_inst = self.simple_array(
                        prefix_tokens=[],
                        all_tokens=all_tokens[offset : offset + self.max_length + 1],
                        doc_idx=doc_idx,
                        seq_idx=seq_idx,
                    )
                    llm_instances.append(llm_inst)
                    seq_idx += 1
                    offset += self.max_length

                targets_segmentation = np.zeros(self.max_length, dtype=np.int32)
                inputs = np.asarray(all_tokens[-self.max_length - 1 : -1])
                targets = np.asarray(all_tokens[-self.max_length :])
                inputs_segmentation = np.ones(self.max_length, dtype=np.int32)
                targets_segmentation[-(len(all_tokens) - offset - 1) :] = 1

                llm_instances.append(
                    {
                        "document_idx": np.asarray(doc_idx, dtype=np.int32),
                        "sequence_idx": np.asarray(seq_idx, dtype=np.int32),
                        "inputs": inputs,
                        "targets": targets,
                        "inputs_segmentation": inputs_segmentation,
                        "targets_segmentation": targets_segmentation,
                        "inputs_position": np.arange(self.max_length),
                        "targets_position": np.arange(self.max_length),
                    }
                )
            else:
                llm_inst = self.simple_array(
                    prefix_tokens=prefix_tokens, all_tokens=all_tokens, doc_idx=doc_idx, seq_idx=0
                )
                llm_instances.append(llm_inst)

        # Fill up last batch with -1 indexed documents.
        if len(llm_instances) % self.batch_size != 0:
            for _ in range(self.batch_size - (len(llm_instances) % self.batch_size)):
                llm_instances.append(self.simple_array(prefix_tokens=[], all_tokens=[], doc_idx=-1, seq_idx=0))

        return MapDataset.source(llm_instances)


class HFTokenizeLogLikelihood:
    """
    Dataset mapper modeling a simplified lm_eval dataset. Post-processing here could be done
    using the grain pipeline. However, instances are not split if the exceed the maximal
    sequence length as for LoglikelihoodRolling
    See: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/model.py
    """

    def map(self, requests: list[Instance]) -> MapDataset:
        """
        Maps a list of lm_eval Instances to a dictionary usable in grain transforms.

        Args:
           requests: List of lm_eval Instances / Requests.

        Returns:
           List of converted instances for lm processing.
        """
        llm_instances = []

        for req in requests:
            assert len(req.args) in [1, 2], "Bad LM_EVAL request input size."
            llm_instances.append(
                {
                    "prefix": req.args[0] if len(req.args) == 2 else "",
                    "text": req.args[1] if len(req.args) == 1 else req.args[0],
                }
            )

        return MapDataset.source(llm_instances)
