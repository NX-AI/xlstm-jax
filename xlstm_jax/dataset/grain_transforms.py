"""Copyright 2023 Google LLC.

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

Operations used by Grain
"""

import dataclasses
from collections.abc import Callable
from typing import NamedTuple

import grain.python as grain
import jax
import numpy as np
import tensorflow as tf
import transformers
from transformers.tokenization_utils_base import BatchEncoding

from .batch import LLMBatch


# Functions used by HF pipeline
@dataclasses.dataclass
class HFNormalizeFeatures(grain.MapTransform):
    """Normalize feature keys for HuggingFace input."""

    def __init__(self, column_name: str):
        self.column_name = column_name

    def map(self, features: dict[str, list[int]]) -> dict[str, np.ndarray]:
        return {
            "inputs": np.asarray(features[self.column_name], dtype=np.int32),
            "targets": np.asarray(features[self.column_name], dtype=np.int32),
        }


@dataclasses.dataclass
class HFPrefixTokenize(grain.MapTransform):
    """Merge prefix and predicted text"""

    def __init__(
        self,
        tokenizer,
        prefix_tokenizer,
        prefix_column_name: str = "prefix",
        text_column_name: str = "text",
        add_bos_token: bool = True,
        add_eos_token: bool = False,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        max_length: int | None = None,
        max_length_prefix: int | None = None,
        keep_other_inputs: bool = True,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            prefix_tokenizer: HuggingFace prefix tokenizer
            prefix_column_name: Column Name in the dataframe for the prefix
            text_column_name: Column Name in the dataframe for the text
            add_bos_toke: If to add a bos token.
            max_length: Maximal total length (context_length)
            max_length_prefix: Maximal length of the prefix - all prefixes are added+padded such that
                the texts start at the same position
        """
        self.tokenizer = tokenizer
        self.prefix_tokenizer = prefix_tokenizer
        self.prefix_column_name = prefix_column_name
        self.text_column_name = text_column_name
        self.max_length = max_length
        self.max_length_prefix = (
            max_length_prefix
            if max_length_prefix is not None
            else (max_length // 2 if max_length is not None else None)
        )
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        self.bos_token_id = (
            bos_token_id
            if bos_token_id is not None
            else (
                self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id
            )
        )
        self.keep_other_inputs = keep_other_inputs

    def map(self, features: dict[str, str]) -> dict[str, np.ndarray]:
        """
        Map prefix / text string to fully padded and tokenized sequence.
        Prefixes are aligned in the array.

        Args:
            features: Dictionary of inputs

        Returns:
            Dictionary of the outputs
        """
        prefix = self.prefix_tokenizer(
            features[self.prefix_column_name],
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=self.max_length_prefix,
            return_tensors="np",
        )
        inputs = self.tokenizer(
            features[self.text_column_name],
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=self.max_length - self.max_length_prefix if self.max_length is not None else None,
            return_tensors="np",
        )

        output_tokens = np.concatenate([prefix.input_ids, inputs.input_ids], axis=1).astype(np.int32)
        if self.add_eos_token:
            output_tokens = np.pad(output_tokens, ((0, 0), (0, 1)), constant_values=self.eos_token_id)
        if self.add_bos_token:
            input_tokens = np.pad(output_tokens[:, :-1], ((0, 0), (1, 0)), constant_values=self.bos_token_id)
            input_mask = np.pad(
                np.concatenate([prefix.attention_mask, inputs.attention_mask[:, :-1]], axis=1),
                ((0, 0), (1, 0)),
                constant_values=1,
            )
            output_mask = np.pad(inputs.attention_mask, ((0, 0), (prefix.attention_mask.shape[1], 0)))
        else:
            input_tokens = output_tokens[:, :-1]
            output_tokens = output_tokens[:, 1:]
            input_mask = np.concatenate([prefix.attention_mask, inputs.attention_mask[:, :-1]], axis=1)
            output_mask = np.pad(inputs.attention_mask, ((0, 0), (prefix.attention_mask.shape[1], 0)))[:, 1:]

        other_inputs = {
            key: val for key, val in features.items() if key not in [self.prefix_column_name, self.text_column_name]
        }
        return {
            "inputs": input_tokens,
            "targets": output_tokens,
            "inputs_segmentation": input_mask,
            "targets_segmentation": output_mask,
            **other_inputs,
        }


# Functions used by Grain pipeline
@dataclasses.dataclass
class ParseFeatures(grain.MapTransform):
    """Parse serialized example."""

    def __init__(self, data_column: str, tokenize: bool):
        self.data_column = data_column
        if tokenize:
            self.dtype = tf.string
        else:
            self.dtype = tf.int64

    def map(self, features):
        def _parse(example):
            parsed = tf.io.parse_example(
                example, {self.data_column: tf.io.FixedLenSequenceFeature([], dtype=self.dtype, allow_missing=True)}
            )
            return parsed

        return _parse(features)


class ParseArrayRecords(grain.MapTransform):
    """Parse serialized example from array_records dataset."""

    def __init__(self, column_name: str):
        """
        Args:
            column_name: Column name to be used as key in the output dictionary.
        """
        self.column_name = column_name

    def map(self, data: bytes) -> dict[str, str]:
        """Map to parse array records.

        Args:
            data: The bytestring-serialized example, e.g. b'Some Text'.

        Returns:
            Parsed data, a dictionary mapping the column_name to the deserialized string (text).
        """
        return {self.column_name: data.decode()}


class ParseTokenizedArrayRecords(grain.MapTransform):
    """Parse serialized example from array_records dataset."""

    def __init__(self, column_name: str):
        """
        Args:
            column_name: Column name to be used as key in the output dictionary.
        """
        self.column_name = column_name

    def map(self, data: bytes) -> dict[str, int]:
        """Map to parse array records.

        Args:
            data: The bytestring-serialized data that has been tokenized, e.g. b'[0, 9392, 1823]'

        Returns:
            Parsed data, a dictionary mapping the column_name to the deserialized string (text).
        """
        return {self.column_name: self.bytestring_to_sequence(data)}

    @staticmethod
    def sequence_to_bytestring(sequence: list[int] | np.ndarray) -> bytes:
        """Convert a token sequence to a numpy bytestring.

        Args:
            sequence: The sequence of tokens. If a numpy array is provided, it must be one-dimensional.

        Returns:
            The bytestring.
        """
        if isinstance(sequence, np.ndarray):
            assert sequence.ndim == 1, "Sequence must be one-dimensional."
        assert np.max(sequence) <= np.iinfo(np.uint16).max, "Tokenizer above max vocab size for encoding."
        return np.array(sequence, dtype=np.uint16).tobytes()

    @staticmethod
    def bytestring_to_sequence(bytestring: bytes) -> list[int]:
        """Convert a numpy bytestring to a token sequence.

        Args:
            bytestring: The bytestring.

        Returns:
            The token sequence.
        """
        return np.frombuffer(bytestring, dtype=np.uint16).tolist()


@dataclasses.dataclass
class NormalizeFeatures(grain.MapTransform):
    """Normalize text feature keys."""

    def __init__(self, column_name: str, tokenize: bool):
        self.column_name = column_name
        self.tokenize = tokenize

    def map(self, features):
        if self.tokenize:
            return {
                "inputs": features[self.column_name].numpy()[0].decode(),
                "targets": features[self.column_name].numpy()[0].decode(),
            }
        else:
            return {"inputs": features[self.column_name].numpy(), "targets": features[self.column_name].numpy()}


@dataclasses.dataclass
class ReformatPacking(grain.MapTransform):
    """Reformat packing outputs."""

    def map(self, data: tuple[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {
            "inputs": data[0]["inputs"],
            "targets": data[0]["targets"],
            "inputs_segmentation": data[1]["inputs"],
            "targets_segmentation": data[1]["targets"],
            "inputs_position": data[2]["inputs"],
            "targets_position": data[2]["targets"],
        }


@dataclasses.dataclass
class ReformatLazyPacking(grain.MapTransform):
    """Reformat packing outputs for the lazy API."""

    def map(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for old_key, new_key in [
            ("inputs_segment_ids", "inputs_segmentation"),
            ("targets_segment_ids", "targets_segmentation"),
            ("inputs_positions", "inputs_position"),
            ("targets_positions", "targets_position"),
        ]:
            data[new_key] = data.pop(old_key)
        return data


@dataclasses.dataclass
class PadToMaxLength(grain.MapTransform):
    """Pads each input to the specified length."""

    def __init__(self, max_length: int):
        self.max_length = max_length

    def map(self, data) -> dict[str, np.ndarray]:
        """Map to each element."""

        def _pad(x, max_length: int):
            pad_amount = max(max_length - x.shape[0], 0)
            pad_amount = [(0, pad_amount)] + [(0, 0)] * (len(x.shape) - 1)
            return np.pad(x, pad_amount)

        data["inputs_segmentation"] = np.ones(data["inputs"].shape, dtype=np.int32)
        data["inputs_position"] = np.arange(data["inputs"].shape[0], dtype=np.int32)
        data["targets_segmentation"] = np.ones(data["targets"].shape, dtype=np.int32)
        data["targets_position"] = np.arange(data["targets"].shape[0], dtype=np.int32)
        for key, _ in data.items():
            data[key] = _pad(data[key], self.max_length)
        return data


def shift_right(x: np.ndarray, axis: int = 1, padding_value: int = 0, pad_by_first_element: bool = False) -> np.ndarray:
    """
    Shift the input to the right by padding and slicing on axis.

    Args:
        x: Input array to shift.
        axis: Axis to shift along.
        padding_value: Value to use for padding.
        pad_by_first_element: If True, does not use padding_value but instead the first element of the array on the
            axis.

    Returns:
        Shifted array.
    """
    if axis < 0:
        axis = x.ndim + axis
    assert axis < x.ndim, f"Axis {axis} is out of bounds for array of shape {x.shape}."
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (1, 0)
    slices = [
        slice(None),
    ] * len(x.shape)
    slices[axis] = slice(0, -1)
    if pad_by_first_element:
        padded = np.concatenate([np.expand_dims(x.take(0, axis=axis), axis=axis), x], axis=axis)
    else:
        padded = np.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(padding_value))
    return padded[tuple(slices)]


def shift_left(x: np.ndarray, axis: int = 1, padding_value: int = 0) -> np.ndarray:
    """
    Shift the input to the left by padding and slicing on axis.

    Args:
        x: Input array to shift.
        axis: Axis to shift along.
        padding_value: Value to use for padding.

    Returns:
        Shifted array.
    """
    if axis < 0:
        axis = x.ndim + axis
    assert axis < x.ndim, f"Axis {axis} is out of bounds for array of shape {x.shape}."
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (0, 1)
    slices = [
        slice(None),
    ] * len(x.shape)
    slices[axis] = slice(1, None)
    padded = np.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(padding_value))
    return padded[tuple(slices)]


def shift_and_refine(
    x: dict[str, np.ndarray],
    shift_target: bool = True,
    axis: int = 1,
    padding_value: int = 0,
) -> dict[str, np.ndarray]:
    """Shift inputs or targets, and adjust segmentation."""
    if shift_target:
        # Last token is made invalid.
        x["targets"] = shift_left(x["targets"], axis=axis, padding_value=padding_value)
        x["targets_segmentation"] = shift_left(x["targets_segmentation"], axis=axis, padding_value=0)
    else:
        # First token becomes start-of-sequence token (padding_value).
        x["inputs"] = shift_right(x["inputs"], axis=axis, padding_value=padding_value)
        # Do not shift inputs_segmentation: by our definition, the previous EOD marks the BOS ok the next document.
    return x


@dataclasses.dataclass
class ShiftData(grain.MapTransform):
    """Shift inputs/targets and refine annotations."""

    def __init__(
        self,
        shift_target: bool = True,
        eod_token_id: int = 0,
        pad_token_id: int = 0,
        axis: int = 1,
    ):
        self.shift_target = shift_target
        # When shifting inputs, we use the end-of-document token as the begin-of-sequence token.
        self.eod_token_id = eod_token_id
        # When shifting targets, we use padding tokens as the last token. Note that this can be
        # any token, as the last token is made invalid via the target mask.
        self.pad_token_id = pad_token_id
        self.axis = axis

    def map(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return shift_and_refine(
            x=data,
            shift_target=self.shift_target,
            padding_value=self.eod_token_id if not self.shift_target else self.pad_token_id,
            axis=self.axis,
        )


@dataclasses.dataclass
class InferSegmentations(grain.MapTransform):
    """
    Infer the segmentation, i.e. document numbers, from the inputs.

    Uses the end-of-document token to infer breaks between documents. This is not needed
    when performing packing, where the segmentations are already set correctly, but is
    useful for grouped text preprocessed datasets, which do not have the segmentations set.

    Args:
        eod_token_id: The token ID to use for the end-of-document token.
    """

    def __init__(
        self,
        eod_token_id: int,
    ):
        self.eod_token_id = eod_token_id

    def map(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Map to infer segmentations."""
        # Identify end-of-document tokens.
        eod_mask = data["inputs"] == self.eod_token_id
        # Document ID is the cumulative sum of the end-of-document mask. We start at 1,
        # as the first document is document 1 and 0 is used for padding. This is ensured
        # by adding 1 - eod_mask[:, 0:1], which adds 1 if the first token is not an EOD token,
        # ie the first token starts with document idx 0, and 0 otherwise.
        doc_id = np.cumsum(eod_mask, axis=-1, dtype=np.int32) + (1 - eod_mask[..., 0:1])
        # Set the segmentations, respecting the padding.
        inp_padding_mask = (data["inputs_segmentation"] != 0).astype(np.int32)
        data["inputs_segmentation"] = doc_id * inp_padding_mask
        out_padding_mask = (data["targets_segmentation"] != 0).astype(np.int32)
        data["targets_segmentation"] = doc_id * out_padding_mask
        # Correcting the positions to start counting from 0 at each new document.
        if data["inputs"].ndim == 1:
            data["inputs_position"] = self._get_positions(eod_mask)
        else:
            data["inputs_position"] = np.stack([self._get_positions(eod_mask[bs]) for bs in range(doc_id.shape[0])])
        # Targets have identical positions to inputs. Copy to prevent in-place modification.
        data["targets_position"] = data["inputs_position"].copy()
        # Mask out the padding.
        data["inputs_position"] *= inp_padding_mask
        data["targets_position"] *= out_padding_mask
        return data

    def _get_positions(self, eod_mask: np.ndarray) -> np.ndarray:
        """Infer positions from end-of-document mask."""
        seq_len = eod_mask.shape[0]
        # Correcting the positions to start counting from 0 at each new document. Example:
        # doc_id   = [[1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]]
        # eod_mask = [[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]]
        # inp_pos  = [[0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3]]
        # First element of eod_mask is optional.
        input_positions = np.array(np.zeros(seq_len), dtype=np.int32)
        doc_borders = [0] + np.where(eod_mask)[0].tolist() + [seq_len]
        for db in range(len(doc_borders) - 1):
            if doc_borders[db + 1] - doc_borders[db] == 0:
                continue
            doc_pos = np.arange(doc_borders[db + 1] - doc_borders[db], dtype=np.int32)
            input_positions[doc_borders[db] : doc_borders[db + 1]] = doc_pos
        return input_positions


@dataclasses.dataclass
class CollateToBatch(grain.MapTransform):
    """Collate data to batch.

    Args:
        batch_class: A NamedTuple or dataclass to hold the batch data.
        key_map: Dictionary to map input to batch keys. Keys that are not found in the dictionary will be used as is.
    """

    def __init__(self, batch_class: NamedTuple, key_map: dict[str, str] | None = None):
        self.batch_class = batch_class
        self.key_map = key_map if key_map is not None else {}

    def map(self, data: dict[str, np.ndarray | jax.Array]) -> LLMBatch:
        """Map to collate data to batch."""
        key_map = {key: self.key_map.get(key, key) for key in data.keys()}
        data = {key_map[key]: data[key] for key in data.keys()}
        target_keys = self.batch_class.__dataclass_fields__.keys()
        return self.batch_class(**{key: data[key] for key in target_keys if key in data})  # type: ignore


@dataclasses.dataclass
class HFTokenize(grain.MapTransform):
    """Tokenize text feature keys."""

    def __init__(
        self,
        create_tokenizer_fn: Callable[..., transformers.AutoTokenizer],
        column_name: str = "text",
        max_length: int | None = None,
        add_eod: bool = True,
        eod_token_id: int | None = None,
    ):
        """
        Args:
            create_tokenizer_fn: Function to create the HuggingFace tokenizer to use.
            column_name: Name of the column to tokenize.
            max_length: Maximum length of the sequence. If None, no truncation is performed.
            add_eod: Whether to add an end-of-document token to the sequence.
                     Note: EOD token is added only if sequence is shorter than max_length, truly marking the end.
            eod_token_id: Token ID to use for the end-of-document token. If None, the tokenizer's EOS token ID is used.
        """
        self.create_tokenizer_fn = create_tokenizer_fn
        self.tokenizer = None
        self.column_name = column_name
        self.max_length = max_length
        self.add_eod = add_eod
        self.eod_token_id = eod_token_id

    def _lazy_init(self):
        self.tokenizer = self.create_tokenizer_fn()
        if self.add_eod:
            self.eod_token_id = self.eod_token_id if self.eod_token_id is not None else self.tokenizer.eos_token_id
        else:
            self.eod_token_id = None

    def _tokenize(self, example: str) -> BatchEncoding[str, list[int]]:
        return self.tokenizer(
            example,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=self.max_length is not None,
            max_length=self.max_length,
        )

    def map(self, data: dict[str, str]) -> dict[str, list[int]]:
        if self.tokenizer is None:
            self._lazy_init()
        tokenized_data = self._tokenize(data[self.column_name])
        if self.add_eod:
            # using the EOS token id of the tokenizer for marking EOD (there is no EOD token in the tokenizer).
            tokenized_data_with_eod = {
                key: (val + [self.eod_token_id]) if self.max_length is None or len(val) < self.max_length else val
                for key, val in tokenized_data.items()
            }
            return tokenized_data_with_eod
        else:
            return tokenized_data


@dataclasses.dataclass
class AddEODToken(grain.MapTransform):
    """Add an end-of-document token to the inputs and targets.

    Args:
        eod_token_id: The token ID to use for the end-of-document token.
        add_eod: Whether to add the EOD token. If false, the transform is a no-op.
        max_length: Maximum length of the sequence. If None, no truncation is performed.
    """

    def __init__(
        self,
        eod_token_id: int,
        add_eod: bool = True,
        max_length: int | None = None,
    ):
        self.eod_token_id = eod_token_id
        self.add_eod = add_eod
        self.max_length = max_length

    def map(self, data: dict[str, list[int]]) -> dict[str, list[int]]:
        """Map to add EOD token."""
        if not self.add_eod:
            return data
        return {
            key: (val + [self.eod_token_id]) if self.max_length is None or len(val) < self.max_length else val
            for key, val in data.items()
        }
