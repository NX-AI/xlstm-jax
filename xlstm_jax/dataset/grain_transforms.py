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
from typing import NamedTuple

import grain.python as grain
import jax
import numpy as np
import tensorflow as tf
from transformers.tokenization_utils_base import BatchEncoding

from xlstm_jax.dataset.batch import LLMBatch


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
        max_length: int | None = None,
        max_length_prefix: int | None = None,
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
        self.max_length_prefix = max_length_prefix if max_length_prefix is not None else max_length // 2
        self.add_bos_token = add_bos_token

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
            max_length=self.max_length - self.max_length_prefix,
            return_tensors="np",
        )
        output_tokens = np.concatenate([prefix.input_ids, inputs.input_ids], axis=1)
        if self.add_bos_token:
            bos_tokens = np.full((output_tokens.shape[0], 1), self.tokenizer.bos_token_id, dtype=np.int32)
            input_tokens = np.concatenate([bos_tokens, output_tokens[:, :-1]], axis=1)
            bos_mask = int(self.add_bos_token) * np.ones_like(prefix.attention_mask[:, :1])
            input_mask = np.concatenate([bos_mask, prefix.attention_mask, inputs.attention_mask[:, :-1]], axis=1)
            output_mask = np.concatenate([np.zeros_like(prefix.attention_mask), inputs.attention_mask], axis=1)

        else:
            input_tokens = output_tokens[:, :-1]
            output_tokens = output_tokens[:, 1:]
            input_mask = np.concatenate([prefix.attention_mask, inputs.attention_mask[:, :-1]], axis=1)
            output_mask = np.concatenate([np.zeros_like(prefix.attention_mask), inputs.attention_mask], axis=1)[1:]

        return {
            "inputs": input_tokens,
            "targets": output_tokens,
            "inputs_segmentation": input_mask,
            "output_segmentation": output_mask,
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
            data: The bytestring-serialized example.

        Returns:
            Parsed data, a dictionary mapping the column_name to the deserialized string (text).
        """
        return {self.column_name: data.decode()}


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
        # When shifting inputs right, the first token is a start-of-sequence token for the 1st sequence.
        # Thus, it gets the same segmentation as the first element.
        x["inputs_segmentation"] = shift_right(
            x["inputs_segmentation"],
            axis=axis,
            pad_by_first_element=True,
        )
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
        # as the first document is document 1 and 0 is used for padding.
        doc_id = np.cumsum(eod_mask, axis=-1, dtype=np.int32) + 1
        # Set the segmentations, respecting the padding.
        inp_padding_mask = (data["inputs_segmentation"] != 0).astype(np.int32)
        data["inputs_segmentation"] = doc_id * inp_padding_mask
        out_padding_mask = (data["targets_segmentation"] != 0).astype(np.int32)
        data["targets_segmentation"] = doc_id * out_padding_mask
        return data


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
        return self.batch_class(**{key: data[key] for key in target_keys})  # type: ignore


@dataclasses.dataclass
class HFTokenize(grain.MapTransform):
    """Tokenize text feature keys."""

    def __init__(
        self,
        tokenizer,
        column_name: str = "text",
        max_length: int | None = None,
        add_eod: bool = True,
        eod_token_id: int | None = None,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer to use.
            column_name: Name of the column to tokenize.
            max_length: Maximum length of the sequence. If None, no truncation is performed.
            add_eod: Whether to add an end-of-document token to the sequence.
                     Note: EOD token is added only if sequence is shorter than max_length, truly marking the end.
            eod_token_id: Token ID to use for the end-of-document token. If None, the tokenizer's EOS token ID is used.
        """
        self.tokenizer = tokenizer
        self.column_name = column_name
        self.max_length = max_length
        self.add_eod = add_eod
        if add_eod:
            self.eod_token_id = eod_token_id if eod_token_id is not None else tokenizer.eos_token_id
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
