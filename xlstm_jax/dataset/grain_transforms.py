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


def shift_right(x: np.ndarray, axis: int = 1, padding_value: int = 0) -> np.ndarray:
    """Shift the input to the right by padding and slicing on axis."""
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (1, 0)
    slices = [
        slice(None),
    ] * len(x.shape)
    slices[axis] = slice(0, -1)
    padded = np.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(padding_value))
    return padded[tuple(slices)]


def shift_left(x: np.ndarray, axis: int = 1, padding_value: int = 0) -> np.ndarray:
    """Shift the input to the left by padding and slicing on axis."""
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
        x["inputs_segmentation"] = shift_right(x["inputs_segmentation"], axis=axis, padding_value=0)

    # Invalidate tokens by setting padded_value (0) where we have padding (i.e. where segmentation mask is zero).
    targets_mask = x["targets_segmentation"] != 0
    inputs_mask = x["inputs_segmentation"] != 0
    x["targets"] = x["targets"] * targets_mask + padding_value * (~targets_mask)
    x["inputs"] = x["inputs"] * inputs_mask + padding_value * (~inputs_mask)
    return x


@dataclasses.dataclass
class ShiftData(grain.MapTransform):
    """Shift inputs/targets and refine annotations."""

    def __init__(self, shift_target: bool = True, axis: int = 1):
        self.shift_target = shift_target
        self.axis = axis

    def map(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return shift_and_refine(data, shift_target=self.shift_target, axis=self.axis)


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

    def __init__(self, tokenizer, column_name: str = "text", max_length: int | None = None, add_eod: bool = True):
        """
        Args:
            tokenizer: HuggingFace tokenizer to use.
            column_name: Name of the column to tokenize.
            max_length: Maximum length of the sequence. If None, no truncation is performed.
            add_eod: Whether to add an end-of-document token to the sequence.
                     Note: EOD token is added only if sequence is shorter than max_length, truly marking the end.
        """
        self.tokenizer = tokenizer
        self.column_name = column_name
        self.max_length = max_length
        self.add_eod = add_eod

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
            eod_token_id = self.tokenizer.eos_token_id
            tokenized_data_with_eod = {
                key: (val + [eod_token_id]) if self.max_length is None or len(val) < self.max_length else val
                for key, val in tokenized_data.items()
            }
            return tokenized_data_with_eod
        else:
            return tokenized_data
