#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import numpy as np
import pytest

from xlstm_jax.dataset.grain_transforms import InferSegmentations


@pytest.mark.parametrize("eod_idx", [0, 100, 50234])
def test_infer_segmentations(eod_idx: int):
    """Test InferSegmentations."""
    pad_idx = -1 if eod_idx == 0 else 0
    # Test with hand-created example.
    example = {
        "inputs": np.array(
            [
                [1, 2, eod_idx, 2, 5, 4, 3, eod_idx, 5, 4, 2],  # Random eod_idx in sequence.
                [eod_idx, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # eod_idx at start.
                [eod_idx, eod_idx, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # double eod_idx at start.
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, eod_idx],  # eod_idx at end.
                [1, 2, 3, eod_idx, pad_idx, pad_idx, pad_idx, pad_idx, pad_idx, pad_idx, pad_idx],  # pad_idx at end.
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, pad_idx],  # pad_idx at end without eod_idx.
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # no eod_idx or pad_idx.
                [eod_idx, 1, eod_idx, 2, 5, 3, eod_idx, 5, 3, eod_idx, pad_idx],  # eod_idx and pad_idx in sequence.
            ]
        ),
    }
    expected_segmentations = np.array(
        [
            [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
            [1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 0],
        ]
    )
    expected_positions = np.array(
        [
            [0, 1, 0, 1, 2, 3, 4, 0, 1, 2, 3],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
            [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [0, 1, 0, 1, 2, 3, 0, 1, 2, 0, 0],
        ]
    )
    # Initial segmentations.
    example["inputs_segmentation"] = (example["inputs"] != pad_idx).astype(np.int32)
    example["targets_segmentation"] = example["inputs_segmentation"]
    # Initial positions.
    example["inputs_position"] = np.arange(example["inputs"].shape[1], dtype=np.int32)[None, :].repeat(
        example["inputs"].shape[0], axis=0
    )
    example["targets_position"] = example["inputs_position"]

    # Output.
    transformation = InferSegmentations(eod_token_id=eod_idx)
    for setting in ["batchwise", "elementwise"]:
        prefix = f"[Setting: {setting}]"
        if setting == "batchwise":
            output = transformation.map(example)
        else:
            output = [
                transformation.map({key: value[i] for key, value in example.items()})
                for i in range(example["inputs"].shape[0])
            ]
            output = {key: np.stack([o[key] for o in output]) for key in output[0].keys()}
        # Check that the output is correct.
        np.testing.assert_array_equal(
            example["inputs"], output["inputs"], err_msg=f"{prefix} Inputs should not change."
        )
        np.testing.assert_array_equal(
            example["inputs_segmentation"],
            expected_segmentations,
            err_msg=f"{prefix} Inputs segmentation incorrectly inferred.",
        )
        np.testing.assert_array_equal(
            example["targets_segmentation"],
            expected_segmentations,
            err_msg=f"{prefix} Targets segmentation incorrectly inferred.",
        )
        np.testing.assert_array_equal(
            example["inputs_position"], expected_positions, err_msg=f"{prefix} Inputs position incorrectly inferred."
        )
        np.testing.assert_array_equal(
            example["targets_position"], expected_positions, err_msg=f"{prefix} Targets position incorrectly inferred."
        )
