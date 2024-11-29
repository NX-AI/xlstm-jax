from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models.configs import ParallelConfig

if pytest.grain_available:
    from xlstm_jax.dataset.grain_iterator import (
        make_grain_llm_dataset,
        make_grain_llm_iterator,
        make_grain_multihost_iterator,
    )
else:
    make_grain_llm_iterator = None
    make_grain_multihost_iterator = None
    make_grain_llm_dataset = None


@pytest.mark.skipif(not pytest.grain_available, reason="Grain is not available.")
@pytest.mark.parametrize("dataloading_host_count", [2, 4])
@pytest.mark.parametrize("num_elements", [34, 201])
@pytest.mark.parametrize("global_batch_size", [32])
@pytest.mark.parametrize("drop_remainder", [True, False])
def test_grain_dataloader_process_split(
    dataloading_host_count: int, num_elements: int, global_batch_size: int, drop_remainder: bool
):
    """Test that the dataset is correctly sharded over dataloading hosts."""
    # Initialize mesh.
    parallel = ParallelConfig(
        fsdp_axis_size=1,
        model_axis_size=1,
        data_axis_size=1,
    )
    # Use only one device for this test, as in practice, each data loader process only sees a single device.
    mesh = initialize_mesh(
        init_distributed_on_slurm=False, parallel_config=parallel, device_array=np.array(jax.devices())[0:1]
    )

    # Create a synthetic dataset with the specified number of elements.
    context_length = 16
    dataset = [
        {key: np.ones((context_length,), dtype=np.int32) * idx for key in ["inputs", "targets"]}
        for idx in range(num_elements)
    ]

    # Load all batches on each host.
    all_batches = {}
    for host_index in range(dataloading_host_count):
        dataloader = make_grain_llm_iterator(
            dataloading_host_index=host_index,
            dataloading_host_count=dataloading_host_count,
            global_mesh=mesh,
            dataset=dataset,
            global_batch_size=global_batch_size,
            max_target_length=context_length,
            shuffle=False,
            data_shuffle_seed=42,
            num_epochs=1,
            drop_remainder=drop_remainder,
            worker_count=0,
        )
        all_batches[host_index] = list(dataloader)

    # Check that the data is loaded correctly.
    all_batches = jax.device_get(all_batches)
    all_data = np.concatenate(
        [batch.targets[:, 0] for host_batches in all_batches.values() for batch in host_batches], axis=0
    )
    if drop_remainder:
        # Check number of loaded elements.
        assert len(all_data) == (num_elements // global_batch_size) * global_batch_size
        # Check that each host has loaded the first N elements of its shard.
        elements_per_host = all_data.shape[0] // dataloading_host_count
        dataset_size_per_host = num_elements // dataloading_host_count
        for host_index in range(dataloading_host_count):
            host_data = all_data[host_index * elements_per_host : (host_index + 1) * elements_per_host]
            size_remainder = num_elements % dataloading_host_count
            start_index = host_index * dataset_size_per_host + min(host_index, size_remainder)
            np.testing.assert_array_equal(
                host_data,
                np.arange(start_index, start_index + elements_per_host),
                f"Host index {host_index} did not load the expected data.",
            )
    else:
        # Check number of loaded elements.
        assert len(all_data) == num_elements, f"Expected {num_elements} elements, but got {len(all_data)}."
        np.testing.assert_array_equal(all_data, np.arange(num_elements))
        # Check that each host has loaded the same number of batches.
        num_batches = len(all_batches[0])
        # NOTE: This is not guaranteed if one host has no remainder, but the other does (e.g. 5 elements, batch size 2).
        # Thus, we should always drop the remainder or pad the dataset from the beginning on for multi-host dataloaders.
        if 0 < (num_elements % global_batch_size) * 1.0 / dataloading_host_count < 1:
            # Some hosts have a remainder, some don't.
            for host_index in range(dataloading_host_count):
                if host_index < num_elements % dataloading_host_count:
                    assert len(all_batches[host_index]) == num_batches, (
                        f"Host index {host_index} did not load the expected number of batches {num_batches}, "
                        f"but instead {len(all_batches[host_index])}."
                    )
                else:
                    assert len(all_batches[host_index]) == num_batches - 1, (
                        f"Host index {host_index} did not load the expected number of batches {num_batches - 1}, "
                        f"but instead {len(all_batches[host_index])}."
                    )
        else:
            # Each host is supposed to have a remainder.
            for host_index in range(dataloading_host_count):
                assert len(all_batches[host_index]) == num_batches, (
                    f"Host index {host_index} did not load the expected number of batches {num_batches}, "
                    f"but instead {len(all_batches[host_index])}."
                )


@pytest.mark.skipif(not pytest.grain_available, reason="Grain is not available.")
@pytest.mark.parametrize("num_elements", [256, 500])
@pytest.mark.parametrize("global_batch_size", [32])
def test_grain_dataloader_shuffling_over_epochs(num_elements: int, global_batch_size: int):
    """Test that the shuffling is different for each epoch."""
    # Initialize mesh.
    parallel = ParallelConfig(
        fsdp_axis_size=1,
        model_axis_size=1,
        data_axis_size=1,
    )
    # Use only one device for this test, as in practice, each data loader process only sees a single device.
    mesh = initialize_mesh(
        init_distributed_on_slurm=False, parallel_config=parallel, device_array=np.array(jax.devices())[0:1]
    )

    # Create a synthetic dataset with the specified number of elements.
    context_length = 16
    dataset = [
        {key: np.ones((context_length,), dtype=np.int32) * idx for key in ["inputs", "targets"]}
        for idx in range(num_elements)
    ]

    # Create iterator.
    dataloader = make_grain_llm_iterator(
        dataloading_host_index=0,
        dataloading_host_count=1,
        global_mesh=mesh,
        dataset=dataset,
        global_batch_size=global_batch_size,
        max_target_length=context_length,
        shuffle=True,
        data_shuffle_seed=42,
        num_epochs=4,
        drop_remainder=True,
        worker_count=0,
    )

    # Load all batches across epochs.
    it_len = dataloader.iterator_length
    batches_per_epoch = {}
    for epoch_idx in range(4):
        batches_per_epoch[epoch_idx] = list(dataloader)
        assert (
            len(batches_per_epoch[epoch_idx]) == it_len
        ), f"Expected {it_len} batches, but got {len(batches_per_epoch[epoch_idx])}."
    # NOTE: The data loader may contain further batches if the number of elements is not divisible by the batch size.
    # in this case, the last elements of a batch are forwarded to the next epoch, which is why the number of batches
    # may be larger than the number of elements divided by the batch size.

    # Make sure that the first batch from each epoch is different.
    for i in range(len(batches_per_epoch) - 1):
        for j in range(i + 1, len(batches_per_epoch)):
            assert not jnp.all(
                jnp.equal(batches_per_epoch[i][0].inputs, batches_per_epoch[j][0].inputs)
            ), f"The first batches of any two epochs should be different, but got the same for {i} and {j}."

    # Make sure that the batches are shuffled within each epoch.
    ep0_batches = jax.device_get(batches_per_epoch[0])
    ep0_batches = np.concatenate([batch.targets[:, 0] for batch in ep0_batches], axis=0)
    assert not np.all(np.sort(ep0_batches) == ep0_batches), "The batches should be shuffled within each epoch."
    assert (
        np.unique(ep0_batches).shape[0] == ep0_batches.shape[0]
    ), "The batches should contain all unique elements of the dataset."


@pytest.mark.skipif(not pytest.grain_available, reason="Grain is not available.")
@pytest.mark.parametrize("switch_after_num_batches", [5, 25])
@pytest.mark.parametrize("use_packing,use_batch_rampup", [(True, False), (False, True), (False, False)])
def test_grain_dataloader_states(switch_after_num_batches: int, use_packing: bool, use_batch_rampup: bool):
    """Test that we can get and set states as expected."""
    num_elements = 500
    global_batch_size = 32
    if use_batch_rampup:
        batch_rampup_factors = {0: 0.125, 5: 0.25, 10: 0.5, 20: 1.0}
    else:
        batch_rampup_factors = None
    # Initialize mesh.
    parallel = ParallelConfig(
        fsdp_axis_size=1,
        model_axis_size=1,
        data_axis_size=1,
    )
    # Use only one device for this test, as in practice, each data loader process only sees a single device.
    mesh = initialize_mesh(
        init_distributed_on_slurm=False, parallel_config=parallel, device_array=np.array(jax.devices())[0:1]
    )

    # Create a synthetic dataset with the specified number of elements.
    context_length = 16
    dataset = [
        {key: np.ones((context_length,), dtype=np.int32) * idx for key in ["inputs", "targets"]}
        for idx in range(num_elements)
    ]

    # Create iterator.
    make_dataloader_fn = partial(
        make_grain_llm_iterator,
        dataloading_host_index=0,
        dataloading_host_count=1,
        global_mesh=mesh,
        dataset=dataset,
        global_batch_size=global_batch_size,
        max_target_length=context_length,
        grain_packing=use_packing,
        grain_packing_bin_count=global_batch_size * 4,  # Test stopping within a packed batch.
        shuffle=True,
        data_shuffle_seed=42,
        num_epochs=4,
        drop_remainder=True,
        batch_rampup_factors=batch_rampup_factors,
    )
    dataloader = make_dataloader_fn()

    # Load a couple of batches and save the state.
    batch_idx = 0
    next_batches = []
    save_state = None
    for _ in range(4):
        for batch in dataloader:
            batch_idx += 1
            if batch_idx == switch_after_num_batches:
                save_state = dataloader.get_state()
            elif batch_idx > switch_after_num_batches:
                next_batches.append(batch)
                if batch_idx == switch_after_num_batches + 10:
                    break
        if len(next_batches) == 10:
            break

    # Create a new iterator and set the state.
    new_dataloader = make_dataloader_fn()
    new_dataloader.set_state(save_state)

    # Load the next batches from the new iterator.
    new_batches = []
    for _ in range(4):
        for batch in new_dataloader:
            new_batches.append(batch)
            if len(new_batches) == len(next_batches):
                break
        if len(new_batches) == len(next_batches):
            break

    # Check that the batches are the same.
    next_batches = jax.device_get(next_batches)
    new_batches = jax.device_get(new_batches)
    for idx, (batch1, batch2) in enumerate(zip(next_batches, new_batches)):
        for attr in ["inputs", "targets"]:
            np.testing.assert_array_equal(
                getattr(batch1, attr), getattr(batch2, attr), f"Batch {idx} does not match for attribute {attr}."
            )


@pytest.mark.skipif(not pytest.grain_available, reason="Grain is not available.")
@pytest.mark.parametrize("global_batch_size", [32])
@pytest.mark.parametrize("drop_remainder", [True, False])
@pytest.mark.parametrize("batch_rampup_factors", [{0: 0.25, 5: 0.5, 10: 1.0, 15: 2.0}, {0: 1.0}])
def test_grain_dataloader_rampup(global_batch_size: int, drop_remainder: bool, batch_rampup_factors: dict[int, float]):
    """Test that the dataset is correctly sharded over dataloading hosts."""
    # Initialize mesh.
    parallel = ParallelConfig(
        fsdp_axis_size=1,
        model_axis_size=1,
        data_axis_size=1,
    )
    # Use only one device for this test, as in practice, each data loader process only sees a single device.
    mesh = initialize_mesh(
        init_distributed_on_slurm=False, parallel_config=parallel, device_array=np.array(jax.devices())[0:1]
    )

    # Create a synthetic dataset with the specified number of elements.
    context_length = 16
    dataset = [
        {key: np.ones((context_length,), dtype=np.int32) * idx for key in ["inputs", "targets"]} for idx in range(2_000)
    ]

    # Load all batches on each host.
    dataloader = make_grain_llm_iterator(
        dataloading_host_index=0,
        dataloading_host_count=1,
        global_mesh=mesh,
        dataset=dataset,
        global_batch_size=global_batch_size,
        max_target_length=context_length,
        shuffle=False,
        data_shuffle_seed=42,
        num_epochs=1,
        drop_remainder=drop_remainder,
        worker_count=0,
        batch_rampup_factors=batch_rampup_factors,
    )
    all_batches = list(dataloader)

    expected_batch_size = global_batch_size
    for step in range(len(all_batches)):
        if step in batch_rampup_factors:
            expected_batch_size = int(global_batch_size * batch_rampup_factors[step])
        batch = all_batches[step]
        # Skip if we drop the remainder and this is the last batch.
        if step < len(all_batches) - 1 or drop_remainder:
            assert (
                batch.inputs.shape[0] == expected_batch_size
            ), f"Expected batch size {expected_batch_size} at step {step}, but got {batch.inputs.shape[0]}."


@pytest.mark.skipif(not pytest.grain_available, reason="Grain is not available.")
@pytest.mark.parametrize("uniform_weights", [True, False])
def test_grain_dataloader_mixing(uniform_weights: bool):
    """Test that mixing multiple datasets returns the expected results."""
    num_elements = 500
    global_batch_size = 32
    # Initialize mesh.
    parallel = ParallelConfig(
        fsdp_axis_size=1,
        model_axis_size=1,
        data_axis_size=1,
    )
    # Use only one device for this test, as in practice, each data loader process only sees a single device.
    mesh = initialize_mesh(
        init_distributed_on_slurm=False, parallel_config=parallel, device_array=np.array(jax.devices())[0:1]
    )

    # Create a synthetic dataset with the specified number of elements. We set each element's value to the
    # dataset index to make it easier to check that the elements are correctly mixed.
    context_length = 16
    dataset = [
        {key: np.ones((context_length,), dtype=np.int32) * idx for key in ["inputs", "targets"]}
        for idx in range(num_elements)
    ]
    dataset_splits = [
        dataset[: num_elements // 2],
        dataset[num_elements // 2 : num_elements // 2 + num_elements // 4],
        dataset[num_elements // 2 + num_elements // 4 :],
    ]
    dataset_weights = [len(ds) for ds in dataset_splits] if not uniform_weights else None

    # Create iterator.
    grain_datasets = []
    for dataset_split in dataset_splits:
        grain_ds = make_grain_llm_dataset(
            dataloading_host_index=0,
            dataloading_host_count=1,
            dataset=dataset_split,
            global_batch_size=global_batch_size,
            max_target_length=context_length,
            shuffle=True,
            data_shuffle_seed=42,
            num_epochs=None,
        )
        grain_datasets.append(grain_ds)

    dataloader = make_grain_multihost_iterator(
        grain_datasets=grain_datasets,
        dataset_lengths=[len(ds) for ds in dataset_splits],
        global_mesh=mesh,
        global_batch_size=global_batch_size,
        dataloading_host_count=1,
        dataset_weights=dataset_weights,
        worker_count=0,
        drop_remainder=True,
    )

    # Load all batches across epochs.
    it_len = dataloader.iterator_length
    if uniform_weights:
        exp_it_len = len(dataset_splits[0]) * 3 // global_batch_size
    else:
        exp_it_len = len(dataset) // global_batch_size
    assert np.allclose(it_len, exp_it_len, atol=1.0), f"Expected {exp_it_len} batches, but got {it_len}."
    batches_per_epoch = {}
    num_epochs = 4
    for epoch_idx in range(num_epochs):
        batches_per_epoch[epoch_idx] = [batch for batch in dataloader]
        assert (
            len(batches_per_epoch[epoch_idx]) == it_len
        ), f"Expected {it_len} batches, but got {len(batches_per_epoch[epoch_idx])} at epoch {epoch_idx}."
    # NOTE: The data loader may contain further batches if the number of elements is not divisible by the batch size.
    # in this case, the last elements of a batch are forwarded to the next epoch, which is why the number of batches
    # may be larger than the number of elements divided by the batch size.

    # Make sure that the first batch from each epoch is different.
    for i in range(len(batches_per_epoch) - 1):
        for j in range(i + 1, len(batches_per_epoch)):
            assert not jnp.all(
                jnp.equal(batches_per_epoch[i][0].inputs, batches_per_epoch[j][0].inputs)
            ), f"The first batches of any two epochs should be different, but got the same for {i} and {j}."

    # Make sure that the batches are shuffled within each epoch.
    ep0_batches = jax.device_get(batches_per_epoch[0])
    ep0_batches = np.concatenate([batch.targets[:, 0] for batch in ep0_batches], axis=0)
    assert not np.all(np.sort(ep0_batches) == ep0_batches), "The batches should be shuffled within each epoch."

    # Make sure that each batch contains elements from all datasets. We can identify the batch elements by the
    # dataset index, which is the same as the element value. The first dataset is the first half of the elements,
    # i.e. [0, num_elements // 2), the second dataset is the second quarter, i.e. [num_elements // 2,
    # num_elements // 2 + num_elements // 4), and the last dataset is the last quarter, i.e.
    # [num_elements // 2 + num_elements // 4, num_elements).
    for batch in batches_per_epoch[0]:
        batch = jax.device_get(batch)
        assert np.min(batch.targets[:, 0]) < num_elements // 2, "The first dataset should be present in each batch."
        assert np.any(
            np.logical_and(
                batch.targets[:, 0] >= num_elements // 2, batch.targets[:, 0] < num_elements // 2 + num_elements // 4
            )
        ), "The second dataset should be present in each batch."
        assert (
            np.max(batch.targets[:, 0]) >= num_elements // 2 + num_elements // 4
        ), "The last dataset should be present in each batch."

    # Check that the frequencies are as expected. As each sequence in the dataset has the unique value of its index,
    # we can count the frequencies of the values in the targets to check how often each sequence occurs.
    all_batches = sum(batches_per_epoch.values(), [])
    all_batches = jax.device_get(all_batches)
    freqs = np.bincount(np.concatenate([batch.targets[:, 0] for batch in all_batches], axis=0))
    if not uniform_weights:
        # In the non-uniform weight setting, we selected the weights such that the datasets are sampled according to
        # their lengths. Thus, we expect each sequence to occur equally often, and roughly once per epoch (up to
        # drop_remainder).
        assert np.allclose(
            freqs, np.mean(freqs), atol=1.0
        ), "The frequencies should be uniform (weights correspond to data length)."
        assert np.allclose(freqs, num_epochs, atol=1.0), "The frequencies should be close to number of epochs."
    else:
        # In the uniform weight setting, we expect the first dataset to occur half as often as the second and third
        # datasets, as the first dataset is twice as large as the second and third datasets.
        assert not np.allclose(freqs, np.mean(freqs), atol=1.0), "The frequencies should not be uniform."
        assert (
            freqs[: num_elements // 2].sum() < freqs[num_elements // 2 :].sum()
        ), "The last samples should be more frequent."
        assert np.allclose(
            freqs[: num_elements // 2], num_epochs, atol=1.0
        ), "The frequencies should be uniform for the first dataset."
        assert np.allclose(
            freqs[num_elements // 2 :], num_epochs * 2, atol=1.0
        ), "The frequencies should be uniform for the last dataset."
