from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

# Create a generic variable that can be 'DataConfig', or any subclass.
T = TypeVar("T", bound="DataConfig")


@dataclass(kw_only=True, frozen=False)
class DataConfig:
    """Base data configuration."""

    shuffle_data: bool = False
    """Whether to shuffle the data. Usually True for training and False for validation."""
    name: str | None = None
    """Name of the dataset. Helpful for logging."""
    data_config_type: str | None = None
    """Type of data configuration. Used for initialization via Hydra. Can be 'synthetic',
    'huggingface_hub', 'huggingface_local', or 'grain_arrayrecord'."""
    global_batch_size: int
    """Global batch size for training."""
    max_target_length: int | None = None
    """Maximum length of the target sequence."""
    data_shuffle_seed: int = 42
    """Seed for data shuffling."""

    @classmethod
    def create_train_eval_configs(
        cls: type[T], train_kwargs: dict | None = None, eval_kwargs: dict | None = None, **kwargs
    ) -> tuple[T, T]:
        """Create training and evaluation configurations.

        Args:
            train_kwargs: Training-exclusive keyword arguments.
            eval_kwargs: Evaluation-exclusive keyword arguments.
            **kwargs: Shared keyword arguments.

        Returns:
            Tuple[DataConfig, DataConfig]: Training and evaluation configurations.
        """
        if train_kwargs is None:
            train_kwargs = {}
        if eval_kwargs is None:
            eval_kwargs = {}
        train_kwargs.update(kwargs)
        eval_kwargs.update(kwargs)
        # Shuffle
        train_kwargs.setdefault("shuffle_data", True)
        eval_kwargs.setdefault("shuffle_data", False)
        # Drop remainder
        if hasattr(cls, "drop_remainder"):
            train_kwargs.setdefault("drop_remainder", True)
            eval_kwargs.setdefault("drop_remainder", False)
        # Split
        if hasattr(cls, "split"):
            train_kwargs.setdefault("split", "train")
            eval_kwargs.setdefault("split", "validation")
        return cls(**train_kwargs), cls(**eval_kwargs)


@dataclass(kw_only=True, frozen=False)
class HFHubDataConfig(DataConfig):
    """HuggingFace dataset configuration for datasets on HuggingFace."""

    hf_path: Path | str
    """Path to the dataset on HuggingFace."""
    hf_cache_dir: Path | str | None = None
    """Directory to cache the dataset."""
    hf_access_token: str | None = None
    """Access token for HuggingFace"""
    hf_data_dir: Path | str | None = None
    """Directory for additional data files."""
    hf_data_files: str | None = None
    """Specific (training or evaluation) files to use"""
    split: str | None = "train"
    """Split to use (for training or evaluation)."""
    hf_num_data_processes: int | None = None
    """Number of processes to use for downloading the dataset."""
    data_column: str = "text"
    """Column name for (training or evaluation) data."""
    max_steps_per_epoch: int | None = None
    """Maximum number of steps per epoch (for training or evaluation)."""
    tokenizer_path: str = "gpt2"
    """Path to the tokenizer."""  # TODO: is this really a path or rather a name?
    add_bos: bool = False
    """Whether to add `beginning of sequence` token."""
    add_eos: bool = False
    """Whether to add `end of sequence` token."""
    add_eod: bool = True
    """Whether to add an end of document token."""
    grain_packing: bool = False
    """Whether to perform packing via grain FirstFitPackIterDataset."""
    grain_packing_bin_count: int | None = None
    """Number of bins for grain packing. If None, use the local batch size. Higher values may improve packing
    efficiency but may also increase memory usage and pre-processing times."""
    worker_count: int = 1
    """Number of workers for data processing."""
    worker_buffer_size: int = 1
    """Buffer size for workers."""
    drop_remainder: bool = False
    """Whether to drop the remainder of the dataset when it does not divide evenly by the global batch size."""
    batch_rampup_factors: dict[int, float] | None = None
    """Ramp up the batch size if provided. The dictionary maps the step count to the scaling factor. See
    the `boundaries_and_scales` doc in `:func:grain_batch_rampup.create_batch_rampup_schedule` for more details."""


@dataclass(kw_only=True, frozen=False)
class SyntheticDataConfig(DataConfig):
    """Synthetic dataset configuration."""

    num_batches: int = 100
    """Number of samples to generate for synthetic (training or evaluation) data."""


@dataclass(kw_only=True, frozen=False)
class GrainArrayRecordsDataConfig(DataConfig):
    """Grain dataset configuration for ArrayRecords datasets."""

    data_path: Path
    """Path to the dataset directory."""
    data_column: str = "text"
    """Column name for (training or evaluation) data."""
    split: str = "train"
    """Dataset split to use, e.g. 'train' or 'validation'. Should be a subdirectory of data_dir."""
    drop_remainder: bool = False
    """Whether to drop the remainder of the dataset when it does not divide evenly by the global batch size."""
    max_steps_per_epoch: int | None = None
    """Maximum number of steps per epoch."""
    tokenize_data: bool = True  # TODO: in data_processing API just called tokenize. Better to have identical names.
    """Whether to tokenize the data data. If False, the data is assumed to be already tokenized."""
    tokenizer_path: str = "gpt2"
    """Path to the tokenizer."""  # TODO: is this really a path or rather a name?
    add_bos: bool = False
    """Whether to add `beginning of sequence` token."""
    add_eos: bool = False
    """Whether to add `end of sequence` token."""
    add_eod: bool = True
    """Whether to add an end of document token."""
    grain_packing: bool = False
    """Whether to perform packing via grain FirstFitPackIterDataset."""
    grain_packing_bin_count: int | None = None
    """Number of bins for grain packing. If None, use the local batch size. Higher values may improve packing
    efficiency but may also increase memory usage and pre-processing times."""
    worker_count: int = 1
    """Number of workers for data processing."""
    worker_buffer_size: int = 1
    """Buffer size for workers."""
    hf_cache_dir: Path | None = None
    """Directory to cache the dataset. Used to get the HF tokenizer."""
    hf_access_token: str | None = None
    """Access token for HuggingFace. Used to get the HF tokenizer."""
    batch_rampup_factors: dict[int, float] | None = None
    """Ramp up the batch size if provided. The dictionary maps the step count to the scaling factor. See
    the `boundaries_and_scales` doc in `:func:grain_batch_rampup.create_batch_rampup_schedule` for more details."""
