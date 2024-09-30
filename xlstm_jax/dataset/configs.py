from dataclasses import dataclass

from xlstm_jax.configs import ConfigDict


@dataclass(kw_only=True, frozen=True)
class DataConfig(ConfigDict):
    """Base data configuration."""

    global_batch_size: int
    """Global batch size for training."""
    max_target_length: int | None = None
    """Maximum length of the target sequence."""
    global_batch_size_to_train_on: int = -1
    """Global batch size to train on. If -1, it will be set to :attr:`global_batch_size`. Otherwise, should be smaller
    than :attr:`global_batch_size` to indicate a reduced batch size for training."""
    global_batch_size_for_eval: int = -1
    """Global batch size for evaluation. If -1, it will be set to :attr:`global_batch_size`. Can be larger or smaller
    than :attr:`global_batch_size`, but at least 1 per device."""
    expansion_factor_real_data: int = -1
    """Expansion factor for real data. If -1, all hosts will load real data."""
    shuffle_train_data: bool = True
    """Whether to shuffle the training data."""
    data_shuffle_seed: int = 42
    """Seed for data shuffling."""


@dataclass(kw_only=True, frozen=True)
class HFDataConfig(DataConfig):
    """HuggingFace dataset configuration."""

    num_train_epochs: int
    """Number of epochs to train on. The data iterator for training data needs to know how many epochs to iterate over
    because of the way `grain` implements shuffling."""
    hf_path: str
    """Path to the dataset on HuggingFace."""
    hf_cache_dir: str | None = None
    """Directory to cache the dataset."""
    hf_access_token: str | None = None
    """Access token for HuggingFace"""
    hf_data_dir: str | None = None
    """Directory for additional data files."""
    hf_train_files: str | None = None
    """Specific training files to use"""
    hf_eval_files: str | None = None
    """Specific evaluation files to use"""
    hf_eval_split: str | None = "validation"
    """Split to use for evaluation."""
    hf_num_data_processes: int | None = None
    """Number of processes to use for downloading the dataset."""
    hf_num_map_processes: int | None = None
    """Number of processes to use for mapping."""
    train_data_column: str = "text"
    """Column name for training data."""
    eval_data_column: str = "text"
    """Column name for evaluation data."""
    tokenize_train_data: bool = True
    """Whether to tokenize training data."""
    tokenize_eval_data: bool = True
    """Whether to tokenize evaluation data."""
    tokenizer_path: str = "gpt2"
    """Path to the tokenizer."""  # TODO: is this really a path or rather a name?
    add_bos: bool = True
    """Whether to add `beginning of sequence` token."""
    add_eos: bool = True
    """Whether to add `end of sequence` token."""


@dataclass(kw_only=True, frozen=True)
class SyntheticDataConfig(DataConfig):
    """Synthetic dataset configuration."""

    num_train_batches: int = 100
    """Number of samples to generate for synthetic training data."""
    num_val_batches: int = 10
    """Number of samples to generate for synthetic validation data."""
