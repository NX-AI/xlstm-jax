from dataclasses import dataclass

from xlstm_jax.configs import ConfigDict


@dataclass(kw_only=True, frozen=True)
class DataConfig(ConfigDict):
    """Base data configuration

    Attributes:
        global_batch_size: int: Global batch size for training.
        max_target_length: Optional[int]: Maximum length of the target sequence.
        global_batch_size_to_train_on: int: Global batch size to train on. If
            -1, it will be set to global_batch_size. Otherwise, should be smaller
            than global_batch_size to indicate a reduced batch size for training.
        global_batch_size_for_eval: int: Global batch size for evaluation. If
            -1, it will be set to global_batch_size. Can be larger or smaller than
            global_batch_size, but at least 1 per device.
        expansion_factor_real_data: int: Expansion factor for real data. If -1,
            all hosts will load real data.
        data_shuffle_seed: int: Seed for data shuffling.
    """

    global_batch_size: int
    max_target_length: int | None = None
    global_batch_size_to_train_on: int = -1
    global_batch_size_for_eval: int = -1
    expansion_factor_real_data: int = -1
    data_shuffle_seed: int = 42


@dataclass(kw_only=True, frozen=True)
class HFDataConfig(DataConfig):
    """HuggingFace dataset configuration

    Attributes:
        num_train_epochs: the data iterator for training data needs to
                          know how many epochs to iterate over
                          because of the way grain implements shuffling
        hf_path: str: Path to the dataset on HuggingFace.
        hf_cache_dir: Optional[str]: Directory to cache the dataset.
        hf_access_token: Optional[str]: Access token for HuggingFace.
        hf_data_dir: Optional[str]: Directory for additional data files.
        hf_train_files: Optional[str]: Specific training files to use.
        hf_eval_files: Optional[str]: Specific evaluation files to use.
        hf_eval_split: Optional[str]: Split to use for evaluation.
        hf_num_data_processes: Optional[int]: Number of processes to use for
            downloading the dataset.
        hf_num_map_processes: Optional[int]: Number of processes to use for mapping.
        train_data_column: str: Column name for training data.
        eval_data_column: str: Column name for evaluation data.
        tokenize_train_data: bool: Whether to tokenize training data.
        tokenize_eval_data: bool: Whether to tokenize evaluation data.
        tokenizer_path: str: Path to the tokenizer.
        add_bos: bool: Whether to add beginning of sequence token.
        add_eos: bool: Whether to add end of sequence token.
    """

    num_train_epochs: int
    hf_path: str
    hf_cache_dir: str | None = None
    hf_access_token: str | None = None
    hf_data_dir: str | None = None
    hf_train_files: str | None = None
    hf_eval_files: str | None = None
    hf_eval_split: str | None = "validation"
    hf_num_data_processes: int | None = None
    hf_num_map_processes: int | None = None
    train_data_column: str = "text"
    eval_data_column: str = "text"
    tokenize_train_data: bool = True
    tokenize_eval_data: bool = True
    tokenizer_path: str = "gpt2"
    add_bos: bool = True
    add_eos: bool = True


@dataclass(kw_only=True, frozen=True)
class SyntheticDataConfig(DataConfig):
    """Synthetic dataset configuration

    Attributes:
        num_train_batches: Number of samples to generate for synthetic training data.
        num_val_batches: Number of samples to generate for synthetic validation data.
    """

    num_train_batches: int = 100
    num_val_batches: int = 10
