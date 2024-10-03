from collections.abc import Iterator
from dataclasses import dataclass

from xlstm_jax.dataset.multihost_dataloading import MultiHostDataLoadIterator


@dataclass
class DataloaderModule:
    train_dataloader: Iterator | MultiHostDataLoadIterator | None = None
    val_dataloader: Iterator | MultiHostDataLoadIterator | None = None
    test_dataloader: Iterator | MultiHostDataLoadIterator | None = None
