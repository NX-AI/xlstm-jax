#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from collections.abc import Iterator
from dataclasses import dataclass

from xlstm_jax.dataset.multihost_dataloading import MultiHostDataLoadIterator

DataIterator = Iterator | MultiHostDataLoadIterator


@dataclass
class DataloaderModule:
    train_dataloader: DataIterator | None = None
    val_dataloader: DataIterator | dict[str, DataIterator] | None = None
    test_dataloader: DataIterator | dict[str, DataIterator] | None = None
