#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from abc import abstractmethod
from typing import Any

import torch


class mLSTMBackend(torch.nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, i: torch.Tensor, f: torch.Tensor
    ) -> torch.Tensor:
        pass
