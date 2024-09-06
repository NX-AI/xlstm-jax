# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Poeppel
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
