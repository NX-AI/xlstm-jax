#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import math
from dataclasses import dataclass


@dataclass
class UpProjConfigMixin:
    proj_factor: float = None  # will be overridden by subclasses
    round_proj_up_dim_up: bool = True
    round_proj_up_to_multiple_of: int = 64

    # internal
    _proj_up_dim: int = None  # will be computed from embedding_dim and proj_factor

    def _set_proj_up_dim(self, embedding_dim: int) -> None:
        if self.proj_factor is not None and embedding_dim is not None:
            proj_up_dim = self.proj_factor * embedding_dim
            multiple_of_multiplier = proj_up_dim / self.round_proj_up_to_multiple_of
            if self.round_proj_up_dim_up:
                multiple_of_multiplier = math.ceil(multiple_of_multiplier)
            else:
                multiple_of_multiplier = math.floor(multiple_of_multiplier)

            self._proj_up_dim = int(multiple_of_multiplier * self.round_proj_up_to_multiple_of)
