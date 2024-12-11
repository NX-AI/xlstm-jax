#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from dataclasses import dataclass, field
from functools import partial

from .config_utils import NameAndKwargs
from .fwbw import mLSTMfwbw, mLSTMfwbwConfig
from .layer_factory import create_layer
from .simple import mLSTMBackendTorch, parallel_stabilized_simple, recurrent_step_stabilized_simple

_mlstm_backend_registry = {
    "parallel_stabilized": mLSTMBackendTorch,
    "fwbw": mLSTMfwbw,
}


@dataclass
class mLSTMBackendNameAndKwargs(NameAndKwargs):
    _registry: dict[str, type] = field(default_factory=lambda: _mlstm_backend_registry)


create_mlstm_backend = partial(create_layer, registry=_mlstm_backend_registry, layer_cfg_key="backend")
