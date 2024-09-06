from dataclasses import dataclass, field
from functools import partial

from .layer_factory import create_layer
from .config_utils import NameAndKwargs
from .fwbw import mLSTMfwbw, mLSTMfwbwConfig
from .simple import mLSTMBackendTorch, recurrent_step_stabilized_simple

_mlstm_backend_registry = {
    "parallel_stabilized": mLSTMBackendTorch,
    "fwbw": mLSTMfwbw,
}


@dataclass
class mLSTMBackendNameAndKwargs(NameAndKwargs):
    _registry: dict[str, type] = field(default_factory=lambda: _mlstm_backend_registry)


create_mlstm_backend = partial(
    create_layer, registry=_mlstm_backend_registry, layer_cfg_key="backend"
)
