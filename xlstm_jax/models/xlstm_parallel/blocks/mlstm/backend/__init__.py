from dataclasses import dataclass, field
from functools import partial

from .attention import mLSTMBackendAttention
from .config import mLSTMBackend
from .config_utils import NameAndKwargs
from .fwbw import mLSTMBackendFwbw
from .layer_factory import create_layer
from .simple import mLSTMBackendParallel
from .triton_kernels import mLSTMBackendTriton

_mlstm_backend_registry = {
    "parallel_stabilized": mLSTMBackendParallel,
    "fwbw_stabilized": mLSTMBackendFwbw,
    "triton_kernels": mLSTMBackendTriton,
    "attention": mLSTMBackendAttention,
}


@dataclass
class mLSTMBackendNameAndKwargs(NameAndKwargs):
    _registry: dict[str, type] = field(default_factory=lambda: _mlstm_backend_registry)


create_mlstm_backend = partial(create_layer, registry=_mlstm_backend_registry, layer_cfg_key="backend")
