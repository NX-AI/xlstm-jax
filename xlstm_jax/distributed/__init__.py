from .array_utils import fold_rng_over_axis, split_array_over_mesh
from .data_parallel import shard_module_params, sync_gradients
from .pipeline_parallel import PipelineModule, execute_pipeline
from .single_gpu import accumulate_gradients
from .tensor_parallel import (
    ModelParallelismWrapper,
    TPAsyncDense,
    TPDense,
    async_gather,
    async_gather_bidirectional,
    async_gather_split,
    async_scatter,
    async_scatter_split,
)
from .xla_utils import set_XLA_flags, simulate_CPU_devices
