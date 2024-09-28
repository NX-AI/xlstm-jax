import logging

LOGGER = logging.getLogger(__name__)

try:
    from .mlstm_chunkwise.max_triton_fwbw_v3.triton_fwbw import mlstm_chunkwise_max_triton
except ImportError:
    # If Triton is not available, raise a warning and define a dummy function.
    LOGGER.warning("Triton is not available.")

    def mlstm_chunkwise_max_triton(*args, **kwargs):
        raise NotImplementedError("Triton is not available.")
