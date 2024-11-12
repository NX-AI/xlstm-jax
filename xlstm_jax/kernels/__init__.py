import logging

LOGGER = logging.getLogger(__name__)

try:
    from .mlstm_chunkwise.max_triton_fwbw_v3.triton_fwbw import mlstm_chunkwise_max_triton
    from .mlstm_chunkwise.max_triton_fwbw_v3noslice.triton_fwbw import (
        mlstm_chunkwise_max_triton as mlstm_chunkwise_max_triton_noslice,
    )
    from .mlstm_chunkwise.max_triton_fwbw_v5xlchunksize.triton_fwbw import (
        mlstm_chunkwise_max_triton as mlstm_chunkwise_max_triton_xlchunksize,
    )
    from .mlstm_chunkwise.triton_stablef.triton_fwbw import mlstm_chunkwise_triton_stablef
    from .mlstm_recurrent.triton_fused_fw import (
        recurrent_step_fw as mlstm_recurrent_step_triton_fused,
    )
except Exception as e:
    # If Triton is not available, raise a warning and define a dummy function.
    LOGGER.warning(
        f"Exception {e} when loading mlstm_kernels, "
        "make sure that Triton is installed and that "
        "you have the xlstm_jax repo in the PYTHONPATH."
    )
    err_msg = (
        f"mlstm_kernels not available (Exception {e}), "
        "make sure that Triton is installed and that "
        "you have the xlstm_jax repo in the PYTHONPATH."
    )

    def mlstm_chunkwise_max_triton(*args, **kwargs):
        raise NotImplementedError(err_msg)

    def mlstm_chunkwise_max_triton_noslice(*args, **kwargs):
        raise NotImplementedError(err_msg)

    def mlstm_chunkwise_triton_stablef(*args, **kwargs):
        raise NotImplementedError(err_msg)

    def mlstm_chunkwise_max_triton_xlchunksize(*args, **kwargs):
        raise NotImplementedError(err_msg)

    def mlstm_recurrent_step_triton_fused(*args, **kwargs):
        raise NotImplementedError(err_msg)
