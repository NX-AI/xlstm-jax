from model.components.conv import test_causal_conv1d
from model.components.linear_headwise import test_linear_headwise
from model.components.ln import test_ln
from model.blocks.mlstm.backend.simple import test_mLSTMBackend
from model_pytorch.blocks.mlstm.backend.simple import parallel_stabilized_simple, recurrent_step_stabilized_simple
from model.blocks.mlstm.layer import test_mLSTMLayer
from model.blocks.mlstm.block import test_mLSTMBlock
from model.blocks.xlstm_block import test_xLSTMBlock
from model.xlstm_block_stack import test_xLSTMBlockStack
from model.xlstm_lm_model import test_xLSTMLMModel

if __name__ == "__main__":
    test_causal_conv1d()
    test_linear_headwise()
    test_ln()
    test_mLSTMLayer()
    test_mLSTMBackend(
        parallel_stabilized_simple_torch=parallel_stabilized_simple,
        recurrent_step_stabilized_simple_torch=recurrent_step_stabilized_simple,
    )
    test_mLSTMBlock()
    test_xLSTMBlock()
    test_xLSTMBlockStack()
    test_xLSTMLMModel()
