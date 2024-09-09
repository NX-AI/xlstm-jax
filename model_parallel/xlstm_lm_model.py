# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from dataclasses import dataclass

from flax import linen as nn
import jax
import jax.numpy as jnp
from functools import partial

from .xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from .blocks.mlstm.block import mLSTMBlockConfig, mLSTMLayerConfig
from .utils import ParallelConfig, prepare_module

from distributed.pipeline_parallel import ModelParallelismWrapper
from distributed.data_parallel import shard_module_params
from distributed.tensor_parallel_transformer import split_array_over_mesh

@dataclass
class xLSTMLMModelConfig(xLSTMBlockStackConfig):
    vocab_size: int = -1
    tie_weights: bool = False
    weight_decay_on_embedding: bool = False
    add_embedding_dropout: bool = False
    parallel: ParallelConfig | None = None


class xLSTMLMModel(nn.Module):
    config: xLSTMLMModelConfig

    @nn.compact
    def __call__(self, idx: jax.Array, train: bool = False) -> jax.Array:
        # Embedding
        tp_size = jax.lax.psum(1, self.config.parallel.model_axis_name)
        embed_fn = partial(
            ModelParallelismWrapper,
            module_fn=partial(
                nn.Embed,
                num_embeddings=self.config.vocab_size,
                features=self.config.embedding_dim // tp_size,
                dtype=self.config.dtype,
                name="embed",
            ),
            model_axis_name=self.config.parallel.model_axis_name,
            name="token_embedding",
        )
        embed_fn = prepare_module(
            embed_fn,
            "Embed",
            config=self.config.parallel
        )
        x = embed_fn()(idx)
        if self.config.add_embedding_dropout:
            x = nn.Dropout(rate=self.config.dropout)(x, deterministic=not train)
        # BlockStack
        stack_fn = prepare_module(
            xLSTMBlockStack,
            "BlockStack",
            config=self.config.parallel
        )
        x = stack_fn(config=self.config, name="xlstm_block_stack")(x, train=train)
        # LMHead
        pred_fn = prepare_module(
            partial(TPOutputLayer, config=self.config, name="lm_head"),
            "LMHead",
            config=self.config.parallel
        )
        logits = pred_fn()(x)
        return logits


class TPOutputLayer(nn.Module):
    config: xLSTMLMModelConfig

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Gather outputs over feature dimension and split over sequence length.
        x = jax.lax.all_gather(x, axis_name=self.config.parallel.model_axis_name, axis=-1, tiled=True)
        x = split_array_over_mesh(x, axis_name=self.config.parallel.model_axis_name, split_axis=1)
        # Shard parameters over model axis.
        norm_fn = shard_module_params(
            nn.LayerNorm,
            axis_name=self.config.parallel.model_axis_name,
            min_weight_size=self.config.parallel.fsdp_min_weight_size,
        )
        dense_fn = shard_module_params(
            nn.Dense,
            axis_name=self.config.parallel.model_axis_name,
            min_weight_size=self.config.parallel.fsdp_min_weight_size,
        )
        # Apply normalization and output layer.
        x = norm_fn(dtype=self.config.dtype, name="out_norm")(x)
        x = dense_fn(
            features=self.config.vocab_size,
            use_bias=False,
            dtype=jnp.float32,
            name="output_layer",
        )(x)
        return x