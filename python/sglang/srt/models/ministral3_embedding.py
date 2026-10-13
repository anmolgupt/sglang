"""Native Ministral3 bare-backbone embeddings, without an unused LM head."""

from collections.abc import Iterable

import torch
from torch import nn

from sglang.srt.configs.embedding_model_spec import (
    resolve_embedding_normalize,
    resolve_transformers_pooling,
)
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.models.ministral3 import Ministral3ForCausalLM
from sglang.srt.models.ministral3 import Ministral3Model as Ministral3Backbone
from sglang.srt.utils import add_prefix


class EmbeddingNemotron3Model(Ministral3ForCausalLM):
    """Reuse native fused Ministral3 kernels with checkpoint-defined pooling.

    This internal loader entry is selected from embedding metadata, not the
    generic HF backbone name. ``Ministral3Backbone`` remains the native
    unpooled backbone from ``models.ministral3``.
    Bidirectional checkpoints use the existing ENCODER_ONLY attention path.
    Keeping ``model`` as the native backbone permits ordinary prefill graph
    capture, with pooling outside the captured body.
    """

    supports_configured_embedding_pooling = True

    def __init__(self, config, quant_config=None, prefix=""):
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        if getattr(config, "sliding_window", None) is not None:
            raise ValueError(
                "Native EmbeddingNemotron3Model embeddings do not implement sliding_window "
                "attention. Use a checkpoint with sliding_window=null; do not "
                "remove a trained attention window to bypass this check."
            )
        self.model = Ministral3Backbone(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.pp_group = self.model.pp_group
        pooling = PoolingType[resolve_transformers_pooling(config, strict=True).name]
        self.pooler = Pooler(
            pooling_type=pooling,
            normalize=resolve_embedding_normalize(config),
            mean_pooling_dtype=torch.float32 if pooling == PoolingType.MEAN else None,
        )
        if getattr(config, "is_causal", True) is False:
            # Set the policy before any forward/graph capture. Current attention
            # backends inspect this attribute at execution/metadata preparation.
            for layer in self.model.layers:
                if hasattr(layer, "self_attn"):
                    layer.self_attn.attn.attn_type = AttentionType.ENCODER_ONLY
        self.capture_aux_hidden_states = False
        self.stacked_params_mapping = self._get_stacked_params_mapping()

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        get_embedding=True,
        pp_proxy_tensors=None,
    ):
        if not get_embedding:
            raise ValueError(
                "EmbeddingNemotron3Model serves embeddings, not generation"
            )
        return super().forward(
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            get_embedding=True,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # HF bare-backbone exports omit the CausalLM wrapper's ``model.`` prefix.
        # The native loader owns QKV/gate-up packing and quantized-weight handling.
        def remapped():
            for name, weight in weights:
                if name.startswith("lm_head."):
                    continue
                if name.startswith(("layers.", "embed_tokens.", "norm.")):
                    name = "model." + name
                yield name, weight

        return super().load_weights(remapped())

    def forward_split_prefill(self, *args, **kwargs):
        raise ValueError(
            "EmbeddingNemotron3Model does not support generation split prefill"
        )

    def get_embed_and_head(self):
        raise ValueError(
            "EmbeddingNemotron3Model is an embedding model without an LM head"
        )

    def set_embed_and_head(self, embed, head):
        raise ValueError(
            "EmbeddingNemotron3Model is an embedding model without an LM head"
        )

    def set_eagle3_layers_to_capture(self, layer_ids=None):
        raise ValueError(
            "EmbeddingNemotron3Model does not support speculative generation"
        )

    def set_dflash_layers_to_capture(self, layer_ids):
        raise ValueError(
            "EmbeddingNemotron3Model does not support speculative generation"
        )


EntryClass = EmbeddingNemotron3Model
