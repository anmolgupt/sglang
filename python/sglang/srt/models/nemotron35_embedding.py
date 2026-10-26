"""NVIDIA Nemotron-3.5-Embed: a Pixtral vision tower and Mistral3 projector on
the bidirectional Nemotron-3-Embed text model, with normalized mean pooling."""

from array import array
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector

from sglang.srt.environ import envs
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.nemotron3_embedding import EmbeddingNemotron3Model
from sglang.srt.models.pixtral import PixtralHFVisionModel
from sglang.srt.utils import add_prefix


class Nemotron35MultiModalProjector(Mistral3MultiModalProjector):
    """The HF Mistral3 projector with SGLang linears, so a quantized checkpoint
    (e.g. NVFP4 ``multi_modal_projector.linear_1/2``) loads; the norm and the
    patch merger stay as in HF."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config)
        bias = config.multimodal_projector_bias
        self.linear_1 = ReplicatedLinear(
            self.linear_1.in_features,
            self.linear_1.out_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_1", prefix),
        )
        self.linear_2 = ReplicatedLinear(
            self.linear_2.in_features,
            self.linear_2.out_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_2", prefix),
        )

    def forward(self, image_features: torch.Tensor, image_sizes) -> torch.Tensor:
        image_features = self.norm(image_features)
        image_features = self.patch_merger(image_features, image_sizes)
        hidden_states, _ = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class EmbeddingNemotron35Model(nn.Module):
    """Nemotron-3.5-Embed: image and text inputs share one bidirectional pass."""

    # Fused SGLang modules -> checkpoint modules, so quantization configs that
    # name the unfused projections (compressed-tensors NVFP4) match them. The
    # vision attention output projection is `attention.proj` here, `o_proj`
    # in the checkpoint.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "attention.proj": ["attention.o_proj"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vision_tower = PixtralHFVisionModel(
            config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vision_tower", prefix),
        )
        self.multi_modal_projector = Nemotron35MultiModalProjector(
            config,
            quant_config=quant_config,
            prefix=add_prefix("multi_modal_projector", prefix),
        )
        self.language_model = EmbeddingNemotron3Model(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

    def pad_input_ids(self, input_ids: array, mm_inputs: MultimodalInputs) -> array:
        return self.vision_tower.pad_input_ids(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # One vision pass for the images of all requests in the batch (in item
        # order), instead of one per request: per-image passes leave the GPU
        # idle between small launches. Passes are capped at
        # SGLANG_NEMOTRON35_VIT_MAX_PATCHES patches to bound activation memory.
        images, sizes = [], []
        for item in items:
            for pixels, (h, w) in zip(item.feature, item.image_sizes):
                images.append(pixels[:, :h, :w])
                sizes.append((h, w))
        patch = self.vision_tower.patch_size
        budget = envs.SGLANG_NEMOTRON35_VIT_MAX_PATCHES.get()
        features, chunk, chunk_patches = [], [], 0
        for i, (h, w) in enumerate(sizes):
            patches = (h // patch) * (w // patch)
            if chunk and chunk_patches + patches > budget:
                features.append(self._image_features(images, sizes, chunk))
                chunk, chunk_patches = [], 0
            chunk.append(i)
            chunk_patches += patches
        features.append(self._image_features(images, sizes, chunk))
        return torch.cat(features, dim=0)

    def _image_features(self, images, sizes, indices: List[int]) -> torch.Tensor:
        chunk_sizes = [sizes[i] for i in indices]
        image_features = self.vision_tower([images[i] for i in indices], chunk_sizes)
        return self.multi_modal_projector(image_features.squeeze(0), chunk_sizes)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={Modality.IMAGE: self.get_image_feature},
            positions=positions,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # The checkpoint stores the submodules at the top level:
        # language_model.*, vision_tower.* and multi_modal_projector.*.
        params_dict = dict(self.multi_modal_projector.named_parameters())
        for name, loaded_weight in weights:
            part, _, rest = name.partition(".")
            if part in ("language_model", "vision_tower"):
                getattr(self, part).load_weights([(rest, loaded_weight)])
            elif part == "multi_modal_projector":
                param = params_dict[rest]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                raise ValueError(f"Unexpected Nemotron-3.5-Embed weight: {name}")


EntryClass = [EmbeddingNemotron35Model]
