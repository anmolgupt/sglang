"""Qwen-VL image preprocessing deferred to the vision encoder's GPU.

With ``--image-processor-backend scheduler``, a tokenizer worker only decodes
an image and plans it: the resized size and the number of patches follow from
the image size alone (``smart_resize``). It sends the uint8 image, and the
model resizes, normalizes and patchifies it on its own GPU right before the
vision tower (``finish_deferred_images``). No tokenizer worker opens a CUDA
context, and the transfer is uint8 pixels instead of FP32 patches.

The resize is Pillow's bicubic in fixed point (``mm.bicubic_resize``), so the
result matches the PIL image processor.
"""

from typing import Union

import msgspec
import numpy as np
import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

DEFERRED_PREPROCESSING_KEY = "qwen_vl_deferred_preprocessing"


class QwenVLDeferredPreprocessing(msgspec.Struct, frozen=True):
    """What the model needs to finish one deferred image."""

    resized_height: int
    resized_width: int
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]

    @property
    def grid_thw(self) -> torch.Tensor:
        """[1, 3] grid (t, h, w) in patches, as the image processor returns it."""
        return torch.tensor(
            [
                [
                    1,
                    self.resized_height // self.patch_size,
                    self.resized_width // self.patch_size,
                ]
            ],
            dtype=torch.long,
        )


def can_defer(image_processor) -> bool:
    """Whether finish_deferred_images reproduces this image processor: resize,
    rescale by 1/255 and normalize with RGB mean/std, then patchify."""
    return (
        all(
            getattr(image_processor, flag, False)
            for flag in ("do_resize", "do_rescale", "do_normalize")
        )
        and hasattr(image_processor, "merge_size")
        and abs(image_processor.rescale_factor * 255 - 1) < 1e-6
        and len(image_processor.image_mean) == 3
        and len(image_processor.image_std) == 3
    )


def plan_image(
    image_processor, *, height: int, width: int, overrides: dict
) -> QwenVLDeferredPreprocessing:
    """Resize plan of one image, from its size and the processor's settings.

    ``overrides`` are the server's image processor kwargs (``--mm-process-config``);
    ``min_pixels`` / ``max_pixels`` there take precedence, as in a processor call.
    """
    size = overrides.get("size") or image_processor.size
    min_pixels = overrides.get("min_pixels", _size_field(size, "shortest_edge"))
    max_pixels = overrides.get("max_pixels", _size_field(size, "longest_edge"))
    patch_size = overrides.get("patch_size", image_processor.patch_size)
    merge_size = overrides.get("merge_size", image_processor.merge_size)
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return QwenVLDeferredPreprocessing(
        resized_height=resized_height,
        resized_width=resized_width,
        patch_size=patch_size,
        temporal_patch_size=image_processor.temporal_patch_size,
        merge_size=merge_size,
        image_mean=tuple(float(v) for v in image_processor.image_mean),
        image_std=tuple(float(v) for v in image_processor.image_std),
    )


def _size_field(size, key: str) -> int:
    # A dict in processor kwargs, a SizeDict on the image processor.
    return size[key] if isinstance(size, dict) else getattr(size, key)


def to_hwc_uint8(image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
    """RGB image as a contiguous uint8 [H, W, 3] CPU tensor."""
    if isinstance(image, Image.Image):
        return torch.from_numpy(np.asarray(image.convert("RGB")).copy())
    if image.dtype != torch.uint8 or image.ndim != 3:
        raise ValueError(
            f"expected a uint8 image tensor, got {image.dtype} {tuple(image.shape)}"
        )
    if image.shape[-1] != 3 and image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    return image.cpu().contiguous()


def finish_deferred_images(items, device: torch.device) -> None:
    """Replace each deferred item's uint8 image with its patches, on ``device``."""
    for item in items:
        config = item.model_specific_data.pop(DEFERRED_PREPROCESSING_KEY, None)
        if config is None:
            continue
        if not isinstance(config, QwenVLDeferredPreprocessing):
            # msgpack IPC (SGLANG_USE_PICKLE_IPC=0) delivers the struct as a dict.
            config = msgspec.convert(config, QwenVLDeferredPreprocessing)
        image = item.feature.to(device, non_blocking=True)
        item.feature = preprocess_image(image=image, config=config)


def preprocess_image(
    *, image: torch.Tensor, config: QwenVLDeferredPreprocessing
) -> torch.Tensor:
    """Resize, rescale, normalize and patchify one CUDA uint8 [H, W, 3] image.

    Returns float32 ``[grid_h * grid_w, 3 * temporal_patch_size * patch_size**2]``,
    the layout of the Qwen2-VL image processor's ``pixel_values``.
    """
    from sglang.kernels.ops.mm.process import bicubic_resize

    x = bicubic_resize(
        image,
        config.resized_height,
        config.resized_width,
        mean=config.image_mean,
        std=config.image_std,
    )  # float32 [H, W, 3], (pixel / 255 - mean) / std
    height, width, channels = x.shape
    p, m, t = config.patch_size, config.merge_size, config.temporal_patch_size
    grid_h, grid_w = height // p, width // p
    x = x.view(grid_h // m, m, p, grid_w // m, m, p, channels)
    x = x.permute(0, 3, 1, 4, 6, 2, 5)  # (gh/m, gw/m, m, m, C, p, p)
    x = x.unsqueeze(5).expand(*x.shape[:5], t, p, p)  # one frame, repeated
    return x.reshape(grid_h * grid_w, channels * t * p * p)
