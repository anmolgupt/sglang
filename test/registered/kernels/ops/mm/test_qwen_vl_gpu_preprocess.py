# SPDX-License-Identifier: Apache-2.0
"""Qwen-VL images preprocessed on the GPU (--image-processor-backend scheduler)
match the Qwen2-VL PIL image processor: grids and pixel values."""

import unittest
from types import SimpleNamespace

import msgspec
import numpy as np
import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
    Qwen2VLImageProcessorPil,
)

from sglang.srt.multimodal.qwen_vl_image_processing import (
    DEFERRED_PREPROCESSING_KEY,
    finish_deferred_images,
    plan_image,
    preprocess_image,
    to_hwc_uint8,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _page(height, width, seed):
    """Document-like content: text-like strokes on white, plus a gradient."""
    rng = np.random.default_rng(seed)
    page = np.full((height, width, 3), 255, dtype=np.uint8)
    rows = rng.integers(0, height, size=height // 4)
    cols = rng.integers(0, width, size=height // 4)
    page[rows, cols] = rng.integers(0, 80, size=(len(rows), 3))
    page[:, : width // 3] = np.linspace(0, 255, width // 3, dtype=np.uint8)[
        None, :, None
    ]
    return Image.fromarray(page)


class TestQwenVLDeferredPreprocess(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = Qwen2VLImageProcessorPil()

    def _reference(self, image):
        out = self.processor(images=[image], return_tensors="pt")
        return out["pixel_values"], out["image_grid_thw"]

    def test_matches_pil_processor(self):
        # Upscaled to min_pixels, kept, and shrunk to max_pixels.
        for height, width in ((40, 50), (448, 672), (1100, 850), (2200, 1700)):
            with self.subTest(height=height, width=width):
                image = _page(height, width, seed=height + width)
                ref_pixels, ref_grid = self._reference(image)
                config = plan_image(
                    self.processor, height=height, width=width, overrides={}
                )
                self.assertTrue(torch.equal(config.grid_thw, ref_grid))
                pixels = preprocess_image(
                    image=to_hwc_uint8(image).cuda(), config=config
                ).cpu()
                self.assertEqual(pixels.shape, ref_pixels.shape)
                torch.testing.assert_close(
                    pixels, ref_pixels.float(), rtol=0, atol=1e-5
                )

    def test_finish_replaces_the_image_and_drops_the_plan(self):
        image = _page(300, 500, seed=3)
        ref_pixels, _ = self._reference(image)
        config = plan_image(self.processor, height=300, width=500, overrides={})
        # msgpack IPC delivers the plan as a plain dict.
        for plan in (config, msgspec.to_builtins(config)):
            with self.subTest(plan=type(plan).__name__):
                item = SimpleNamespace(
                    feature=to_hwc_uint8(image),
                    model_specific_data={DEFERRED_PREPROCESSING_KEY: plan},
                )
                finish_deferred_images([item], device=torch.device("cuda"))
                self.assertNotIn(DEFERRED_PREPROCESSING_KEY, item.model_specific_data)
                self.assertTrue(item.feature.is_cuda)
                torch.testing.assert_close(
                    item.feature.cpu(), ref_pixels.float(), rtol=0, atol=1e-5
                )

    def test_items_without_a_plan_are_untouched(self):
        feature = torch.randn(16, 1176)
        item = SimpleNamespace(feature=feature, model_specific_data={})
        finish_deferred_images([item], device=torch.device("cuda"))
        self.assertIs(item.feature, feature)


if __name__ == "__main__":
    unittest.main()
