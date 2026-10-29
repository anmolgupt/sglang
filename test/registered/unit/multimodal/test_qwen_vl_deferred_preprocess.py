"""CPU side of Qwen-VL deferred image preprocessing: the resize plan and image
conversion done in the tokenizer worker."""

import unittest

import numpy as np
import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
    Qwen2VLImageProcessorPil,
)

from sglang.srt.multimodal.qwen_vl_image_processing import (
    can_defer,
    plan_image,
    to_hwc_uint8,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPlanImage(CustomTestCase):
    def setUp(self):
        self.processor = Qwen2VLImageProcessorPil()

    def _processor_grid(self, height, width, **kwargs):
        image = Image.new("RGB", (width, height))
        return self.processor(images=[image], return_tensors="pt", **kwargs)[
            "image_grid_thw"
        ]

    def test_grid_matches_the_image_processor(self):
        for height, width in ((40, 50), (448, 672), (1100, 850), (2200, 1700)):
            with self.subTest(height=height, width=width):
                config = plan_image(
                    self.processor, height=height, width=width, overrides={}
                )
                self.assertTrue(
                    torch.equal(config.grid_thw, self._processor_grid(height, width))
                )
                self.assertEqual(config.resized_height % 28, 0)
                self.assertEqual(config.resized_width % 28, 0)

    def test_server_pixel_limits_take_precedence(self):
        overrides = {"max_pixels": 256 * 28 * 28}
        config = plan_image(
            self.processor, height=2200, width=1700, overrides=overrides
        )
        self.assertLessEqual(
            config.resized_height * config.resized_width, 256 * 28 * 28
        )
        expected = self._processor_grid(
            2200, 1700, size={"shortest_edge": 56 * 56, "longest_edge": 256 * 28 * 28}
        )
        self.assertTrue(torch.equal(config.grid_thw, expected))

    def test_defers_only_what_it_reproduces(self):
        self.assertTrue(can_defer(self.processor))
        self.assertFalse(can_defer(Qwen2VLImageProcessorPil(do_normalize=False)))
        self.assertFalse(can_defer(Qwen2VLImageProcessorPil(rescale_factor=1 / 127.5)))


class TestToHWCUint8(CustomTestCase):
    def test_pil_and_tensor_inputs(self):
        array = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        expected = torch.from_numpy(array)
        self.assertTrue(torch.equal(to_hwc_uint8(Image.fromarray(array)), expected))
        self.assertTrue(
            torch.equal(to_hwc_uint8(Image.fromarray(array).convert("RGBA")), expected)
        )
        self.assertTrue(torch.equal(to_hwc_uint8(expected.permute(2, 0, 1)), expected))
        self.assertTrue(to_hwc_uint8(expected.permute(2, 0, 1)).is_contiguous())
        with self.assertRaises(ValueError):
            to_hwc_uint8(expected.float())


if __name__ == "__main__":
    unittest.main()
