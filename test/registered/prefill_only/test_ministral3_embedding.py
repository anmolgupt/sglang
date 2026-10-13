"""Offline semantic regression for a tiny bidirectional MEAN checkpoint."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import Ministral3Config, Ministral3Model, PreTrainedTokenizerFast

from sglang import Engine
from sglang.srt.layers.pooler import PoolingType, pool_hidden_states
from sglang.srt.model_loader.utils import get_model_architecture
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=100, suite="stage-b-test-1-gpu-small-amd")


class TestMinistral3Embedding(CustomTestCase):
    def test_bidirectional_mean_and_batch_order_without_embedding_flag(self):
        self.check_checkpoint(default_cuda=False)

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.is_bf16_supported(),
        "default CUDA kernel test; portable FP32 test also runs on ROCm",
    )
    def test_default_cuda_bf16_bidirectional_mean(self):
        self.check_checkpoint(default_cuda=True)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is None,
        "NVIDIA CUDA segment_reduce optimization",
    )
    def test_cuda_mean_short_after_long(self):
        first = torch.ones((4096, 4), device="cuda", dtype=torch.bfloat16)
        second = torch.full((8, 4), 1 / 65536, device="cuda", dtype=torch.bfloat16)
        for chunks in ((first, second), (second, first)):
            lengths = [len(row) for row in chunks]
            batch = SimpleNamespace(
                extend_seq_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
                extend_seq_lens_cpu=lengths,
            )
            with patch.object(
                torch, "segment_reduce", wraps=torch.segment_reduce
            ) as reduce:
                actual = pool_hidden_states(
                    PoolingType.MEAN,
                    torch.cat(chunks),
                    batch,
                    mean_pooling_dtype=torch.float32,
                )
            reduce.assert_called_once()
            expected = torch.stack([row.float().mean(0) for row in chunks])
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def check_checkpoint(self, *, default_cuda):
        # FP32/torch-native isolates semantics. Default CUDA/BF16 exercises the
        # real backend/fused-op/graph selection with a separate cosine gate.
        torch.manual_seed(42)
        config = Ministral3Config(
            vocab_size=128,
            # Production-compatible head size for default FlashInfer/FA kernels;
            # head_dim=32 works with torch_native but not the ragged CUDA path.
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=128,
            max_position_embeddings=128,
            original_max_position_embeddings=128,
            initializer_range=0.1,
            architectures=["Ministral3Model"],
            is_causal=False,
            pooling="avg",
            normalize=True,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        config._attn_implementation = "sdpa"
        hf_model = Ministral3Model(config).eval()
        tokenizer_backend = Tokenizer(
            WordLevel({f"t{i}": i for i in range(128)}, unk_token="t0")
        )
        tokenizer_backend.pre_tokenizer = WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_backend,
            unk_token="t0",
            pad_token="t0",
            bos_token="t1",
            eos_token="t2",
        )
        prompts = ["t3 t4", "t5 t6 t7 t8 t9", " ".join(["t10", "t11"] * 24)]
        references = []
        with torch.no_grad():
            for prompt in prompts:
                tokens = tokenizer(prompt, return_tensors="pt")
                hidden = hf_model(**tokens).last_hidden_state
                references.append(F.normalize(hidden.mean(dim=1), dim=-1)[0])
        with tempfile.TemporaryDirectory() as model_dir:
            hf_model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            del hf_model
            for implementation in ("sglang", "transformers"):
                environment = (
                    {} if default_cuda else {"SGLANG_FORCE_FUSED_OP_BACKEND": "torch"}
                )
                with (
                    self.subTest(implementation=implementation),
                    patch.dict(os.environ, environment),
                ):
                    if default_cuda:
                        os.environ.pop("SGLANG_FORCE_FUSED_OP_BACKEND", None)
                    self.check_backend(
                        model_dir,
                        implementation,
                        prompts,
                        references,
                        default_cuda=default_cuda,
                    )

    def check_backend(
        self, model_dir, implementation, prompts, references, *, default_cuda
    ):
        options = {} if default_cuda else {"attention_backend": "torch_native"}
        engine = Engine(
            model_path=model_dir,
            model_impl=implementation,
            dtype="bfloat16" if default_cuda else "float32",
            mem_fraction_static=0.15,
            log_level="info",
            **options,
            # No is_embedding flag or attention/pooling overrides:
            # startup must infer the task and cache policy correctly.
        )
        try:
            model_config = engine.tokenizer_manager.model_config
            self.assertTrue(model_config.is_embedding_nemotron3)
            self.assertFalse(model_config.is_generation)
            self.assertEqual(model_config.hf_config.architectures, ["Ministral3Model"])
            _, resolved_arch = get_model_architecture(model_config)
            self.assertEqual(
                resolved_arch,
                "EmbeddingNemotron3Model"
                if implementation == "sglang"
                else "TransformersEmbeddingModel",
            )
            if default_cuda:
                self.assertEqual(
                    engine.tokenizer_manager.config_value(
                        "cuda_graph_config"
                    ).prefill.backend,
                    "breakable",
                )
            for order in ([0], [1], [2], [0, 1, 2], [2, 1, 0]):
                results = engine.encode([prompts[i] for i in order])
                self.assertEqual(len(results), len(order))
                for result, index in zip(results, order):
                    actual = torch.tensor(result["embedding"])
                    self.assertEqual(actual.shape, references[index].shape)
                    self.assertTrue(torch.isfinite(actual).all())
                    if default_cuda:
                        cosine = F.cosine_similarity(
                            actual, references[index], dim=0
                        ).item()
                        print(
                            f"BF16 {implementation} order={order} index={index}: "
                            f"cosine={cosine:.9f} max_abs={(actual - references[index]).abs().max().item():.9f}",
                            flush=True,
                        )
                        self.assertGreaterEqual(cosine, 0.999)
                        self.assertAlmostEqual(actual.norm().item(), 1.0, delta=0.005)
                    else:
                        torch.testing.assert_close(
                            actual, references[index], atol=2e-5, rtol=2e-4
                        )
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
