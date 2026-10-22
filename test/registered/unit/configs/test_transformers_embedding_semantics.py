import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.embedding_model_spec import (
    AttentionPattern,
    BCGPrefillPolicy,
    EmbeddingTask,
    PoolingStrategy,
    declares_checkpoint_embedding_config,
    resolve_embedding_model_spec,
    resolve_embedding_normalize,
    resolve_transformers_pooling,
    validate_configured_embedding_attention,
    validate_configured_embedding_backend,
)
from sglang.srt.configs.model_config import is_embedding_nemotron3
from sglang.srt.layers.pooler import Pooler, PoolingType, pool_hidden_states
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.models.ministral3_embedding import EmbeddingNemotron3Model
from sglang.srt.models.transformers import TransformersBase, TransformersEmbeddingModel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def config(*, causal=False, multimodal=False, pooling="avg"):
    text = SimpleNamespace(
        is_causal=causal,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=8,
        head_dim=4,
    )
    if multimodal:
        return SimpleNamespace(
            architectures=["Mistral3Model"],
            model_type="mistral3",
            text_config=text,
            pooling=pooling,
        )
    text.architectures = ["Ministral3Model"]
    text.model_type = "ministral3"
    text.pooling = pooling
    return text


def spec(cfg, requested=True, supports=True):
    return resolve_embedding_model_spec(
        cfg.architectures,
        is_embedding_requested=requested,
        is_embedding_gemma=False,
        hf_config=cfg,
        supports_configured_pooling=supports,
        is_embedding_nemotron3=is_embedding_nemotron3(cfg),
    )


class AttentionFlag(torch.nn.Module):
    def __init__(self, causal):
        super().__init__()
        self.is_causal = causal


class TestTransformersEmbeddingSemantics(unittest.TestCase):
    def test_metadata_probe_covers_outer_and_language_fields(self):
        self.assertFalse(declares_checkpoint_embedding_config(SimpleNamespace()))
        self.assertFalse(
            declares_checkpoint_embedding_config(SimpleNamespace(is_causal=True))
        )
        for fields in (
            {"is_causal": False},
            {"pooling": "last"},
            {"pooling_type": "MEAN"},
            {"normalize": False},
        ):
            for cfg in (
                SimpleNamespace(**fields),
                SimpleNamespace(text_config=SimpleNamespace(**fields)),
            ):
                self.assertTrue(declares_checkpoint_embedding_config(cfg))

    def test_native_causal_last_metadata_does_not_warn(self):
        for pooling in ("LAST", "lasttoken", "last_token"):
            cfg = SimpleNamespace(
                architectures=["Qwen2ForCausalLM"], pooling=pooling, is_causal=True
            )
            with self.assertNoLogs(
                "sglang.srt.configs.embedding_model_spec", "WARNING"
            ):
                result = spec(cfg, supports=False)
            self.assertEqual(result.family, "explicit_decoder_embedding")
        cfg.pooling = "weightedmean"
        with self.assertLogs(
            "sglang.srt.configs.embedding_model_spec", "WARNING"
        ) as logs:
            result = spec(cfg, supports=False)
        self.assertIn("Unsupported checkpoint", logs.output[0])
        self.assertEqual(result.pooling, PoolingStrategy.MODEL_DEFINED)

    def test_configured_causal_last_advertises_actual_normalization(self):
        for fields in ({"pooling": "last", "normalize": False}, {"normalize": False}):
            cfg = SimpleNamespace(
                architectures=["SomeDecoder"], is_causal=True, **fields
            )
            result = spec(cfg)
            self.assertEqual(result.family, "configured_pooling_embedding")
            self.assertEqual(result.pooling, PoolingStrategy.LAST)
            self.assertFalse(result.normalize)
            self.assertFalse(result.safe_disable_radix_cache)
            self.assertFalse(result.safe_disable_chunked_prefill)

    def test_selected_backend_must_implement_declared_semantics(self):
        configured = spec(config())
        for model_cls in (EmbeddingNemotron3Model, TransformersEmbeddingModel):
            validate_configured_embedding_backend(configured, model_cls)
        with self.assertRaisesRegex(ValueError, "--model-impl transformers"):
            validate_configured_embedding_backend(configured, AttentionFlag)
        # Existing LAST/generation/native-encoder policies need no new marker.
        validate_configured_embedding_backend(
            spec(SimpleNamespace(architectures=["Qwen2ForCausalLM"]), supports=False),
            AttentionFlag,
        )
        validate_configured_embedding_backend(None, AttentionFlag)

    def test_pooling_metadata_and_precedence(self):
        for name in ("avg", "MEAN", "average"):
            self.assertEqual(
                resolve_transformers_pooling(config(pooling=name)), PoolingStrategy.MEAN
            )
        cfg = config(multimodal=True)
        cfg.pooling_type = "CLS"
        self.assertEqual(resolve_transformers_pooling(cfg), PoolingStrategy.CLS)
        self.assertEqual(
            resolve_transformers_pooling(SimpleNamespace()), PoolingStrategy.LAST
        )
        self.assertEqual(
            resolve_transformers_pooling(SimpleNamespace(pooling_type="last")),
            PoolingStrategy.LAST,
        )
        with self.assertRaisesRegex(
            ValueError, "Unsupported checkpoint embedding pooling"
        ):
            resolve_transformers_pooling(config(pooling="unknown"), strict=True)

    def test_bidirectional_metadata_flat_and_nested(self):
        for multimodal in (False, True):
            result = spec(config(multimodal=multimodal))
            self.assertEqual(result.attention, AttentionPattern.BIDIRECTIONAL)
            self.assertEqual(result.pooling, PoolingStrategy.MEAN)
            self.assertEqual(result.supports_multimodal, multimodal)
            self.assertTrue(result.safe_disable_radix_cache)
            self.assertTrue(result.safe_disable_chunked_prefill)
            self.assertFalse(result.safe_disable_kv_cache)
            self.assertEqual(result.bcg_prefill_policy, BCGPrefillPolicy.DEFAULT)
            unrequested = spec(config(multimodal=multimodal), False)
            self.assertEqual(
                unrequested.task,
                EmbeddingTask.NONE if multimodal else EmbeddingTask.EMBED,
            )
            self.assertEqual(unrequested.auto_enable_embedding, not multimodal)

    def test_causal_defaults_and_native_regressions(self):
        cfg = config(causal=True, pooling="LAST")
        result = spec(cfg)
        self.assertEqual(result.attention, AttentionPattern.CAUSAL)
        self.assertFalse(result.safe_disable_radix_cache)
        cfg.pooling = "mean"
        result = spec(cfg)
        self.assertEqual(result.attention, AttentionPattern.CAUSAL)
        self.assertTrue(result.safe_disable_radix_cache)
        # Existing native architectures retain their own pooling policies.
        cfg.architectures = ["BertModel"]
        self.assertEqual(spec(cfg).pooling, PoolingStrategy.CLS)
        cfg.architectures = ["BertForSequenceClassification"]
        self.assertEqual(spec(cfg).task, EmbeddingTask.CLASSIFY)

    def test_language_attention_not_vision_attention(self):
        for causal in (False, True):
            for multimodal in (False, True):
                cfg = config(causal=causal, multimodal=multimodal)
                base = TransformersEmbeddingModel.__new__(TransformersEmbeddingModel)
                torch.nn.Module.__init__(base)
                base.config = cfg
                base.text_config = getattr(cfg, "text_config", cfg)
                # HF language modules can declare True even for a non-causal config.
                base.model = torch.nn.Sequential(AttentionFlag(True))
                if multimodal:
                    base.model.append(AttentionFlag(False))
                base.start_layer, base.end_layer = 0, 1
                base.quant_config = None
                with patch(
                    "sglang.srt.models.transformers.get_parallel",
                    return_value=SimpleNamespace(tp_size=1),
                ):
                    attention = base._create_attention_instances()[0]
                expected = (
                    AttentionType.DECODER if causal else AttentionType.ENCODER_ONLY
                )
                self.assertEqual(attention.attn_type, expected)
                self.assertFalse(attention.is_cross_attention)

    def test_full_prefill_policies(self):
        from sglang.srt.arg_groups import model_hook

        for causal, pooling, expected in (
            (False, "avg", True),
            (True, "mean", True),
            (True, "LAST", False),
        ):
            cfg = config(causal=causal, pooling=pooling)
            server = SimpleNamespace(model_path="/fixture", is_embedding=True)
            model = SimpleNamespace(
                hf_config=cfg,
                embedding_model_spec=spec(cfg),
                is_multimodal=False,
                joint_head_config=None,
            )
            with (
                patch.object(model_hook, "resolving_view", return_value=server),
                patch.object(model_hook, "model_config_of", return_value=model),
                patch.object(model_hook, "declare_resolution") as declare,
            ):
                model_hook.handle_model_capability_adjustments(server)
            changes = {
                key: value
                for call in declare.call_args_list
                for key, value in call.kwargs.items()
            }
            self.assertEqual(
                changes,
                (
                    {"disable_radix_cache": True, "chunked_prefill_size": -1}
                    if expected
                    else {}
                ),
            )
            # Do not collapse duplicate declarations and accidentally hide
            # redundant policy blocks carried over from an older main.
            self.assertEqual(len(declare.call_args_list), 2 if expected else 0)

    def test_existing_native_metadata_preserves_model_defined_behavior(self):
        for architecture in (
            "Qwen2ForCausalLM",
            "Qwen3ForCausalLM",
            "Gemma3ForCausalLM",
        ):
            for pooling in ("MEAN", "CLS", "lasttoken", "eos", "weightedmean"):
                cfg = SimpleNamespace(
                    architectures=[architecture], pooling=pooling, is_causal=False
                )
                with self.assertLogs(
                    "sglang.srt.configs.embedding_model_spec", level="WARNING"
                ) as logs:
                    result = spec(cfg, supports=False)
                self.assertEqual(result.family, "explicit_decoder_embedding")
                self.assertEqual(result.pooling, PoolingStrategy.MODEL_DEFINED)
                self.assertEqual(result.attention, AttentionPattern.CAUSAL)
                self.assertFalse(result.safe_disable_radix_cache)
                self.assertFalse(result.safe_disable_chunked_prefill)
                self.assertIn("not applied", logs.output[0])
                validate_configured_embedding_backend(result, AttentionFlag)

    def test_legacy_unknown_pooling_warns_but_new_bare_contract_rejects(self):
        cfg = SimpleNamespace(pooling="weightedmean")
        with self.assertLogs(
            "sglang.srt.configs.embedding_model_spec", level="WARNING"
        ) as logs:
            self.assertEqual(resolve_transformers_pooling(cfg), PoolingStrategy.LAST)
        self.assertIn("legacy LAST", logs.output[0])
        with self.assertRaisesRegex(ValueError, "Unsupported checkpoint"):
            resolve_transformers_pooling(cfg, strict=True)

    def test_unambiguous_pooling_aliases(self):
        for name, expected in (
            ("lasttoken", PoolingStrategy.LAST),
            ("last_token", PoolingStrategy.LAST),
            ("mean_tokens", PoolingStrategy.MEAN),
            ("cls_token", PoolingStrategy.CLS),
        ):
            self.assertEqual(
                resolve_transformers_pooling(SimpleNamespace(pooling=name)), expected
            )

    def test_configured_causal_cls_needs_capability_and_full_prefill(self):
        cfg = config(causal=True, pooling="CLS")
        result = spec(cfg)
        self.assertTrue(result.safe_disable_radix_cache)
        self.assertTrue(result.safe_disable_chunked_prefill)
        with self.assertRaisesRegex(ValueError, "--model-impl transformers"):
            validate_configured_embedding_backend(result, AttentionFlag)

    def test_normalization_metadata_precedence(self):
        cfg = config(multimodal=True)
        cfg.text_config.normalize = False
        self.assertFalse(resolve_embedding_normalize(cfg))
        self.assertFalse(spec(cfg).normalize)
        cfg.normalize = True
        self.assertTrue(resolve_embedding_normalize(cfg))
        self.assertTrue(spec(cfg).normalize)

    def test_normalization_string_overrides_are_not_silently_truthy(self):
        for value, expected in (("false", False), (" TRUE ", True), ("unknown", True)):
            with self.assertLogs("sglang.srt.configs.embedding_model_spec", "WARNING"):
                actual = resolve_embedding_normalize(SimpleNamespace(normalize=value))
            self.assertEqual(actual, expected)

    def test_existing_encoder_and_generation_attention_unchanged(self):
        for model_cls, module_causal, config_causal, cross, attention_type in (
            (TransformersBase, True, False, False, AttentionType.DECODER),
            (TransformersEmbeddingModel, False, False, True, AttentionType.DECODER),
            (TransformersEmbeddingModel, False, True, True, AttentionType.DECODER),
        ):
            cfg = config(causal=config_causal)
            base = model_cls.__new__(model_cls)
            torch.nn.Module.__init__(base)
            base.config = base.text_config = cfg
            base.model = torch.nn.Sequential(AttentionFlag(module_causal))
            base.start_layer, base.end_layer = 0, 1
            base.quant_config = None
            with patch(
                "sglang.srt.models.transformers.get_parallel",
                return_value=SimpleNamespace(tp_size=1),
            ):
                attention = base._create_attention_instances()[0]
            self.assertEqual(attention.is_cross_attention, cross)
            self.assertEqual(attention.attn_type, attention_type)

    def test_portable_mean_has_no_segment_reduce_dependency(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            hidden = torch.arange(28).reshape(7, 4).to(dtype)
            batch = SimpleNamespace(
                extend_seq_lens=torch.tensor([5, 2], dtype=torch.int32),
                extend_seq_lens_cpu=[5, 2],
            )
            with patch.object(torch, "segment_reduce", side_effect=AssertionError):
                output = pool_hidden_states(PoolingType.MEAN, hidden, batch)
            self.assertEqual(output.dtype, dtype)
            expected = torch.stack(
                [hidden[:5].double().mean(0), hidden[5:].double().mean(0)]
            )
            torch.testing.assert_close(output, expected.to(dtype))

    def test_model_defined_transformers_pooling_preserves_legacy_policy(self):
        def fake_base_init(self, cfg):
            torch.nn.Module.__init__(self)
            self.config = self.text_config = cfg
            self.pp_group = SimpleNamespace(is_last_rank=True)
            self.ignore_unexpected_prefixes = []

        for architecture in (
            "BertModel",
            "MistralModel",
            "Qwen2ForSequenceClassification",
        ):
            for name, expected in (
                ("CLS", PoolingType.CLS),
                ("MEAN", PoolingType.LAST),
            ):
                cfg = SimpleNamespace(
                    architectures=[architecture],
                    pooling_type=name,
                    pooling="avg",
                    normalize=False,
                    is_causal=False,
                )
                with (
                    patch.object(TransformersBase, "__init__", fake_base_init),
                    self.assertLogs("sglang.srt.models.transformers", level="WARNING"),
                ):
                    model = TransformersEmbeddingModel(cfg)
                self.assertEqual(model.pooler.pooling_type, expected)
                self.assertFalse(model.pooler.normalize)

    def test_model_defined_transformers_attention_preserves_legacy_policy(self):
        for architecture in (
            "BertModel",
            "MistralModel",
            "Qwen2ForSequenceClassification",
        ):
            cfg = config(causal=False)
            cfg.architectures = [architecture]
            base = TransformersEmbeddingModel.__new__(TransformersEmbeddingModel)
            torch.nn.Module.__init__(base)
            base.config = base.text_config = cfg
            base.model = torch.nn.Sequential(AttentionFlag(True))
            base.start_layer, base.end_layer = 0, 1
            base.quant_config = None
            with patch(
                "sglang.srt.models.transformers.get_parallel",
                return_value=SimpleNamespace(tp_size=1),
            ):
                attention = base._create_attention_instances()[0]
            self.assertEqual(attention.attn_type, AttentionType.DECODER)
            self.assertFalse(attention.is_cross_attention)

    def test_mean_invalid_lengths_fail_clearly(self):
        for lengths in ([], [0, 7], [-1, 8], [5, 3]):
            batch = SimpleNamespace(
                extend_seq_lens=torch.tensor(lengths), extend_seq_lens_cpu=lengths
            )
            with self.assertRaisesRegex(ValueError, "positive per-request"):
                pool_hidden_states(PoolingType.MEAN, torch.ones(7, 4), batch)

    def test_mean_missing_host_lengths_does_not_copy_accelerator_tensor(self):
        batch = SimpleNamespace(
            extend_seq_lens=SimpleNamespace(device=torch.device("cuda"))
        )
        with self.assertRaisesRegex(ValueError, "extend_seq_lens_cpu"):
            pool_hidden_states(PoolingType.MEAN, torch.ones(7, 4), batch)

    def test_rocm_uses_portable_mean(self):
        from sglang.srt.layers.pooler import _use_cuda_segment_reduce

        tensor_device = SimpleNamespace(device=torch.device("cuda"))
        with patch.object(torch.version, "hip", "test-rocm"):
            self.assertFalse(_use_cuda_segment_reduce(tensor_device))
        with patch.object(torch.version, "hip", None):
            self.assertTrue(_use_cuda_segment_reduce(tensor_device))

    def test_bidirectional_backend_guard_is_not_cuda_only(self):
        result = spec(config())
        for backend in (
            "torch_native",
            "triton",
            "intel_amx",
            "intel_xpu",
            "flashinfer",
        ):
            validate_configured_embedding_attention(result, backend)
        for backend in ("aiter", "wave", "ascend", "unknown"):
            with self.assertRaisesRegex(ValueError, "--attention-backend torch_native"):
                validate_configured_embedding_attention(result, backend)
            # Do not constrain existing native models or causal MEAN support.
            validate_configured_embedding_attention(spec(config(causal=True)), backend)
            validate_configured_embedding_attention(
                spec(
                    SimpleNamespace(architectures=["Qwen2ForCausalLM"]), supports=False
                ),
                backend,
            )
            validate_configured_embedding_attention(None, backend)

    def test_packed_mean_precision_dtype_and_order(self):
        first = torch.ones((512, 4), dtype=torch.bfloat16)
        second = torch.full((8, 4), 1 / 256, dtype=torch.bfloat16)
        for chunks in ((first, second), (second, first)):
            hidden = torch.cat(chunks)
            batch = SimpleNamespace(
                extend_seq_lens=torch.tensor([len(row) for row in chunks]),
                dimensions=None,
            )
            expected = torch.stack([row.float().mean(0) for row in chunks])
            raw = pool_hidden_states(PoolingType.MEAN, hidden, batch)
            self.assertEqual(raw.dtype, hidden.dtype)
            torch.testing.assert_close(raw.float(), expected, rtol=0, atol=0)
            precise = Pooler(
                PoolingType.MEAN, normalize=True, mean_pooling_dtype=torch.float32
            )(hidden, batch).embeddings
            self.assertEqual(precise.dtype, torch.float32)
            torch.testing.assert_close(
                precise, torch.nn.functional.normalize(expected, dim=-1)
            )

    def test_last_cls_and_dimensions(self):
        hidden = torch.arange(28, dtype=torch.float32).reshape(7, 4).to(torch.bfloat16)
        batch = SimpleNamespace(extend_seq_lens=torch.tensor([5, 2]), dimensions=None)
        torch.testing.assert_close(
            pool_hidden_states(PoolingType.LAST, hidden, batch), hidden[[4, 6]]
        )
        torch.testing.assert_close(
            pool_hidden_states(PoolingType.CLS, hidden, batch), hidden[[0, 5]]
        )
        batch.dimensions = [2, 3]
        output = Pooler(PoolingType.MEAN, True, mean_pooling_dtype=torch.float32)(
            hidden, batch
        ).embeddings
        self.assertEqual([row.shape[0] for row in output], [2, 3])
        for row in output:
            torch.testing.assert_close(row.norm(), torch.tensor(1.0))

    def test_packed_mean_small_request_after_large_request(self):
        # FP32 cumulative-endpoint subtraction can still erase a later request.
        first = torch.ones((4096, 4), dtype=torch.bfloat16)
        second = torch.full((8, 4), 1 / 65536, dtype=torch.bfloat16)
        for chunks in ((first, second), (second, first)):
            batch = SimpleNamespace(
                extend_seq_lens=torch.tensor([len(row) for row in chunks]),
                dimensions=None,
            )
            expected = torch.stack([row.float().mean(0) for row in chunks])
            result = pool_hidden_states(
                PoolingType.MEAN,
                torch.cat(chunks),
                batch,
                mean_pooling_dtype=torch.float32,
            )
            torch.testing.assert_close(result, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
