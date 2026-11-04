import builtins
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers import Gemma3TextConfig, Ministral3Config, Qwen2Config

from sglang.srt.layers.pooler import PoolingType
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.models.ministral3 import Ministral3ForCausalLM
from sglang.srt.models.ministral3_embedding import EmbeddingNemotron3Model
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class Backbone(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pp_group = SimpleNamespace(is_last_rank=True)
        self.layers = [
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    attn=SimpleNamespace(attn_type=AttentionType.DECODER)
                )
            )
        ]


class TestMinistral3Embedding(unittest.TestCase):
    def test_model_config_default_paths_do_not_import_registry(self):
        from sglang.srt.configs.model_config import ModelConfig

        original_import = builtins.__import__

        def reject_registry(name, *args, **kwargs):
            if name == "sglang.srt.models.registry":
                raise AssertionError("default embedding config imported model registry")
            return original_import(name, *args, **kwargs)

        configs = (
            Qwen2Config(architectures=["Qwen2ForCausalLM"]),
            Qwen2Config(architectures=["Qwen3ForCausalLM"]),
            Qwen2Config(architectures=["BertModel"], pooling="avg", is_causal=False),
            Gemma3TextConfig(
                architectures=["Gemma3TextModel"],
                use_bidirectional_attention=True,
                is_causal=False,
            ),
            Ministral3Config(
                architectures=["Ministral3Model"], pooling="avg", is_causal=False
            ),
        )
        for cfg in configs:
            with tempfile.TemporaryDirectory() as model_dir:
                cfg.save_pretrained(model_dir)
                with (
                    self.subTest(architectures=cfg.architectures),
                    patch("builtins.__import__", side_effect=reject_registry),
                    patch(
                        "sglang.srt.configs.model_config.supports_checkpoint_embedding_config",
                        side_effect=AssertionError("unnecessary capability lookup"),
                    ),
                ):
                    result = ModelConfig(
                        model_dir, is_embedding=True, trust_remote_code=False
                    )
                self.assertIsNotNone(result.embedding_model_spec)

    def test_declared_metadata_still_resolves_backend_capability(self):
        from sglang.srt.configs.model_config import ModelConfig, ModelImpl

        for metadata in ({"normalize": False}, {"pooling": "mean"}):
            cfg = Qwen2Config(architectures=["Qwen2ForCausalLM"], **metadata)
            with tempfile.TemporaryDirectory() as model_dir:
                cfg.save_pretrained(model_dir)
                with patch(
                    "sglang.srt.configs.model_config.supports_checkpoint_embedding_config",
                    return_value=True,
                ) as capability:
                    result = ModelConfig(
                        model_dir, is_embedding=True, trust_remote_code=False
                    )
                capability.assert_called_once_with(cfg.architectures, ModelImpl.AUTO)
                self.assertEqual(
                    result.embedding_model_spec.family, "configured_pooling_embedding"
                )

    def make_model(self, causal=False, pooling="avg", sliding_window=None):
        with patch(
            "sglang.srt.models.ministral3_embedding.Ministral3Backbone", Backbone
        ):
            return EmbeddingNemotron3Model(
                SimpleNamespace(
                    is_causal=causal, pooling=pooling, sliding_window=sliding_window
                )
            )

    def test_windowed_checkpoint_rejected_before_backbone_construction(self):
        for causal in (False, True):
            with self.assertRaisesRegex(ValueError, "do not.*remove"):
                self.make_model(causal=causal, sliding_window=128)

    def test_bidirectional_mean_no_head(self):
        model = self.make_model()
        self.assertEqual(
            model.model.layers[0].self_attn.attn.attn_type, AttentionType.ENCODER_ONLY
        )
        self.assertEqual(model.pooler.pooling_type, PoolingType.MEAN)
        self.assertEqual(model.pooler.mean_pooling_dtype, torch.float32)
        self.assertFalse(hasattr(model, "lm_head"))

    def test_causal_last_preserved(self):
        model = self.make_model(causal=True, pooling="LAST")
        self.assertEqual(
            model.model.layers[0].self_attn.attn.attn_type, AttentionType.DECODER
        )
        self.assertEqual(model.pooler.pooling_type, PoolingType.LAST)

    def test_weight_mapping_delegates_to_native_loader(self):
        model = self.make_model()
        weights = [
            (name, torch.ones(1))
            for name in (
                "layers.0.self_attn.q_proj.weight",
                "embed_tokens.weight",
                "norm.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "lm_head.weight",
            )
        ]
        seen = []

        def capture(self, values):
            seen.extend(list(values))

        with patch.object(Ministral3ForCausalLM, "load_weights", capture):
            model.load_weights(iter(weights))
        self.assertEqual(
            [x[0] for x in seen],
            [
                "model.layers.0.self_attn.q_proj.weight",
                "model.embed_tokens.weight",
                "model.norm.weight",
                "model.layers.0.self_attn.k_proj.weight",
            ],
        )

    def test_reject_generation(self):
        with self.assertRaisesRegex(ValueError, "not generation"):
            self.make_model().forward(None, None, None, get_embedding=False)

    def test_bare_architecture_alone_preserves_generation_dispatch(self):
        from sglang.srt.configs.model_config import is_generation_model

        self.assertTrue(is_generation_model(["Ministral3Model"]))
        self.assertFalse(is_generation_model(["Ministral3Model"], is_embedding=True))
        self.assertTrue(is_generation_model(["Ministral3ForCausalLM"]))

    def test_nemotron_detection_requires_backbone_and_embedding_metadata(self):
        from sglang.srt.configs.model_config import is_embedding_nemotron3

        for fields, expected in (
            ({}, False),
            ({"is_causal": True}, False),
            ({"normalize": True}, False),
            ({"normalize": False}, False),
            ({"pooling": None, "pooling_type": None}, False),
            ({"is_causal": False}, True),
            ({"pooling": "avg"}, True),
            ({"pooling_type": "LAST"}, True),
        ):
            for nested in (False, True):
                cfg = SimpleNamespace(architectures=["Ministral3Model"])
                owner = SimpleNamespace(**fields)
                if nested:
                    cfg.text_config = owner
                else:
                    vars(cfg).update(fields)
                with self.subTest(fields=fields, nested=nested):
                    self.assertEqual(is_embedding_nemotron3(cfg), expected)
        for arch in ("Ministral3ForCausalLM", "Mistral3ForConditionalGeneration"):
            cfg = SimpleNamespace(
                architectures=[arch],
                is_causal=False,
                pooling="avg",
                text_config=SimpleNamespace(architectures=["Ministral3Model"]),
            )
            self.assertFalse(is_embedding_nemotron3(cfg))

    def test_model_config_and_loader_use_metadata_not_backbone_name(self):
        from sglang.srt.configs.model_config import ModelConfig, ModelImpl
        from sglang.srt.model_loader.utils import get_model_architecture
        from sglang.srt.models.registry import ModelRegistry

        self.assertFalse(ModelRegistry.is_arch_supported("Ministral3Model"))
        self.assertTrue(ModelRegistry.is_arch_supported("EmbeddingNemotron3Model"))
        for metadata, native in (
            ({}, False),
            ({"is_causal": True}, False),
            ({"normalize": False}, False),
            ({"is_causal": False, "pooling": "avg"}, True),
            ({"is_causal": True, "pooling": "LAST"}, True),
        ):
            for requested in (False, True):
                cfg = Ministral3Config(architectures=["Ministral3Model"], **metadata)
                with tempfile.TemporaryDirectory() as model_dir:
                    cfg.save_pretrained(model_dir)
                    with self.subTest(metadata=metadata, requested=requested):
                        result = ModelConfig(
                            model_dir, is_embedding=requested, trust_remote_code=False
                        )
                        self.assertEqual(result.is_embedding_nemotron3, native)
                        self.assertEqual(
                            result.is_generation, not (native or requested)
                        )
                        self.assertEqual(
                            result.embedding_model_spec.auto_enable_embedding, native
                        )
                        model_cls, arch = get_model_architecture(result)
                        if native:
                            self.assertIs(model_cls, EmbeddingNemotron3Model)
                            self.assertEqual(arch, "EmbeddingNemotron3Model")
                        else:
                            self.assertEqual(
                                arch,
                                "TransformersEmbeddingModel"
                                if requested
                                else "TransformersForCausalLM",
                            )
                        self.assertEqual(
                            result.hf_config.architectures, ["Ministral3Model"]
                        )
                        # Explicit Transformers uses the original HF architecture.
                        result.model_impl = ModelImpl.TRANSFORMERS
                        _, arch = get_model_architecture(result)
                        self.assertEqual(
                            arch,
                            "TransformersEmbeddingModel"
                            if native or requested
                            else "TransformersForCausalLM",
                        )
                        self.assertEqual(
                            result.hf_config.architectures, ["Ministral3Model"]
                        )

    def test_generation_helpers_fail_clearly(self):
        model = self.make_model()
        for fn, args in (
            (model.forward_split_prefill, ()),
            (model.get_embed_and_head, ()),
            (model.set_embed_and_head, (None, None)),
            (model.set_eagle3_layers_to_capture, ()),
            (model.set_dflash_layers_to_capture, ([],)),
        ):
            with self.assertRaisesRegex(ValueError, "EmbeddingNemotron3Model"):
                fn(*args)

    def test_native_weight_mapping_has_one_source(self):
        model = self.make_model()
        self.assertEqual(
            model.stacked_params_mapping,
            Ministral3ForCausalLM._get_stacked_params_mapping(),
        )

    def test_legacy_loader_uses_shared_mapping(self):
        model = self.make_model()
        model.config.tie_word_embeddings = False
        param = nn.Parameter(torch.zeros(1))
        loaded = []
        param.weight_loader = lambda p, weight, shard: loaded.append(shard)
        with (
            patch.object(
                model,
                "_get_stacked_params_mapping",
                wraps=model._get_stacked_params_mapping,
            ) as mapping,
            patch.object(
                model,
                "named_parameters",
                return_value=[("model.layers.0.self_attn.qkv_proj.weight", param)],
            ),
        ):
            model._legacy_load_weights(
                (f"model.layers.0.self_attn.{name}_proj.weight", torch.ones(1))
                for name in ("q", "k", "v")
            )
        mapping.assert_called_once_with()
        self.assertEqual(loaded, ["q", "k", "v"])

    def test_backend_choice_preserves_existing_native_contract(self):
        from sglang.srt.configs.model_config import (
            ModelImpl,
            supports_checkpoint_embedding_config,
        )

        with patch("sglang.srt.models.registry.ModelRegistry") as registry:
            registry.is_arch_supported.return_value = True
            registry.resolve_model_cls.return_value = (nn.Module, "Qwen2ForCausalLM")
            self.assertFalse(
                supports_checkpoint_embedding_config(
                    ["Qwen2ForCausalLM"], ModelImpl.AUTO
                )
            )
            registry.resolve_model_cls.return_value = (
                EmbeddingNemotron3Model,
                "EmbeddingNemotron3Model",
            )
            self.assertTrue(
                supports_checkpoint_embedding_config(
                    ["EmbeddingNemotron3Model"], ModelImpl.AUTO
                )
            )
            registry.is_arch_supported.return_value = False
            self.assertTrue(
                supports_checkpoint_embedding_config(["CustomModel"], ModelImpl.AUTO)
            )
            self.assertTrue(
                supports_checkpoint_embedding_config(
                    ["Qwen2ForCausalLM"], ModelImpl.TRANSFORMERS
                )
            )
            self.assertFalse(
                supports_checkpoint_embedding_config(
                    ["Qwen2ForCausalLM"], ModelImpl.MINDSPORE
                )
            )


if __name__ == "__main__":
    unittest.main()
