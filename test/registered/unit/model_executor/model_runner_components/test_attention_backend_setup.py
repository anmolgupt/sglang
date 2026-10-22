import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.model_executor.model_runner_components import (
    attention_backend_setup,
)
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    ResolvedAttentionBackendStr,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _FakeBackend:
    def __init__(self, name):
        self.name = name
        # Real backends always carry this (AttentionBackend class attribute).
        self.needs_cpu_seq_lens = True


def test_split_full_attention_applies_model_wrapper_once():
    # The hybrid backend takes the speculative attention mode from the
    # published configuration.
    from sglang.srt.runtime_context import get_context

    override = get_context().override_server_args(speculative_attention_mode="prefill")
    override.install()
    try:
        runner = SimpleNamespace(
            server_args=SimpleNamespace(speculative_attention_mode="prefill"),
            model_config=SimpleNamespace(context_len=2048),
            kv_cache_dtype=None,
            token_to_kv_pool=object(),
            req_to_token_pool=object(),
            kv_index_translator=None,
            init_new_workspace=None,
        )
        wrapper_inputs = []
        wrapped_backend = object()

        def wrap_once(model_runner, backend):
            assert model_runner is runner
            wrapper_inputs.append(backend)
            return wrapped_backend

        constructors = {
            "decode-test": lambda model_runner: _FakeBackend("decode"),
            "prefill-test": lambda model_runner: _FakeBackend("prefill"),
        }
        resolved = ResolvedAttentionBackendStr(
            decode="decode-test", prefill="prefill-test"
        )

        with (
            patch.dict(attention_backend_setup.ATTENTION_BACKENDS, constructors),
            patch.object(
                attention_backend_setup,
                "attn_backend_wrapper",
                side_effect=wrap_once,
            ),
        ):
            result = attention_backend_setup._build_resolved_backend(
                model_runner=runner,
                resolved=resolved,
                init_new_workspace=True,
            )

        assert result is wrapped_backend
        assert len(wrapper_inputs) == 1
        split_backend = wrapper_inputs[0]
        assert isinstance(split_backend, HybridAttnBackend)
        assert split_backend.decode_backend.name == "decode"
        assert split_backend.prefill_backend.name == "prefill"
        assert runner.init_new_workspace is True
    finally:
        override.restore()


def test_equal_resolved_backends_ignore_stale_global_backend():
    runner = SimpleNamespace(
        server_args=SimpleNamespace(
            attention_backend="global-test",
            speculative_attention_mode="prefill",
        ),
        kv_cache_dtype=None,
        token_to_kv_pool=object(),
        req_to_token_pool=object(),
        init_new_workspace=None,
    )
    constructors = {
        "global-test": lambda _runner: _FakeBackend("global"),
        "resolved-test": lambda _runner: _FakeBackend("resolved"),
    }
    resolved = ResolvedAttentionBackendStr(
        decode="resolved-test",
        prefill="resolved-test",
    )

    with (
        patch.dict(attention_backend_setup.ATTENTION_BACKENDS, constructors),
        patch.object(
            attention_backend_setup,
            "attn_backend_wrapper",
            side_effect=lambda _runner, backend: backend,
        ),
    ):
        result = attention_backend_setup._build_resolved_backend(
            model_runner=runner,
            resolved=resolved,
            init_new_workspace=False,
        )

    assert result.name == "resolved"


@pytest.mark.parametrize("backend", ["aiter", "wave", "ascend"])
def test_configured_bidirectional_rejects_unaudited_backend_before_construction(
    backend,
):
    from sglang.srt.configs.embedding_model_spec import CONFIGURED_POOLING_EMBEDDING

    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            embedding_model_spec=SimpleNamespace(
                family=CONFIGURED_POOLING_EMBEDDING, bidirectional_attention=True
            )
        )
    )
    resolved = ResolvedAttentionBackendStr(prefill=backend, decode=backend)
    with patch.dict(
        attention_backend_setup.ATTENTION_BACKENDS,
        {backend: lambda _runner: pytest.fail("Backend must not be constructed")},
    ):
        with pytest.raises(ValueError, match="--attention-backend torch_native"):
            attention_backend_setup._build_resolved_backend(
                model_runner=runner, resolved=resolved, init_new_workspace=False
            )


@pytest.mark.parametrize("backend", ["intel_amx", "intel_xpu"])
def test_configured_bidirectional_accepts_non_cuda_backend_factory(backend):
    from sglang.srt.configs.embedding_model_spec import CONFIGURED_POOLING_EMBEDDING

    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            embedding_model_spec=SimpleNamespace(
                family=CONFIGURED_POOLING_EMBEDDING, bidirectional_attention=True
            )
        ),
        init_new_workspace=None,
    )
    resolved = ResolvedAttentionBackendStr(prefill=backend, decode=backend)
    with (
        patch.dict(
            attention_backend_setup.ATTENTION_BACKENDS,
            {backend: lambda _runner: _FakeBackend(backend)},
        ),
        patch.object(
            attention_backend_setup,
            "attn_backend_wrapper",
            side_effect=lambda _runner, full_backend: full_backend,
        ),
    ):
        result = attention_backend_setup._build_resolved_backend(
            model_runner=runner, resolved=resolved, init_new_workspace=False
        )
    assert result.name == backend


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
