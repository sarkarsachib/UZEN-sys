"""Offline coverage for the declarative Hugging Face decoder profiles."""

from __future__ import annotations

import numpy as np

from uzen.core.wie.hf_profiles import hf_profile_for, profile_catalogue
from uzen.core.wie.reconstruction import reconstruct_hf_model


def _llama_like_arrays(
    *,
    hidden: int = 8,
    heads: int = 2,
    kv_heads: int = 2,
    layers: int = 2,
    intermediate: int = 16,
    vocab: int = 16,
    max_seq: int = 8,
    qkv_bias: bool = False,
    o_bias: bool = False,
    mlp_bias: bool = False,
    norm_affine: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    rng = np.random.default_rng(7)
    head_dim = hidden // heads
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    arrays = {
        "model.embed_tokens.weight": rng.standard_normal((vocab, hidden)).astype(np.float32),
        "lm_head.weight": rng.standard_normal((vocab, hidden)).astype(np.float32),
    }
    if norm_affine:
        arrays["model.norm.weight"] = np.ones(hidden, dtype=np.float32)
    for layer in range(layers):
        src = f"model.layers.{layer}"
        if norm_affine:
            arrays[f"{src}.input_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
            arrays[f"{src}.post_attention_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
        arrays[f"{src}.self_attn.q_proj.weight"] = rng.standard_normal((q_dim, hidden)).astype(np.float32)
        arrays[f"{src}.self_attn.k_proj.weight"] = rng.standard_normal((kv_dim, hidden)).astype(np.float32)
        arrays[f"{src}.self_attn.v_proj.weight"] = rng.standard_normal((kv_dim, hidden)).astype(np.float32)
        arrays[f"{src}.self_attn.o_proj.weight"] = rng.standard_normal((hidden, q_dim)).astype(np.float32)
        arrays[f"{src}.mlp.gate_proj.weight"] = rng.standard_normal((intermediate, hidden)).astype(np.float32)
        arrays[f"{src}.mlp.up_proj.weight"] = rng.standard_normal((intermediate, hidden)).astype(np.float32)
        arrays[f"{src}.mlp.down_proj.weight"] = rng.standard_normal((hidden, intermediate)).astype(np.float32)
        if qkv_bias:
            arrays[f"{src}.self_attn.q_proj.bias"] = np.zeros(q_dim, dtype=np.float32)
            arrays[f"{src}.self_attn.k_proj.bias"] = np.zeros(kv_dim, dtype=np.float32)
            arrays[f"{src}.self_attn.v_proj.bias"] = np.zeros(kv_dim, dtype=np.float32)
        if o_bias:
            arrays[f"{src}.self_attn.o_proj.bias"] = np.zeros(hidden, dtype=np.float32)
        if mlp_bias:
            arrays[f"{src}.mlp.gate_proj.bias"] = np.zeros(intermediate, dtype=np.float32)
            arrays[f"{src}.mlp.up_proj.bias"] = np.zeros(intermediate, dtype=np.float32)
            arrays[f"{src}.mlp.down_proj.bias"] = np.zeros(hidden, dtype=np.float32)
    config = {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "num_hidden_layers": layers,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
        "max_position_embeddings": max_seq,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
    }
    return arrays, config


def test_profile_catalogue_has_twelve_plus_families_and_explicit_statuses():
    profiles = profile_catalogue()
    assert len(profiles) >= 12
    assert len({entry["model_type"] for entry in profiles}) == len(profiles)
    assert sum(bool(entry["reconstruct"]) for entry in profiles) >= 8
    assert all(entry["reconstruct"] or entry["reason"] for entry in profiles)


def test_qwen2_biases_are_mapped_and_executed():
    arrays, config = _llama_like_arrays(qkv_bias=True)
    result = reconstruct_hf_model(arrays, config)
    assert result.report.status == "validated", result.report.reasons
    assert result.report.spec["profile"] == "qwen2"
    assert any(item["target"] == "layers.0.q_proj.bias" for item in result.report.tensor_mapping)
    assert result.report.execution is not None
    assert result.report.execution["status"] == "passed"


def test_minicpm_partial_rotary_dim_is_honoured():
    arrays, config = _llama_like_arrays()
    config["model_type"] = "minicpm"
    config["architectures"] = ["MiniCPMForCausalLM"]
    config["partial_rotary_factor"] = 0.5
    result = reconstruct_hf_model(arrays, config)
    assert result.report.status == "validated", result.report.reasons
    assert result.report.spec["rotary_dim"] == 2
    assert any(node["op"] == "rope" and node["params"].get("rotary_dim") == 2 for node in result.report.graph["nodes"])


def test_olmo_non_affine_norms_have_no_norm_weights():
    arrays, config = _llama_like_arrays(norm_affine=False)
    config["model_type"] = "olmo"
    config["architectures"] = ["OlmoForCausalLM"]
    result = reconstruct_hf_model(arrays, config)
    assert result.report.status == "validated", result.report.reasons
    assert "norm.weight" not in result.package.tensors
    assert "layers.0.input_norm.weight" not in result.package.tensors


def test_stablelm_is_not_mapped_to_rmsnorm_swiglu():
    arrays, config = _llama_like_arrays()
    config["model_type"] = "stablelm"
    config["architectures"] = ["StableLmForCausalLM"]
    result = reconstruct_hf_model(arrays, config)
    assert result.report.status == "unsupported"
    assert "LayerNorm" in result.report.reasons[0]


def test_sliding_window_mistral_is_reported_unsupported_not_unwindowed():
    profile = hf_profile_for("mistral")
    assert profile is not None and not profile.reconstruct
    arrays, config = _llama_like_arrays()
    config["model_type"] = "mistral"
    config["architectures"] = ["MistralForCausalLM"]
    result = reconstruct_hf_model(arrays, config)
    assert result.report.status == "unsupported"
    assert "sliding-window" in result.report.reasons[0]
