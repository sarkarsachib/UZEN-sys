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


def _gptj_arrays(
    *, hidden: int = 8, heads: int = 2, layers: int = 1, intermediate: int = 16,
    vocab: int = 16, max_seq: int = 8, rotary_dim: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    rng = np.random.default_rng(19)
    arrays = {
        "transformer.wte.weight": rng.standard_normal((vocab, hidden)).astype(np.float32),
        "transformer.ln_f.weight": np.ones(hidden, dtype=np.float32),
        "transformer.ln_f.bias": np.zeros(hidden, dtype=np.float32),
        "lm_head.weight": rng.standard_normal((vocab, hidden)).astype(np.float32),
        "lm_head.bias": np.zeros(vocab, dtype=np.float32),
    }
    for layer in range(layers):
        src = f"transformer.h.{layer}"
        arrays.update(
            {
                f"{src}.ln_1.weight": np.ones(hidden, dtype=np.float32),
                f"{src}.ln_1.bias": np.zeros(hidden, dtype=np.float32),
                f"{src}.attn.q_proj.weight": rng.standard_normal((hidden, hidden)).astype(np.float32),
                f"{src}.attn.k_proj.weight": rng.standard_normal((hidden, hidden)).astype(np.float32),
                f"{src}.attn.v_proj.weight": rng.standard_normal((hidden, hidden)).astype(np.float32),
                f"{src}.attn.out_proj.weight": rng.standard_normal((hidden, hidden)).astype(np.float32),
                f"{src}.mlp.fc_in.weight": rng.standard_normal((intermediate, hidden)).astype(np.float32),
                f"{src}.mlp.fc_in.bias": np.zeros(intermediate, dtype=np.float32),
                f"{src}.mlp.fc_out.weight": rng.standard_normal((hidden, intermediate)).astype(np.float32),
                f"{src}.mlp.fc_out.bias": np.zeros(hidden, dtype=np.float32),
                f"{src}.attn.bias": np.zeros((1, 1, max_seq, max_seq), dtype=np.float32),
                f"{src}.attn.masked_bias": np.asarray(0.0, dtype=np.float32),
            }
        )
    config = {
        "model_type": "gptj",
        "architectures": ["GPTJForCausalLM"],
        "n_embd": hidden,
        "n_head": heads,
        "n_layer": layers,
        "n_inner": intermediate,
        "vocab_size": vocab,
        "n_positions": max_seq,
        "rotary_dim": rotary_dim,
        "activation_function": "gelu_new",
        "hidden_act": "gelu",
        "layer_norm_epsilon": 1e-5,
        "tie_word_embeddings": False,
        "scale_attn_weights": True,
    }
    return arrays, config


def test_gemma_zero_centered_norms_and_embedding_scale_are_mapped():
    arrays, config = _llama_like_arrays()
    arrays.pop("lm_head.weight")
    for name in ["model.norm.weight"] + [
        f"model.layers.{layer}.{norm}.weight"
        for layer in range(2)
        for norm in ("input_layernorm", "post_attention_layernorm")
    ]:
        arrays[name] = np.zeros_like(arrays[name])
    config["model_type"] = "gemma"
    config["architectures"] = ["GemmaForCausalLM"]
    config["hidden_act"] = "gelu_pytorch_tanh"
    config.pop("tie_word_embeddings")
    result = reconstruct_hf_model(arrays, config)
    assert result.report.status == "validated", result.report.reasons
    assert result.report.spec["profile"] == "gemma"
    assert result.report.spec["embed_scale"] == np.sqrt(8)
    assert result.report.spec["zero_centered_norm"] is True
    assert result.report.spec["tie_embeddings"] is True
    assert result.package is not None
    assert np.array_equal(result.package.get("norm.weight"), np.ones(8, dtype=np.float32))
    assert any(
        node["op"] == "scale" and node["params"].get("factor") == np.sqrt(8)
        for node in result.report.graph["nodes"]
    )
    assert any(
        node["op"] == "gelu"
        for node in result.report.graph["nodes"]
    )


def test_gptj_parallel_residual_and_interleaved_partial_rope_are_mapped():
    arrays, config = _gptj_arrays()
    result = reconstruct_hf_model(arrays, config)
    assert result.report.status == "validated", result.report.reasons
    assert result.report.spec["profile"] == "gptj"
    assert result.report.spec["activation"] == "gelu"
    assert result.report.spec["rotary_dim"] == 4
    assert result.report.extra == []
    nodes = result.report.graph["nodes"]
    assert any(
        node["op"] == "rope"
        and node["params"].get("interleaved") is True
        and node["params"].get("rotary_dim") == 4
        for node in nodes
    )
    res2 = next(node for node in nodes if node["id"] == "l0_res2")
    assert res2["inputs"]["a"] == "l0_res1"
    assert res2["inputs"]["b"] == "l0_fc_out"
    assert not any(node["id"] == "l0_ln2" for node in nodes)


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


def test_mistral_sliding_window_is_mapped_into_attention():
    profile = hf_profile_for("mistral")
    assert profile is not None and profile.reconstruct
    arrays, config = _llama_like_arrays()
    config["model_type"] = "mistral"
    config["architectures"] = ["MistralForCausalLM"]
    config["sliding_window"] = 2
    result = reconstruct_hf_model(arrays, config)
    assert result.report.status == "validated", result.report.reasons
    assert result.report.spec["sliding_window"] == 2
    assert any(
        node["op"] == "attention" and node.get("params", {}).get("window") == 2
        for node in result.report.graph["nodes"]
    )
