"""Declarative Hugging Face decoder-family profiles.

A profile is deliberately data, not a guessed adapter.  ``reconstruct`` is true
only for families whose config fields and tensor names are handled by the shared
Llama-like reconstruction path.  Families that need a different attention or
positional layout stay explicit import profiles and report ``unsupported`` with
their concrete reason instead of being mapped to the nearest architecture.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HFProfile:
    model_type: str
    architectures: tuple[str, ...]
    reconstruct: bool = False
    qkv_bias: bool = False
    o_bias: bool = False
    mlp_bias: bool = False
    norm_affine: bool = True
    layer_norm: bool = False
    norm_eps: float | None = None
    partial_rotary: bool = False
    clip_qkv: bool = False
    reason: str = ""


_PROFILES = (
    # Llama-like layouts.  These keep the Llama tensor names and differ only in
    # optional projection biases, norm affine parameters or partial RoPE.
    HFProfile("qwen2", ("Qwen2ForCausalLM",), True, qkv_bias=True),
    HFProfile("yi", ("YiForCausalLM",), True),
    HFProfile(
        "stablelm",
        ("StableLmForCausalLM",),
        reason="StableLM uses biased LayerNorm plus a non-SiLU MLP; no RMSNorm/SwiGLU graph is substituted",
    ),
    HFProfile("aquila", ("AquilaForCausalLM",), True, qkv_bias=True),
    HFProfile("baichuan", ("BaichuanForCausalLM",), True, qkv_bias=True),
    HFProfile("internlm2", ("InternLM2ForCausalLM",), True, qkv_bias=True),
    HFProfile(
        "olmo",
        ("OlmoForCausalLM",),
        True,
        norm_affine=False,
        layer_norm=True,
        norm_eps=1e-5,
        clip_qkv=True,
    ),
    HFProfile("minicpm", ("MiniCPMForCausalLM",), True, partial_rotary=True),
    HFProfile("orion", ("OrionForCausalLM",), True),
    # Explicit import profiles: recognized, but no graph is guessed.
    HFProfile(
        "mistral",
        ("MistralForCausalLM",),
        reason="sliding-window attention is not implemented; no unwindowed graph is substituted",
    ),
    HFProfile(
        "gemma",
        ("GemmaForCausalLM",),
        reason="Gemma's embedding scale and GeGLU layout need a dedicated graph",
    ),
    HFProfile(
        "gptj",
        ("GPTJForCausalLM",),
        reason="GPT-J uses separate Q/K/V projections and a parallel residual",
    ),
    HFProfile(
        "gpt_neo",
        ("GPTNeoForCausalLM",),
        reason="GPT-Neo combines learned positions with local attention",
    ),
    HFProfile(
        "falcon",
        ("FalconForCausalLM",),
        reason="Falcon uses multiquery attention with ALiBi",
    ),
    HFProfile(
        "bloom",
        ("BloomForCausalLM",),
        reason="Bloom uses ALiBi and a fused QKV layout",
    ),
    HFProfile(
        "mpt",
        ("MptForCausalLM",),
        reason="MPT uses ALiBi without learned or rotary positions",
    ),
    HFProfile(
        "phi",
        ("PhiForCausalLM", "Phi3ForCausalLM"),
        reason="Phi's fused projections and partial-RoPE layout need a dedicated graph",
    ),
)

HF_PROFILES: dict[str, HFProfile] = {
    name: profile
    for profile in _PROFILES
    for name in (profile.model_type, *(architecture.lower() for architecture in profile.architectures))
}


def hf_profile_for(model_type: str, architectures: list[str] | None = None) -> HFProfile | None:
    profile = HF_PROFILES.get(str(model_type or "").lower())
    if profile is not None:
        return profile
    for architecture in architectures or []:
        profile = HF_PROFILES.get(str(architecture).lower())
        if profile is not None:
            return profile
    return None


def profile_catalogue() -> list[dict[str, object]]:
    return [
        {
            "model_type": profile.model_type,
            "architectures": list(profile.architectures),
            "reconstruct": profile.reconstruct,
            "reason": profile.reason,
        }
        for profile in _PROFILES
    ]
