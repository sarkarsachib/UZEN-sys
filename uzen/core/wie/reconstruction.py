"""Reconstruct an NIR graph from an imported Hugging Face checkpoint.

This module only reconstructs architectures for which the mapping is explicit and
the resulting graph can be validated against the imported tensors. It does not
guess missing layers. Unsupported architectures and unsupported fields are
returned as explicit ``unsupported`` reports; partial or inconsistent checkpoints
never become a validated artifact.

GPT-2 uses learned position embeddings, pre-LayerNorm blocks, fused QKV, causal
attention, GELU, and a tied LM head. Its Hugging Face Conv1D weights are
transposed into the NIR linear convention, and those transforms are recorded per
tensor. Llama uses separate Q/K/V/O projections, RMSNorm, SwiGLU, real RoPE, and
GQA-aware K/V repetition. OPT uses a learned absolute position table with its
+2 offset, pre-LayerNorm blocks with biased LayerNorm and biased Q/K/V/O and MLP
projections, and a ReLU feed-forward block. GPT-NeoX uses partial rotary
embeddings over a fused, per-head-interleaved QKV projection and a parallel
attention/MLP residual. Those mappings and graph nodes are explicit for every
adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from uzen.core.nir.executor import execute
from uzen.core.nir.graph import Graph
from uzen.core.nir.types import Ref, TensorType
from uzen.core.nws.dtypes import dtype_name
from uzen.core.nws.families import ArchitectureDefinition, build_architecture
from uzen.core.nws.tensors import WeightPackage
from uzen.core.wie.hf_profiles import HFProfile, hf_profile_for
from uzen.errors import UzenError


@dataclass
class ReconstructionReport:
    status: str
    architecture: str | None
    family: str | None
    spec: dict[str, Any]
    graph: dict[str, Any] | None
    graph_summary: dict[str, Any] | None
    tensor_mapping: list[dict[str, Any]]
    missing: list[str]
    extra: list[str]
    shape_errors: list[str]
    unresolved: list[str]
    warnings: list[str]
    validation: dict[str, Any] | None
    execution: dict[str, Any] | None
    source_tensor_count: int
    mapped_tensor_count: int
    progress: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def graph_reconstructed(self) -> bool:
        return self.graph is not None and self.validation is not None and bool(self.validation.get("ok"))

    @property
    def execution_validated(self) -> bool:
        return self.execution is not None and self.execution.get("status") == "passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "architecture": self.architecture,
            "family": self.family,
            "spec": self.spec,
            "graph": self.graph,
            "graph_summary": self.graph_summary,
            "tensor_mapping": self.tensor_mapping,
            "missing": self.missing,
            "extra": self.extra,
            "shape_errors": self.shape_errors,
            "unresolved": self.unresolved,
            "warnings": self.warnings,
            "validation": self.validation,
            "execution": self.execution,
            "source_tensor_count": self.source_tensor_count,
            "mapped_tensor_count": self.mapped_tensor_count,
            "progress": self.progress,
            "reasons": self.reasons,
            "graph_reconstructed": self.graph_reconstructed,
            "execution_validated": self.execution_validated,
        }


@dataclass
class ReconstructionResult:
    report: ReconstructionReport
    package: WeightPackage | None = None
    definition: ArchitectureDefinition | None = field(default=None, repr=False)

    @property
    def usable(self) -> bool:
        return self.report.status in ("validated", "partial") and self.package is not None


def reconstruct_hf_model(
    arrays: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    source_dtypes: Mapping[str, str] | None = None,
    name: str | None = None,
    source_hash: str | None = None,
    on_progress: Any | None = None,
) -> ReconstructionResult:
    """Build and validate a graph, or return a precise unsupported/failed report."""
    progress: list[dict[str, Any]] = []
    unresolved: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []
    extra: list[str] = []
    shape_errors: list[str] = []
    reasons: list[str] = []

    def step(stage: str, status: str, message: str) -> None:
        progress.append({"stage": stage, "status": status, "message": message})
        if on_progress is not None:
            on_progress(stage, status, message)

    model_type = str(config.get("model_type") or "").lower()
    architectures = [str(a) for a in (config.get("architectures") or [])]
    architecture = architectures[0] if architectures else (model_type or None)
    profile = hf_profile_for(model_type, architectures)
    if profile is not None and not profile.reconstruct:
        reason = (
            f"{profile.model_type} reconstruction is not implemented; "
            f"the model is recognized as an import profile: {profile.reason}"
        )
        step("architecture", "unsupported", reason)
        return ReconstructionResult(
            report=ReconstructionReport(
                status="unsupported",
                architecture=architecture,
                family="transformer_decoder",
                spec={},
                graph=None,
                graph_summary=None,
                tensor_mapping=[],
                missing=[],
                extra=sorted(arrays),
                shape_errors=[],
                unresolved=[reason],
                warnings=[],
                validation=None,
                execution=None,
                source_tensor_count=len(arrays),
                mapped_tensor_count=0,
                progress=progress,
                reasons=[reason],
            )
        )
    if model_type == "llama" or profile is not None:
        return _reconstruct_llama(
            arrays,
            config,
            architecture=architecture,
            source_dtypes=source_dtypes,
            name=name,
            source_hash=source_hash,
            on_progress=on_progress,
            profile=profile,
        )
    if model_type == "opt":
        return _reconstruct_opt(
            arrays,
            config,
            architecture=architecture,
            source_dtypes=source_dtypes,
            name=name,
            source_hash=source_hash,
            on_progress=on_progress,
        )
    if model_type == "gpt_neox":
        return _reconstruct_neox(
            arrays,
            config,
            architecture=architecture,
            source_dtypes=source_dtypes,
            name=name,
            source_hash=source_hash,
            on_progress=on_progress,
        )
    if model_type not in ("gpt2", "gpt2lmhead"):
        reason = (
            f"architecture adapter for model_type '{model_type or 'unknown'}' is not implemented; "
            "no graph was guessed"
        )
        step("architecture", "unsupported", reason)
        return ReconstructionResult(
            report=ReconstructionReport(
                status="unsupported",
                architecture=architecture,
                family="transformer_decoder" if model_type else None,
                spec={},
                graph=None,
                graph_summary=None,
                tensor_mapping=[],
                missing=[],
                extra=sorted(arrays),
                shape_errors=[],
                unresolved=[reason],
                warnings=[],
                validation=None,
                execution=None,
                source_tensor_count=len(arrays),
                mapped_tensor_count=0,
                progress=progress,
                reasons=[reason],
            )
        )

    try:
        spec = _gpt2_spec(config, warnings)
    except ValueError as exc:
        step("config", "unsupported", str(exc))
        return ReconstructionResult(
            report=ReconstructionReport(
                status="unsupported",
                architecture=architecture,
                family="transformer_decoder",
                spec={},
                graph=None,
                graph_summary=None,
                tensor_mapping=[],
                missing=[],
                extra=sorted(arrays),
                shape_errors=[],
                unresolved=[str(exc)],
                warnings=warnings,
                validation=None,
                execution=None,
                source_tensor_count=len(arrays),
                mapped_tensor_count=0,
                progress=progress,
                reasons=[str(exc)],
            )
        )
    step("config", "complete", "GPT-2 configuration validated against the supported adapter")

    try:
        definition = build_architecture("transformer_decoder", spec)
    except UzenError as exc:
        step("architecture", "failed", str(exc))
        return ReconstructionResult(
            report=ReconstructionReport(
                status="failed",
                architecture=architecture,
                family="transformer_decoder",
                spec=spec,
                graph=None,
                graph_summary=None,
                tensor_mapping=[],
                missing=[],
                extra=sorted(arrays),
                shape_errors=[],
                unresolved=[str(exc)],
                warnings=warnings,
                validation=None,
                execution=None,
                source_tensor_count=len(arrays),
                mapped_tensor_count=0,
                progress=progress,
                reasons=[f"the NIR transformer builder rejected the configuration: {exc}"],
            )
        )
    step("architecture", "complete", "NIR transformer graph built from explicit GPT-2 fields")

    canonical, mapping, missing, extra, shape_errors, unresolved = _gpt2_tensors(
        arrays, spec, definition, unresolved, warnings
    )
    step(
        "tensor_mapping",
        "complete" if not missing and not shape_errors else "failed",
        f"mapped {len(mapping)} tensors; {len(missing)} missing, {len(shape_errors)} shape errors, "
        f"{len(extra)} unmapped source tensors",
    )

    if missing or shape_errors:
        reasons.extend(
            [
                "all NIR parameters must be mapped from source tensors before a graph can be validated",
                "missing tensors are reported, never filled with guessed values",
            ]
        )
        return ReconstructionResult(
            report=ReconstructionReport(
                status="unsupported",
                architecture=architecture,
                family="transformer_decoder",
                spec=spec,
                graph=None,
                graph_summary=None,
                tensor_mapping=mapping,
                missing=missing,
                extra=extra,
                shape_errors=shape_errors,
                unresolved=unresolved,
                warnings=warnings,
                validation=None,
                execution=None,
                source_tensor_count=len(arrays),
                mapped_tensor_count=len(canonical),
                progress=progress,
                reasons=reasons,
            )
        )

    package = WeightPackage.from_arrays(
        canonical,
        name=name or "imported_model",
        family="transformer_decoder",
        family_spec=spec,
        metadata={
            "reconstruction": {
                "architecture": architecture,
                "source_hash": source_hash,
                "tensor_mapping": mapping,
                "graph": definition.graph.to_dict(),
                "adapter": "gpt2",
            },
            "source_hash": source_hash,
        },
        source_dtypes={
            target: source_dtypes[item["source"]] if source_dtypes and item["source"] in source_dtypes else item["dtype"]
            for target, item in ((m["target"], m) for m in mapping)
        },
    )
    validation = definition.graph.validate(package)
    step(
        "graph_validation",
        "complete" if validation.ok else "failed",
        f"graph validation {'passed' if validation.ok else 'failed'} with {validation.num_nodes} nodes",
    )
    validation_doc = {
        "ok": validation.ok,
        "errors": list(validation.errors),
        "warnings": list(validation.warnings),
        "num_nodes": validation.num_nodes,
        "num_params": validation.num_params,
        "param_names": list(validation.param_names),
        "order": list(validation.order),
    }
    if not validation.ok:
        reasons.append("graph validation failed; no validated artifact should be published")
        return ReconstructionResult(
            report=ReconstructionReport(
                status="failed",
                architecture=architecture,
                family="transformer_decoder",
                spec=spec,
                graph=definition.graph.to_dict(),
                graph_summary=definition.graph.summary(),
                tensor_mapping=mapping,
                missing=missing,
                extra=extra,
                shape_errors=shape_errors,
                unresolved=unresolved,
                warnings=warnings,
                validation=validation_doc,
                execution=None,
                source_tensor_count=len(arrays),
                mapped_tensor_count=len(canonical),
                progress=progress,
                reasons=reasons,
            )
        )

    execution = _execute(definition, package, spec)
    step(
        "forward_execution",
        execution.get("status", "failed"),
        execution.get("message", "forward pass completed"),
    )
    if extra:
        unresolved.append(
            "source tensors not represented by the NIR graph: " + ", ".join(extra[:12])
        )
    status = "validated" if execution.get("status") == "passed" and not extra and not unresolved else "partial"
    if status == "partial":
        reasons.append(
            "the graph executes, but some source tensors or configuration details remain unresolved"
        )
    else:
        reasons.append("all source tensors mapped, graph validated, and a forward pass produced finite logits")
    if execution.get("status") != "passed":
        status = "failed"
        reasons.append("forward execution failed; the graph is not marked validated")
    report = ReconstructionReport(
        status=status,
        architecture=architecture,
        family="transformer_decoder",
        spec=spec,
        graph=definition.graph.to_dict(),
        graph_summary=definition.graph.summary(),
        tensor_mapping=mapping,
        missing=missing,
        extra=extra,
        shape_errors=shape_errors,
        unresolved=unresolved,
        warnings=warnings,
        validation=validation_doc,
        execution=execution,
        source_tensor_count=len(arrays),
        mapped_tensor_count=len(canonical),
        progress=progress,
        reasons=reasons,
    )
    if status == "failed":
        return ReconstructionResult(report=report, package=None, definition=definition)
    return ReconstructionResult(report=report, package=package, definition=definition)


def _gpt2_spec(config: Mapping[str, Any], warnings: list[str]) -> dict[str, Any]:
    try:
        d_model = int(config["n_embd"])
        n_heads = int(config["n_head"])
        n_layers = int(config["n_layer"])
        vocab_size = int(config["vocab_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"GPT-2 configuration is missing a required integer field: {exc}") from exc
    max_seq = int(config.get("n_positions") or config.get("n_ctx") or 0)
    n_inner = config.get("n_inner")
    d_ff = int(n_inner) if n_inner is not None else 4 * d_model
    if d_model <= 0 or n_heads <= 0 or n_layers <= 0 or vocab_size <= 0 or max_seq <= 0 or d_ff <= 0:
        raise ValueError("GPT-2 dimensions, layers, vocabulary and context length must be positive")
    if d_model % n_heads != 0:
        raise ValueError(f"n_embd={d_model} is not divisible by n_head={n_heads}")
    activation = str(config.get("activation_function") or config.get("hidden_act") or "gelu")
    if activation not in ("gelu", "gelu_new", "relu", "silu"):
        raise ValueError(f"activation '{activation}' has no explicit NIR mapping")
    if activation == "gelu_new":
        warnings.append(
            "activation_function='gelu_new' mapped to NIR 'gelu' (both use the tanh approximation)"
        )
    if config.get("scale_attn_weights") is False:
        raise ValueError("GPT-2 scale_attn_weights=false is not representable by the current attention operator")
    if config.get("position_embedding_type") not in (None, "absolute"):
        raise ValueError("only absolute GPT-2 position embeddings are supported")
    tie = bool(config.get("tie_word_embeddings", True))
    if config.get("layer_norm_epsilon") not in (None, 1e-5, 1e-05):
        warnings.append(
            "layer_norm_epsilon differs from the NIR transformer's fixed 1e-5; "
            "the epsilon is not silently changed"
        )
    for key in ("attn_pdrop", "embd_pdrop", "resid_pdrop", "hidden_dropout_prob"):
        if config.get(key):
            warnings.append(f"{key}={config.get(key)} is a training-time dropout; inference omits it")
    if config.get("type_vocab_size", 1) not in (None, 1):
        warnings.append("type_vocab_size is present but this GPT-2 adapter does not use token-type embeddings")
    if activation == "gelu":
        nir_activation = "gelu_exact"
    elif activation == "gelu_new":
        nir_activation = "gelu"
    else:
        nir_activation = activation
    return {
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq,
        "norm": "layernorm",
        "activation": nir_activation,
        "tie_embeddings": tie,
    }


def _gpt2_tensors(
    arrays: Mapping[str, np.ndarray],
    spec: Mapping[str, Any],
    definition: ArchitectureDefinition,
    unresolved: list[str],
    warnings: list[str],
) -> tuple[
    dict[str, np.ndarray],
    list[dict[str, Any]],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    expected = dict(definition.parameter_shapes)
    canonical: dict[str, np.ndarray] = {}
    mapping: list[dict[str, Any]] = []
    missing: list[str] = []
    shape_errors: list[str] = []
    used: set[str] = set()
    tie = bool(spec["tie_embeddings"])

    def add(target: str, source: str | None, transform: str = "identity", note: str = "") -> None:
        if target not in expected:
            shape_errors.append(f"adapter produced unexpected target '{target}'")
            return
        if source is None or source not in arrays:
            missing.append(target)
            mapping.append(
                {
                    "target": target,
                    "source": source,
                    "transform": transform,
                    "status": "missing",
                    "expected_shape": list(expected[target]),
                    "note": note,
                }
            )
            return
        arr = arrays[source]
        transformed = np.swapaxes(arr, -1, -2) if transform == "transpose" else arr
        shape = [int(d) for d in transformed.shape]
        status = "mapped"
        if tuple(shape) != tuple(expected[target]):
            status = "shape_error"
            shape_errors.append(
                f"{target}: source {source} shape {shape} != expected {list(expected[target])}"
            )
        if status == "mapped":
            canonical[target] = np.ascontiguousarray(transformed)
            used.add(source)
        mapping.append(
            {
                "target": target,
                "source": source,
                "transform": transform,
                "status": status,
                "source_shape": [int(d) for d in arr.shape],
                "expected_shape": list(expected[target]),
                "actual_shape": shape,
                "dtype": dtype_name(arr.dtype),
                "note": note,
            }
        )

    add("embed.weight", "transformer.wte.weight")
    add("pos_embed.weight", "transformer.wpe.weight")
    add("ln_f.weight", "transformer.ln_f.weight")
    add("ln_f.bias", "transformer.ln_f.bias")
    wte = arrays.get("transformer.wte.weight")
    lm_head = arrays.get("lm_head.weight")
    if tie:
        if lm_head is not None:
            if wte is not None and lm_head.shape == wte.shape and not np.array_equal(lm_head, wte):
                unresolved.append(
                    "tied embedding tensors transformer.wte.weight and lm_head.weight differ; "
                    "the checkpoint cannot be represented as a tied graph"
                )
            add("lm_head.weight", "lm_head.weight", note="tied LM head")
        else:
            add("lm_head.weight", "transformer.wte.weight", "tied_embedding", "LM head tied to word embedding")
    else:
        add("lm_head.weight", "lm_head.weight")
        add("lm_head.bias", "lm_head.bias")

    for layer in range(int(spec["n_layers"])):
        src = f"transformer.h.{layer}"
        dst = f"layers.{layer}"
        add(f"{dst}.ln1.weight", f"{src}.ln_1.weight")
        add(f"{dst}.ln1.bias", f"{src}.ln_1.bias")
        add(f"{dst}.qkv.weight", f"{src}.attn.c_attn.weight", "transpose", "GPT-2 Conv1D (in,out) -> NIR (out,in)")
        add(f"{dst}.qkv.bias", f"{src}.attn.c_attn.bias")
        add(f"{dst}.attn_out.weight", f"{src}.attn.c_proj.weight", "transpose", "GPT-2 Conv1D (in,out) -> NIR (out,in)")
        add(f"{dst}.attn_out.bias", f"{src}.attn.c_proj.bias")
        add(f"{dst}.ln2.weight", f"{src}.ln_2.weight")
        add(f"{dst}.ln2.bias", f"{src}.ln_2.bias")
        add(f"{dst}.mlp_up.weight", f"{src}.mlp.c_fc.weight", "transpose", "GPT-2 Conv1D (in,out) -> NIR (out,in)")
        add(f"{dst}.mlp_up.bias", f"{src}.mlp.c_fc.bias")
        add(f"{dst}.mlp_down.weight", f"{src}.mlp.c_proj.weight", "transpose", "GPT-2 Conv1D (in,out) -> NIR (out,in)")
        add(f"{dst}.mlp_down.bias", f"{src}.mlp.c_proj.bias")

    extra = sorted(set(arrays) - used)
    for name in extra:
        unresolved.append(f"source tensor '{name}' is not represented by the reconstructed GPT-2 graph")
    return canonical, mapping, sorted(set(missing)), extra, shape_errors, unresolved


def _execute(
    definition: ArchitectureDefinition, package: WeightPackage, spec: Mapping[str, Any]
) -> dict[str, Any]:
    seq = min(4, int(spec["max_seq_len"]))
    inputs = definition.sample_inputs(batch=1, seq=seq, seed=0)
    try:
        result = execute(definition.graph, inputs, package, check_finite=True, validate_types=True)
        logits = result.output("logits")
        expected = (1, seq, int(spec["vocab_size"]))
        if tuple(logits.shape) != expected:
            return {
                "status": "failed",
                "message": f"logits shape {tuple(logits.shape)} != expected {expected}",
                "inputs": {k: list(v.shape) for k, v in inputs.items()},
            }
        return {
            "status": "passed",
            "message": "forward pass produced finite logits with the expected shape",
            "inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in inputs.items()},
            "outputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in result.outputs.items()},
            "num_ops": result.num_ops,
            "sample_sequence_length": seq,
        }
    except Exception as exc:  # execution failure is part of the report, not hidden
        return {
            "status": "failed",
            "message": str(exc),
            "inputs": {k: list(v.shape) for k, v in inputs.items()},
        }


# --------------------------------------------------------------------------- #
# Llama adapter: explicit RoPE + GQA + SwiGLU graph
# --------------------------------------------------------------------------- #


def _reconstruct_llama(
    arrays: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    architecture: str | None,
    source_dtypes: Mapping[str, str] | None,
    name: str | None,
    source_hash: str | None,
    on_progress: Any | None = None,
    profile: HFProfile | None = None,
) -> ReconstructionResult:
    progress: list[dict[str, Any]] = []
    unresolved: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []
    extra: list[str] = []
    shape_errors: list[str] = []
    reasons: list[str] = []

    def step(stage: str, status: str, message: str) -> None:
        progress.append({"stage": stage, "status": status, "message": message})
        if on_progress is not None:
            on_progress(stage, status, message)

    try:
        spec = _llama_spec(config, warnings, profile=profile)
    except ValueError as exc:
        step("config", "unsupported", str(exc))
        return ReconstructionResult(
            report=ReconstructionReport(
                status="unsupported",
                architecture=architecture,
                family="transformer_decoder",
                spec={},
                graph=None,
                graph_summary=None,
                tensor_mapping=[],
                missing=[],
                extra=sorted(arrays),
                shape_errors=[],
                unresolved=[str(exc)],
                warnings=warnings,
                validation=None,
                execution=None,
                source_tensor_count=len(arrays),
                mapped_tensor_count=0,
                progress=progress,
                reasons=[str(exc)],
            )
        )
    step("config", "complete", "Llama/RoPE/GQA configuration validated")

    definition = _build_llama_definition(spec)
    step("architecture", "complete", "explicit Llama NIR graph with RoPE, GQA and SwiGLU built")
    canonical, mapping, missing, extra, shape_errors, unresolved = _llama_tensors(
        arrays, spec, definition, unresolved
    )
    step(
        "tensor_mapping",
        "complete" if not missing and not shape_errors else "failed",
        f"mapped {len(mapping)} tensors; {len(missing)} missing, {len(shape_errors)} shape errors, "
        f"{len(extra)} unmapped source tensors",
    )
    if missing or shape_errors:
        reasons.append("all Llama NIR parameters must be present with expected shapes; none are invented")
        return ReconstructionResult(
            report=ReconstructionReport(
                status="unsupported",
                architecture=architecture,
                family="transformer_decoder",
                spec=spec,
                graph=None,
                graph_summary=None,
                tensor_mapping=mapping,
                missing=missing,
                extra=extra,
                shape_errors=shape_errors,
                unresolved=unresolved,
                warnings=warnings,
                validation=None,
                execution=None,
                source_tensor_count=len(arrays),
                mapped_tensor_count=len(canonical),
                progress=progress,
                reasons=reasons,
            )
        )

    package = WeightPackage.from_arrays(
        canonical,
        name=name or "llama_model",
        family="transformer_decoder",
        family_spec=spec,
        metadata={
            "reconstruction": {
                "architecture": architecture,
                "adapter": "llama",
                "source_hash": source_hash,
                "tensor_mapping": mapping,
                "graph": definition.graph.to_dict(),
            },
            "source_hash": source_hash,
        },
        source_dtypes={
            target: source_dtypes[item["source"]] if source_dtypes and item["source"] in source_dtypes else item["dtype"]
            for target, item in ((m["target"], m) for m in mapping)
        },
    )
    validation = definition.graph.validate(package)
    validation_doc = {
        "ok": validation.ok,
        "errors": list(validation.errors),
        "warnings": list(validation.warnings),
        "num_nodes": validation.num_nodes,
        "num_params": validation.num_params,
        "param_names": list(validation.param_names),
        "order": list(validation.order),
    }
    step(
        "graph_validation",
        "complete" if validation.ok else "failed",
        f"graph validation {'passed' if validation.ok else 'failed'} with {validation.num_nodes} nodes",
    )
    if not validation.ok:
        reasons.append("Llama graph validation failed; no validated artifact should be published")
        return ReconstructionResult(
            report=ReconstructionReport(
                status="failed",
                architecture=architecture,
                family="transformer_decoder",
                spec=spec,
                graph=definition.graph.to_dict(),
                graph_summary=definition.graph.summary(),
                tensor_mapping=mapping,
                missing=missing,
                extra=extra,
                shape_errors=shape_errors,
                unresolved=unresolved,
                warnings=warnings,
                validation=validation_doc,
                execution=None,
                source_tensor_count=len(arrays),
                mapped_tensor_count=len(canonical),
                progress=progress,
                reasons=reasons,
            )
        )

    execution = _execute_llama(definition, package, spec)
    step(
        "forward_execution",
        execution.get("status", "failed"),
        execution.get("message", "forward pass completed"),
    )
    if extra:
        unresolved.append(
            "source tensors not represented by the Llama graph: " + ", ".join(extra[:12])
        )
    status = "validated" if execution.get("status") == "passed" and not extra and not unresolved else "partial"
    if execution.get("status") != "passed":
        status = "failed"
        reasons.append("Llama forward execution failed; the graph is not marked validated")
    elif status == "partial":
        reasons.append("Llama graph executes, but some source tensors remain unresolved")
    else:
        reasons.append("all Llama tensors mapped, graph validated, and forward execution produced finite logits")
    report = ReconstructionReport(
        status=status,
        architecture=architecture,
        family="transformer_decoder",
        spec=spec,
        graph=definition.graph.to_dict(),
        graph_summary=definition.graph.summary(),
        tensor_mapping=mapping,
        missing=missing,
        extra=extra,
        shape_errors=shape_errors,
        unresolved=unresolved,
        warnings=warnings,
        validation=validation_doc,
        execution=execution,
        source_tensor_count=len(arrays),
        mapped_tensor_count=len(canonical),
        progress=progress,
        reasons=reasons,
    )
    if status == "failed":
        return ReconstructionResult(report=report, package=None, definition=definition)
    return ReconstructionResult(report=report, package=package, definition=definition)


def _llama_spec(
    config: Mapping[str, Any],
    warnings: list[str],
    *,
    profile: HFProfile | None = None,
) -> dict[str, Any]:
    try:
        hidden = int(config["hidden_size"])
        n_heads = int(config["num_attention_heads"])
        n_layers = int(config["num_hidden_layers"])
        intermediate = int(config["intermediate_size"])
        vocab = int(config["vocab_size"])
        max_seq = int(config["max_position_embeddings"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Llama configuration is missing a required integer field: {exc}") from exc
    n_kv = int(config.get("num_key_value_heads", n_heads))
    head_dim = int(config.get("head_dim") or hidden // n_heads)
    if min(hidden, n_heads, n_layers, intermediate, vocab, max_seq, n_kv, head_dim) <= 0:
        raise ValueError("Llama dimensions, layers, vocabulary and context length must be positive")
    if n_heads % n_kv != 0:
        raise ValueError(f"num_attention_heads={n_heads} must be divisible by num_key_value_heads={n_kv}")
    qkv_bias = bool(profile.qkv_bias) if profile is not None else False
    o_bias = bool(profile.o_bias) if profile is not None else False
    mlp_bias = bool(profile.mlp_bias) if profile is not None else False
    norm_affine = bool(profile.norm_affine) if profile is not None else True
    if config.get("attention_bias") and not qkv_bias:
        raise ValueError("attention_bias=true has no explicit mapping for this profile")
    if config.get("mlp_bias") and not mlp_bias:
        raise ValueError("mlp_bias=true has no explicit mapping for this profile")
    partial = float(
        config.get(
            "partial_rotary_factor",
            0.25 if profile is not None and profile.partial_rotary else 1.0,
        )
    )
    if partial != 1.0 and not (profile is not None and profile.partial_rotary):
        raise ValueError("partial_rotary_factor has no explicit mapping for this profile")
    if partial <= 0 or partial > 1:
        raise ValueError("partial_rotary_factor must be in (0, 1]")
    rotary_dim = int(head_dim * partial)
    if rotary_dim <= 0 or rotary_dim % 2 != 0:
        raise ValueError("partial rotary dimension must be a positive even integer")
    clip_qkv = float(config.get("clip_qkv") or 0.0)
    if clip_qkv < 0:
        raise ValueError("clip_qkv must be non-negative")
    if clip_qkv and not (profile is not None and profile.clip_qkv):
        raise ValueError("clip_qkv has no explicit mapping for this profile")
    if config.get("rope_scaling") is not None:
        raise ValueError("rope_scaling is not implemented; no scaled-RoPE graph was guessed")
    activation = str(config.get("hidden_act") or "silu")
    if activation != "silu":
        raise ValueError(f"hidden_act='{activation}' is not supported; Llama adapter implements SwiGLU with silu")
    norm_type = "layernorm" if profile is not None and profile.layer_norm else "rmsnorm"
    eps_default = (
        profile.norm_eps
        if profile is not None and profile.norm_eps is not None
        else (1e-5 if norm_type == "layernorm" else 1e-6)
    )
    eps = float(config.get("layer_norm_eps", config.get("rms_norm_eps", eps_default)))
    theta = float(config.get("rope_theta", 10000.0))
    if eps <= 0 or theta <= 0:
        raise ValueError("rms_norm_eps and rope_theta must be positive")
    if config.get("attention_dropout"):
        warnings.append("attention_dropout is a training-time setting; inference omits it")
    if config.get("pretraining_tp", 1) not in (1, None):
        raise ValueError("pretraining_tp != 1 is a tensor-parallel layout and is not supported")
    return {
        "architecture": profile.model_type if profile is not None else "llama",
        "profile": profile.model_type if profile is not None else "llama",
        "hidden_size": hidden,
        "n_heads": n_heads,
        "n_kv_heads": n_kv,
        "head_dim": head_dim,
        "n_layers": n_layers,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
        "max_seq_len": max_seq,
        "rms_norm_eps": eps,
        "norm_type": norm_type,
        "rope_theta": theta,
        "rotary_dim": rotary_dim,
        "clip_qkv": clip_qkv,
        "qkv_bias": qkv_bias,
        "o_bias": o_bias,
        "mlp_bias": mlp_bias,
        "norm_affine": norm_affine,
        "activation": "silu",
        "tie_embeddings": bool(config.get("tie_word_embeddings", False)),
    }


def _build_llama_definition(spec: Mapping[str, Any]) -> ArchitectureDefinition:
    d = int(spec["hidden_size"])
    heads = int(spec["n_heads"])
    kv_heads = int(spec["n_kv_heads"])
    head_dim = int(spec["head_dim"])
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    eps = float(spec["rms_norm_eps"])
    theta = float(spec["rope_theta"])
    layers = int(spec["n_layers"])
    intermediate = int(spec["intermediate_size"])
    vocab = int(spec["vocab_size"])
    architecture = str(spec.get("profile") or "llama")
    qkv_bias = bool(spec.get("qkv_bias", False))
    o_bias = bool(spec.get("o_bias", False))
    mlp_bias = bool(spec.get("mlp_bias", False))
    norm_affine = bool(spec.get("norm_affine", True))
    norm_type = str(spec.get("norm_type", "rmsnorm"))
    if norm_type not in ("rmsnorm", "layernorm"):
        raise ValueError(f"unsupported norm type '{norm_type}'")
    norm_op = "layernorm" if norm_type == "layernorm" else "rmsnorm"
    rotary_dim = int(spec.get("rotary_dim", head_dim))
    clip_qkv = float(spec.get("clip_qkv", 0.0))
    g = Graph(name=f"{architecture}_L{layers}", metadata={"family": "transformer_decoder", "architecture": architecture, "spec": dict(spec)})
    g.add_input("tokens", TensorType.of(("B", "T"), "int64"))
    g.add_input("positions", TensorType.of(("B", "T"), "int64"))
    sh: dict[str, tuple[int, ...]] = {
        "embed.weight": (vocab, d),
        "lm_head.weight": (vocab, d),
    }
    if norm_affine:
        sh["norm.weight"] = (d,)
        if norm_type == "layernorm":
            sh["norm.bias"] = (d,)
    cur = g.add_node("embed", "embedding", {"x": Ref("input:tokens")}, {"weight": "param:embed.weight"})

    def split_heads(x: Ref, layer: int, name: str, heads_count: int) -> Ref:
        reshaped = g.add_node(
            f"l{layer}_{name}_reshape",
            "reshape",
            {"x": x},
            {"shape": ["B", "T", heads_count, head_dim]},
        )
        return g.add_node(
            f"l{layer}_{name}_heads",
            "transpose",
            {"x": reshaped},
            {"perm": [0, 2, 1, 3]},
        )

    for layer in range(layers):
        p = f"layers.{layer}"
        sh[f"{p}.q_proj.weight"] = (q_dim, d)
        sh[f"{p}.k_proj.weight"] = (kv_dim, d)
        sh[f"{p}.v_proj.weight"] = (kv_dim, d)
        sh[f"{p}.o_proj.weight"] = (d, q_dim)
        sh[f"{p}.gate_proj.weight"] = (intermediate, d)
        sh[f"{p}.up_proj.weight"] = (intermediate, d)
        sh[f"{p}.down_proj.weight"] = (d, intermediate)
        if norm_affine:
            sh[f"{p}.input_norm.weight"] = (d,)
            sh[f"{p}.post_norm.weight"] = (d,)
            norm1_params = {"weight": f"param:{p}.input_norm.weight", "eps": eps}
            norm2_params = {"weight": f"param:{p}.post_norm.weight", "eps": eps}
            if norm_type == "layernorm":
                sh[f"{p}.input_norm.bias"] = (d,)
                sh[f"{p}.post_norm.bias"] = (d,)
                norm1_params["bias"] = f"param:{p}.input_norm.bias"
                norm2_params["bias"] = f"param:{p}.post_norm.bias"
        else:
            norm1_params = {"eps": eps}
            norm2_params = {"eps": eps}
        if qkv_bias:
            sh[f"{p}.q_proj.bias"] = (q_dim,)
            sh[f"{p}.k_proj.bias"] = (kv_dim,)
            sh[f"{p}.v_proj.bias"] = (kv_dim,)
        if o_bias:
            sh[f"{p}.o_proj.bias"] = (d,)
        if mlp_bias:
            sh[f"{p}.gate_proj.bias"] = (intermediate,)
            sh[f"{p}.up_proj.bias"] = (intermediate,)
            sh[f"{p}.down_proj.bias"] = (d,)

        def qkv_params(name: str) -> dict[str, str]:
            params = {"weight": f"param:{p}.{name}_proj.weight"}
            if qkv_bias:
                params["bias"] = f"param:{p}.{name}_proj.bias"
            return params

        def mlp_params(name: str) -> dict[str, str]:
            params = {"weight": f"param:{p}.{name}_proj.weight"}
            if mlp_bias:
                params["bias"] = f"param:{p}.{name}_proj.bias"
            return params

        norm1 = g.add_node(
            f"l{layer}_input_norm",
            norm_op,
            {"x": cur},
            norm1_params,
        )
        q = g.add_node(f"l{layer}_q_proj", "linear", {"x": norm1}, qkv_params("q"))
        k = g.add_node(f"l{layer}_k_proj", "linear", {"x": norm1}, qkv_params("k"))
        v = g.add_node(f"l{layer}_v_proj", "linear", {"x": norm1}, qkv_params("v"))
        if clip_qkv:
            q = g.add_node(f"l{layer}_q_clip", "clip", {"x": q}, {"min": -clip_qkv, "max": clip_qkv})
            k = g.add_node(f"l{layer}_k_clip", "clip", {"x": k}, {"min": -clip_qkv, "max": clip_qkv})
            v = g.add_node(f"l{layer}_v_clip", "clip", {"x": v}, {"min": -clip_qkv, "max": clip_qkv})
        qh = split_heads(q, layer, "q", heads)
        kh = split_heads(k, layer, "k", kv_heads)
        vh = split_heads(v, layer, "v", kv_heads)
        qr = g.add_node(
            f"l{layer}_q_rope",
            "rope",
            {"x": qh, "positions": Ref("input:positions")},
            {"theta": theta, "rotary_dim": rotary_dim},
        )
        kr = g.add_node(
            f"l{layer}_k_rope",
            "rope",
            {"x": kh, "positions": Ref("input:positions")},
            {"theta": theta, "rotary_dim": rotary_dim},
        )
        kr = g.add_node(
            f"l{layer}_k_repeat",
            "repeat_kv",
            {"x": kr},
            {"n_heads": heads},
        )
        vr = g.add_node(
            f"l{layer}_v_repeat",
            "repeat_kv",
            {"x": vh},
            {"n_heads": heads},
        )
        attn = g.add_node(
            f"l{layer}_attn",
            "attention",
            {"q": qr, "k": kr, "v": vr},
            {"causal": True, "scale": "auto"},
        )
        merged = g.add_node(f"l{layer}_merge", "merge_heads", {"x": attn}, {"n_heads": heads})
        o_params = {"weight": f"param:{p}.o_proj.weight"}
        if o_bias:
            o_params["bias"] = f"param:{p}.o_proj.bias"
        proj = g.add_node(f"l{layer}_o_proj", "linear", {"x": merged}, o_params)
        cur = g.add_node(f"l{layer}_res1", "add", {"a": cur, "b": proj})
        norm2 = g.add_node(
            f"l{layer}_post_norm",
            norm_op,
            {"x": cur},
            norm2_params,
        )
        gate = g.add_node(f"l{layer}_gate", "linear", {"x": norm2}, mlp_params("gate"))
        gate = g.add_node(f"l{layer}_silu", "silu", {"x": gate})
        up = g.add_node(f"l{layer}_up", "linear", {"x": norm2}, mlp_params("up"))
        gated = g.add_node(f"l{layer}_gated", "mul", {"a": gate, "b": up})
        down = g.add_node(f"l{layer}_down", "linear", {"x": gated}, mlp_params("down"))
        cur = g.add_node(f"l{layer}_res2", "add", {"a": cur, "b": down})

    final_params = {"eps": eps}
    if norm_affine:
        final_params["weight"] = "param:norm.weight"
        if norm_type == "layernorm":
            final_params["bias"] = "param:norm.bias"
    final = g.add_node("final_norm", norm_op, {"x": cur}, final_params)
    logits = g.add_node(
        "lm_head",
        "linear",
        {"x": final},
        {"weight": "param:lm_head.weight"},
    )
    g.set_output("logits", logits)
    g.set_output("hidden", final)
    return ArchitectureDefinition(
        family="transformer_decoder",
        spec=dict(spec),
        graph=g,
        parameter_shapes=sh,
        default_inputs={"tokens": ("B", "T"), "positions": ("B", "T")},
        input_dtypes={"tokens": "int64", "positions": "int64"},
    )


def _llama_tensors(
    arrays: Mapping[str, np.ndarray],
    spec: Mapping[str, Any],
    definition: ArchitectureDefinition,
    unresolved: list[str],
) -> tuple[
    dict[str, np.ndarray],
    list[dict[str, Any]],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    expected = dict(definition.parameter_shapes)
    canonical: dict[str, np.ndarray] = {}
    mapping: list[dict[str, Any]] = []
    missing: list[str] = []
    shape_errors: list[str] = []
    used: set[str] = set()
    tie = bool(spec["tie_embeddings"])
    norm_affine = bool(spec.get("norm_affine", True))
    qkv_bias = bool(spec.get("qkv_bias", False))
    o_bias = bool(spec.get("o_bias", False))
    mlp_bias = bool(spec.get("mlp_bias", False))

    def add(target: str, source: str | None, note: str = "") -> None:
        if target not in expected:
            shape_errors.append(f"adapter produced unexpected target '{target}'")
            return
        if source is None or source not in arrays:
            missing.append(target)
            mapping.append({"target": target, "source": source, "transform": "identity", "status": "missing", "expected_shape": list(expected[target]), "note": note})
            return
        arr = arrays[source]
        shape = [int(v) for v in arr.shape]
        status = "mapped"
        if tuple(shape) != tuple(expected[target]):
            status = "shape_error"
            shape_errors.append(f"{target}: source {source} shape {shape} != expected {list(expected[target])}")
        if status == "mapped":
            canonical[target] = np.ascontiguousarray(arr)
            used.add(source)
        mapping.append({"target": target, "source": source, "transform": "identity", "status": status, "source_shape": shape, "expected_shape": list(expected[target]), "dtype": dtype_name(arr.dtype), "note": note})

    add("embed.weight", "model.embed_tokens.weight")
    if norm_affine:
        add("norm.weight", "model.norm.weight")
        if spec.get("norm_type") == "layernorm":
            add("norm.bias", "model.norm.bias")
    lm_head = arrays.get("lm_head.weight")
    if tie:
        if lm_head is not None:
            if "model.embed_tokens.weight" in arrays and lm_head.shape == arrays["model.embed_tokens.weight"].shape and not np.array_equal(lm_head, arrays["model.embed_tokens.weight"]):
                unresolved.append("tied embed_tokens/lm_head tensors differ; no tied Llama graph is claimed")
            add("lm_head.weight", "lm_head.weight", "tied LM head")
        else:
            add("lm_head.weight", "model.embed_tokens.weight", "LM head tied to embedding")
    else:
        add("lm_head.weight", "lm_head.weight")

    for layer in range(int(spec["n_layers"])):
        src = f"model.layers.{layer}"
        dst = f"layers.{layer}"
        if norm_affine:
            add(f"{dst}.input_norm.weight", f"{src}.input_layernorm.weight")
            add(f"{dst}.post_norm.weight", f"{src}.post_attention_layernorm.weight")
            if spec.get("norm_type") == "layernorm":
                add(f"{dst}.input_norm.bias", f"{src}.input_layernorm.bias")
                add(f"{dst}.post_norm.bias", f"{src}.post_attention_layernorm.bias")
        add(f"{dst}.q_proj.weight", f"{src}.self_attn.q_proj.weight")
        add(f"{dst}.k_proj.weight", f"{src}.self_attn.k_proj.weight")
        add(f"{dst}.v_proj.weight", f"{src}.self_attn.v_proj.weight")
        if qkv_bias:
            add(f"{dst}.q_proj.bias", f"{src}.self_attn.q_proj.bias")
            add(f"{dst}.k_proj.bias", f"{src}.self_attn.k_proj.bias")
            add(f"{dst}.v_proj.bias", f"{src}.self_attn.v_proj.bias")
        add(f"{dst}.o_proj.weight", f"{src}.self_attn.o_proj.weight")
        if o_bias:
            add(f"{dst}.o_proj.bias", f"{src}.self_attn.o_proj.bias")
        add(f"{dst}.gate_proj.weight", f"{src}.mlp.gate_proj.weight")
        add(f"{dst}.up_proj.weight", f"{src}.mlp.up_proj.weight")
        add(f"{dst}.down_proj.weight", f"{src}.mlp.down_proj.weight")
        if mlp_bias:
            add(f"{dst}.gate_proj.bias", f"{src}.mlp.gate_proj.bias")
            add(f"{dst}.up_proj.bias", f"{src}.mlp.up_proj.bias")
            add(f"{dst}.down_proj.bias", f"{src}.mlp.down_proj.bias")

    extra = sorted(set(arrays) - used)
    for name in extra:
        unresolved.append(f"source tensor '{name}' is not represented by the Llama graph")
    return canonical, mapping, sorted(set(missing)), extra, shape_errors, unresolved


def _execute_llama(
    definition: ArchitectureDefinition, package: WeightPackage, spec: Mapping[str, Any]
) -> dict[str, Any]:
    seq = min(4, int(spec["max_seq_len"]))
    tokens = np.arange(seq, dtype=np.int64).reshape(1, seq)
    positions = np.arange(seq, dtype=np.int64).reshape(1, seq)
    inputs = {"tokens": tokens, "positions": positions}
    try:
        result = execute(definition.graph, inputs, package, check_finite=True, validate_types=True)
        logits = result.output("logits")
        expected = (1, seq, int(spec["vocab_size"]))
        if tuple(logits.shape) != expected:
            return {"status": "failed", "message": f"logits shape {tuple(logits.shape)} != expected {expected}", "inputs": {k: list(v.shape) for k, v in inputs.items()}}
        return {
            "status": "passed",
            "message": "Llama forward pass produced finite logits with the expected shape",
            "inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in inputs.items()},
            "outputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in result.outputs.items()},
            "num_ops": result.num_ops,
            "sample_sequence_length": seq,
        }
    except Exception as exc:
        return {"status": "failed", "message": str(exc), "inputs": {k: list(v.shape) for k, v in inputs.items()}}


# --------------------------------------------------------------------------- #
# OPT adapter: learned absolute positions (+2 offset), pre-LN, biased linears
# --------------------------------------------------------------------------- #


def _reconstruct_opt(
    arrays: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    architecture: str | None,
    source_dtypes: Mapping[str, str] | None,
    name: str | None,
    source_hash: str | None,
    on_progress: Any | None = None,
) -> ReconstructionResult:
    progress: list[dict[str, Any]] = []
    unresolved: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []
    extra: list[str] = []
    shape_errors: list[str] = []
    reasons: list[str] = []

    def step(stage: str, status: str, message: str) -> None:
        progress.append({"stage": stage, "status": status, "message": message})
        if on_progress is not None:
            on_progress(stage, status, message)

    try:
        spec = _opt_spec(config, warnings)
    except ValueError as exc:
        step("config", "unsupported", str(exc))
        return ReconstructionResult(
            report=ReconstructionReport(
                status="unsupported",
                architecture=architecture,
                family="transformer_decoder",
                spec={},
                graph=None,
                graph_summary=None,
                tensor_mapping=[],
                missing=[],
                extra=sorted(arrays),
                shape_errors=[],
                unresolved=[str(exc)],
                warnings=warnings,
                validation=None,
                execution=None,
                source_tensor_count=len(arrays),
                mapped_tensor_count=0,
                progress=progress,
                reasons=[str(exc)],
            )
        )
    step("config", "complete", "OPT configuration validated against the supported adapter")

    definition = _build_opt_definition(spec)
    step("architecture", "complete", "explicit OPT NIR graph with offset positions and biased projections built")
    canonical, mapping, missing, extra, shape_errors, unresolved = _opt_tensors(
        arrays, spec, definition, unresolved
    )
    step(
        "tensor_mapping",
        "complete" if not missing and not shape_errors else "failed",
        f"mapped {len(mapping)} tensors; {len(missing)} missing, {len(shape_errors)} shape errors, "
        f"{len(extra)} unmapped source tensors",
    )
    if missing or shape_errors:
        reasons.append("all OPT NIR parameters must be present with expected shapes; none are invented")
        return ReconstructionResult(
            report=ReconstructionReport(
                status="unsupported",
                architecture=architecture,
                family="transformer_decoder",
                spec=spec,
                graph=None,
                graph_summary=None,
                tensor_mapping=mapping,
                missing=missing,
                extra=extra,
                shape_errors=shape_errors,
                unresolved=unresolved,
                warnings=warnings,
                validation=None,
                execution=None,
                source_tensor_count=len(arrays),
                mapped_tensor_count=len(canonical),
                progress=progress,
                reasons=reasons,
            )
        )

    package = WeightPackage.from_arrays(
        canonical,
        name=name or "opt_model",
        family="transformer_decoder",
        family_spec=spec,
        metadata={
            "reconstruction": {
                "architecture": architecture,
                "adapter": "opt",
                "source_hash": source_hash,
                "tensor_mapping": mapping,
                "graph": definition.graph.to_dict(),
            },
            "source_hash": source_hash,
        },
        source_dtypes={
            target: source_dtypes[item["source"]] if source_dtypes and item["source"] in source_dtypes else item["dtype"]
            for target, item in ((m["target"], m) for m in mapping)
        },
    )
    validation = definition.graph.validate(package)
    validation_doc = {
        "ok": validation.ok,
        "errors": list(validation.errors),
        "warnings": list(validation.warnings),
        "num_nodes": validation.num_nodes,
        "num_params": validation.num_params,
        "param_names": list(validation.param_names),
        "order": list(validation.order),
    }
    step(
        "graph_validation",
        "complete" if validation.ok else "failed",
        f"graph validation {'passed' if validation.ok else 'failed'} with {validation.num_nodes} nodes",
    )
    if not validation.ok:
        reasons.append("OPT graph validation failed; no validated artifact should be published")
        return ReconstructionResult(
            report=ReconstructionReport(
                status="failed",
                architecture=architecture,
                family="transformer_decoder",
                spec=spec,
                graph=definition.graph.to_dict(),
                graph_summary=definition.graph.summary(),
                tensor_mapping=mapping,
                missing=missing,
                extra=extra,
                shape_errors=shape_errors,
                unresolved=unresolved,
                warnings=warnings,
                validation=validation_doc,
                execution=None,
                source_tensor_count=len(arrays),
                mapped_tensor_count=len(canonical),
                progress=progress,
                reasons=reasons,
            )
        )

    execution = _execute_opt(definition, package, spec)
    step(
        "forward_execution",
        execution.get("status", "failed"),
        execution.get("message", "forward pass completed"),
    )
    if extra:
        unresolved.append(
            "source tensors not represented by the OPT graph: " + ", ".join(extra[:12])
        )
    status = "validated" if execution.get("status") == "passed" and not extra and not unresolved else "partial"
    if execution.get("status") != "passed":
        status = "failed"
        reasons.append("OPT forward execution failed; the graph is not marked validated")
    elif status == "partial":
        reasons.append("OPT graph executes, but some source tensors remain unresolved")
    else:
        reasons.append("all OPT tensors mapped, graph validated, and forward execution produced finite logits")
    report = ReconstructionReport(
        status=status,
        architecture=architecture,
        family="transformer_decoder",
        spec=spec,
        graph=definition.graph.to_dict(),
        graph_summary=definition.graph.summary(),
        tensor_mapping=mapping,
        missing=missing,
        extra=extra,
        shape_errors=shape_errors,
        unresolved=unresolved,
        warnings=warnings,
        validation=validation_doc,
        execution=execution,
        source_tensor_count=len(arrays),
        mapped_tensor_count=len(canonical),
        progress=progress,
        reasons=reasons,
    )
    if status == "failed":
        return ReconstructionResult(report=report, package=None, definition=definition)
    return ReconstructionResult(report=report, package=package, definition=definition)


def _opt_spec(config: Mapping[str, Any], warnings: list[str]) -> dict[str, Any]:
    try:
        d_model = int(config["hidden_size"])
        n_heads = int(config["num_attention_heads"])
        n_layers = int(config["num_hidden_layers"])
        d_ff = int(config["ffn_dim"])
        vocab_size = int(config["vocab_size"])
        max_seq = int(config["max_position_embeddings"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"OPT configuration is missing a required integer field: {exc}") from exc
    if min(d_model, n_heads, n_layers, d_ff, vocab_size, max_seq) <= 0:
        raise ValueError("OPT dimensions, layers, vocabulary and context length must be positive")
    if d_model % n_heads != 0:
        raise ValueError(f"hidden_size={d_model} is not divisible by num_attention_heads={n_heads}")
    word_embed_proj_dim = int(config.get("word_embed_proj_dim") or d_model)
    if word_embed_proj_dim != d_model:
        raise ValueError(
            "word_embed_proj_dim differs from hidden_size; OPT project_in/project_out "
            "are not implemented"
        )
    if config.get("do_layer_norm_before") is False:
        raise ValueError("do_layer_norm_before=false is not represented by this OPT adapter")
    if config.get("_remove_final_layer_norm"):
        raise ValueError("_remove_final_layer_norm=true is not represented by this OPT adapter")
    if config.get("scale_embedding"):
        raise ValueError("scale_embedding=true is not represented by this OPT adapter")
    activation = str(config.get("activation_function") or "relu")
    if activation != "relu":
        raise ValueError(
            f"activation_function='{activation}' is not supported; this OPT adapter implements relu"
        )
    eps = float(config.get("layer_norm_eps") or 1e-5)
    if eps <= 0:
        raise ValueError("layer_norm_eps must be positive")
    if config.get("attention_dropout"):
        warnings.append("attention_dropout is a training-time setting; inference omits it")
    if config.get("dropout"):
        warnings.append("dropout is a training-time setting; inference omits it")
    if config.get("layerdrop"):
        warnings.append("layerdrop is a training-time setting; inference omits it")
    return {
        "architecture": "opt",
        "d_model": d_model,
        "n_heads": n_heads,
        "head_dim": d_model // n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq,
        "pos_table_size": max_seq + 2,
        "position_offset": 2,
        "norm": "layernorm",
        "layer_norm_eps": eps,
        "activation": "relu",
        "tie_embeddings": True
        if config.get("tie_word_embeddings") is None
        else bool(config.get("tie_word_embeddings")),
    }


def _build_opt_definition(spec: Mapping[str, Any]) -> ArchitectureDefinition:
    d = int(spec["d_model"])
    heads = int(spec["n_heads"])
    head_dim = int(spec["head_dim"])
    layers = int(spec["n_layers"])
    ffn = int(spec["d_ff"])
    vocab = int(spec["vocab_size"])
    table = int(spec["pos_table_size"])
    offset = int(spec["position_offset"])
    eps = float(spec["layer_norm_eps"])
    g = Graph(
        name=f"opt_L{layers}",
        metadata={"family": "transformer_decoder", "architecture": "opt", "spec": dict(spec)},
    )
    g.add_input("tokens", TensorType.of(("B", "T"), "int64"))
    sh: dict[str, tuple[int, ...]] = {
        "embed.weight": (vocab, d),
        "pos_embed.weight": (table, d),
        "lm_head.weight": (vocab, d),
    }
    cur = g.add_node("embed", "embedding", {"x": Ref("input:tokens")}, {"weight": "param:embed.weight"})
    cur = g.add_node(
        "pos",
        "pos_embedding",
        {"x": cur},
        {"weight": "param:pos_embed.weight", "offset": offset},
    )

    def split_heads(x: Ref, layer: int, name: str) -> Ref:
        reshaped = g.add_node(
            f"l{layer}_{name}_reshape",
            "reshape",
            {"x": x},
            {"shape": ["B", "T", heads, head_dim]},
        )
        return g.add_node(
            f"l{layer}_{name}_heads",
            "transpose",
            {"x": reshaped},
            {"perm": [0, 2, 1, 3]},
        )

    for layer in range(layers):
        p = f"layers.{layer}"
        sh[f"{p}.ln1.weight"] = (d,)
        sh[f"{p}.ln1.bias"] = (d,)
        sh[f"{p}.q_proj.weight"] = (d, d)
        sh[f"{p}.q_proj.bias"] = (d,)
        sh[f"{p}.k_proj.weight"] = (d, d)
        sh[f"{p}.k_proj.bias"] = (d,)
        sh[f"{p}.v_proj.weight"] = (d, d)
        sh[f"{p}.v_proj.bias"] = (d,)
        sh[f"{p}.attn_out.weight"] = (d, d)
        sh[f"{p}.attn_out.bias"] = (d,)
        sh[f"{p}.ln2.weight"] = (d,)
        sh[f"{p}.ln2.bias"] = (d,)
        sh[f"{p}.mlp_up.weight"] = (ffn, d)
        sh[f"{p}.mlp_up.bias"] = (ffn,)
        sh[f"{p}.mlp_down.weight"] = (d, ffn)
        sh[f"{p}.mlp_down.bias"] = (d,)

        ln1 = g.add_node(
            f"l{layer}_ln1",
            "layernorm",
            {"x": cur},
            {
                "weight": f"param:{p}.ln1.weight",
                "bias": f"param:{p}.ln1.bias",
                "eps": eps,
            },
        )
        q = g.add_node(
            f"l{layer}_q_proj",
            "linear",
            {"x": ln1},
            {"weight": f"param:{p}.q_proj.weight", "bias": f"param:{p}.q_proj.bias"},
        )
        k = g.add_node(
            f"l{layer}_k_proj",
            "linear",
            {"x": ln1},
            {"weight": f"param:{p}.k_proj.weight", "bias": f"param:{p}.k_proj.bias"},
        )
        v = g.add_node(
            f"l{layer}_v_proj",
            "linear",
            {"x": ln1},
            {"weight": f"param:{p}.v_proj.weight", "bias": f"param:{p}.v_proj.bias"},
        )
        qh = split_heads(q, layer, "q")
        kh = split_heads(k, layer, "k")
        vh = split_heads(v, layer, "v")
        attn = g.add_node(
            f"l{layer}_attn",
            "attention",
            {"q": qh, "k": kh, "v": vh},
            {"causal": True, "scale": "auto"},
        )
        merged = g.add_node(f"l{layer}_merge", "merge_heads", {"x": attn}, {"n_heads": heads})
        proj = g.add_node(
            f"l{layer}_attn_out",
            "linear",
            {"x": merged},
            {"weight": f"param:{p}.attn_out.weight", "bias": f"param:{p}.attn_out.bias"},
        )
        cur = g.add_node(f"l{layer}_res1", "add", {"a": cur, "b": proj})
        ln2 = g.add_node(
            f"l{layer}_ln2",
            "layernorm",
            {"x": cur},
            {
                "weight": f"param:{p}.ln2.weight",
                "bias": f"param:{p}.ln2.bias",
                "eps": eps,
            },
        )
        up = g.add_node(
            f"l{layer}_mlp_up",
            "linear",
            {"x": ln2},
            {"weight": f"param:{p}.mlp_up.weight", "bias": f"param:{p}.mlp_up.bias"},
        )
        act = g.add_node(f"l{layer}_act", "relu", {"x": up})
        down = g.add_node(
            f"l{layer}_mlp_down",
            "linear",
            {"x": act},
            {"weight": f"param:{p}.mlp_down.weight", "bias": f"param:{p}.mlp_down.bias"},
        )
        cur = g.add_node(f"l{layer}_res2", "add", {"a": cur, "b": down})

    sh["ln_f.weight"] = (d,)
    sh["ln_f.bias"] = (d,)
    final = g.add_node(
        "ln_f",
        "layernorm",
        {"x": cur},
        {"weight": "param:ln_f.weight", "bias": "param:ln_f.bias", "eps": eps},
    )
    logits = g.add_node(
        "lm_head",
        "linear",
        {"x": final},
        {"weight": "param:lm_head.weight"},
    )
    g.set_output("logits", logits)
    g.set_output("hidden", final)
    return ArchitectureDefinition(
        family="transformer_decoder",
        spec=dict(spec),
        graph=g,
        parameter_shapes=sh,
        default_inputs={"tokens": ("B", "T")},
        input_dtypes={"tokens": "int64"},
    )


def _opt_tensors(
    arrays: Mapping[str, np.ndarray],
    spec: Mapping[str, Any],
    definition: ArchitectureDefinition,
    unresolved: list[str],
) -> tuple[
    dict[str, np.ndarray],
    list[dict[str, Any]],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    expected = dict(definition.parameter_shapes)
    canonical: dict[str, np.ndarray] = {}
    mapping: list[dict[str, Any]] = []
    missing: list[str] = []
    shape_errors: list[str] = []
    used: set[str] = set()
    tie = bool(spec["tie_embeddings"])

    def add(target: str, source: str | None, transform: str = "identity", note: str = "") -> None:
        if target not in expected:
            shape_errors.append(f"adapter produced unexpected target '{target}'")
            return
        if source is None or source not in arrays:
            missing.append(target)
            mapping.append(
                {
                    "target": target,
                    "source": source,
                    "transform": transform,
                    "status": "missing",
                    "expected_shape": list(expected[target]),
                    "note": note,
                }
            )
            return
        arr = arrays[source]
        shape = [int(d) for d in arr.shape]
        status = "mapped"
        if tuple(shape) != tuple(expected[target]):
            status = "shape_error"
            shape_errors.append(
                f"{target}: source {source} shape {shape} != expected {list(expected[target])}"
            )
        if status == "mapped":
            canonical[target] = np.ascontiguousarray(arr)
            used.add(source)
        mapping.append(
            {
                "target": target,
                "source": source,
                "transform": transform,
                "status": status,
                "source_shape": shape,
                "expected_shape": list(expected[target]),
                "dtype": dtype_name(arr.dtype),
                "note": note,
            }
        )

    add("embed.weight", "model.decoder.embed_tokens.weight")
    add(
        "pos_embed.weight",
        "model.decoder.embed_positions.weight",
        note="OPT position table includes the +2 position offset",
    )
    add("ln_f.weight", "model.decoder.final_layer_norm.weight")
    add("ln_f.bias", "model.decoder.final_layer_norm.bias")
    embed = arrays.get("model.decoder.embed_tokens.weight")
    lm_head = arrays.get("lm_head.weight")
    if tie:
        if lm_head is not None:
            if embed is not None and lm_head.shape == embed.shape and not np.array_equal(lm_head, embed):
                unresolved.append(
                    "tied embed_tokens/lm_head tensors differ; no tied OPT graph is claimed"
                )
            add("lm_head.weight", "lm_head.weight", note="tied LM head")
        else:
            add(
                "lm_head.weight",
                "model.decoder.embed_tokens.weight",
                "tied_embedding",
                "LM head tied to the word embedding",
            )
    else:
        add("lm_head.weight", "lm_head.weight")

    for layer in range(int(spec["n_layers"])):
        src = f"model.decoder.layers.{layer}"
        dst = f"layers.{layer}"
        add(f"{dst}.ln1.weight", f"{src}.self_attn_layer_norm.weight")
        add(f"{dst}.ln1.bias", f"{src}.self_attn_layer_norm.bias")
        add(f"{dst}.q_proj.weight", f"{src}.self_attn.q_proj.weight")
        add(f"{dst}.q_proj.bias", f"{src}.self_attn.q_proj.bias")
        add(f"{dst}.k_proj.weight", f"{src}.self_attn.k_proj.weight")
        add(f"{dst}.k_proj.bias", f"{src}.self_attn.k_proj.bias")
        add(f"{dst}.v_proj.weight", f"{src}.self_attn.v_proj.weight")
        add(f"{dst}.v_proj.bias", f"{src}.self_attn.v_proj.bias")
        add(f"{dst}.attn_out.weight", f"{src}.self_attn.out_proj.weight")
        add(f"{dst}.attn_out.bias", f"{src}.self_attn.out_proj.bias")
        add(f"{dst}.ln2.weight", f"{src}.final_layer_norm.weight")
        add(f"{dst}.ln2.bias", f"{src}.final_layer_norm.bias")
        add(f"{dst}.mlp_up.weight", f"{src}.fc1.weight")
        add(f"{dst}.mlp_up.bias", f"{src}.fc1.bias")
        add(f"{dst}.mlp_down.weight", f"{src}.fc2.weight")
        add(f"{dst}.mlp_down.bias", f"{src}.fc2.bias")

    extra = sorted(set(arrays) - used)
    for tensor_name in extra:
        unresolved.append(f"source tensor '{tensor_name}' is not represented by the OPT graph")
    return canonical, mapping, sorted(set(missing)), extra, shape_errors, unresolved


def _execute_opt(
    definition: ArchitectureDefinition, package: WeightPackage, spec: Mapping[str, Any]
) -> dict[str, Any]:
    seq = min(4, int(spec["max_seq_len"]))
    tokens = np.arange(seq, dtype=np.int64).reshape(1, seq)
    inputs = {"tokens": tokens}
    try:
        result = execute(definition.graph, inputs, package, check_finite=True, validate_types=True)
        logits = result.output("logits")
        expected = (1, seq, int(spec["vocab_size"]))
        if tuple(logits.shape) != expected:
            return {
                "status": "failed",
                "message": f"logits shape {tuple(logits.shape)} != expected {expected}",
                "inputs": {k: list(v.shape) for k, v in inputs.items()},
            }
        return {
            "status": "passed",
            "message": "OPT forward pass produced finite logits with the expected shape",
            "inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in inputs.items()},
            "outputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in result.outputs.items()},
            "num_ops": result.num_ops,
            "sample_sequence_length": seq,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "message": str(exc),
            "inputs": {k: list(v.shape) for k, v in inputs.items()},
        }


# --------------------------------------------------------------------------- #
# GPT-NeoX adapter: partial RoPE, interleaved fused QKV, parallel residual
# --------------------------------------------------------------------------- #


def _reconstruct_neox(
    arrays: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    architecture: str | None,
    source_dtypes: Mapping[str, str] | None,
    name: str | None,
    source_hash: str | None,
    on_progress: Any | None = None,
) -> ReconstructionResult:
    progress: list[dict[str, Any]] = []
    unresolved: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []
    extra: list[str] = []
    shape_errors: list[str] = []
    reasons: list[str] = []

    def step(stage: str, status: str, message: str) -> None:
        progress.append({"stage": stage, "status": status, "message": message})
        if on_progress is not None:
            on_progress(stage, status, message)

    try:
        spec = _neox_spec(config, warnings)
    except ValueError as exc:
        step("config", "unsupported", str(exc))
        return ReconstructionResult(
            report=ReconstructionReport(
                status="unsupported",
                architecture=architecture,
                family="transformer_decoder",
                spec={},
                graph=None,
                graph_summary=None,
                tensor_mapping=[],
                missing=[],
                extra=sorted(arrays),
                shape_errors=[],
                unresolved=[str(exc)],
                warnings=warnings,
                validation=None,
                execution=None,
                source_tensor_count=len(arrays),
                mapped_tensor_count=0,
                progress=progress,
                reasons=[str(exc)],
            )
        )
    step("config", "complete", "GPT-NeoX configuration validated against the supported adapter")

    definition = _build_neox_definition(spec)
    step(
        "architecture",
        "complete",
        "explicit GPT-NeoX NIR graph with partial RoPE and parallel residual built",
    )
    canonical, mapping, missing, extra, shape_errors, unresolved = _neox_tensors(
        arrays, spec, definition, unresolved
    )
    step(
        "tensor_mapping",
        "complete" if not missing and not shape_errors else "failed",
        f"mapped {len(mapping)} tensors; {len(missing)} missing, {len(shape_errors)} shape errors, "
        f"{len(extra)} unmapped source tensors",
    )
    if missing or shape_errors:
        reasons.append(
            "all GPT-NeoX NIR parameters must be present with expected shapes; none are invented"
        )
        return ReconstructionResult(
            report=ReconstructionReport(
                status="unsupported",
                architecture=architecture,
                family="transformer_decoder",
                spec=spec,
                graph=None,
                graph_summary=None,
                tensor_mapping=mapping,
                missing=missing,
                extra=extra,
                shape_errors=shape_errors,
                unresolved=unresolved,
                warnings=warnings,
                validation=None,
                execution=None,
                source_tensor_count=len(arrays),
                mapped_tensor_count=len(canonical),
                progress=progress,
                reasons=reasons,
            )
        )

    package = WeightPackage.from_arrays(
        canonical,
        name=name or "gpt_neox_model",
        family="transformer_decoder",
        family_spec=spec,
        metadata={
            "reconstruction": {
                "architecture": architecture,
                "adapter": "gpt_neox",
                "source_hash": source_hash,
                "tensor_mapping": mapping,
                "graph": definition.graph.to_dict(),
            },
            "source_hash": source_hash,
        },
        source_dtypes={
            target: source_dtypes[item["source"]] if source_dtypes and item["source"] in source_dtypes else item["dtype"]
            for target, item in ((m["target"], m) for m in mapping)
        },
    )
    validation = definition.graph.validate(package)
    validation_doc = {
        "ok": validation.ok,
        "errors": list(validation.errors),
        "warnings": list(validation.warnings),
        "num_nodes": validation.num_nodes,
        "num_params": validation.num_params,
        "param_names": list(validation.param_names),
        "order": list(validation.order),
    }
    step(
        "graph_validation",
        "complete" if validation.ok else "failed",
        f"graph validation {'passed' if validation.ok else 'failed'} with {validation.num_nodes} nodes",
    )
    if not validation.ok:
        reasons.append(
            "GPT-NeoX graph validation failed; no validated artifact should be published"
        )
        return ReconstructionResult(
            report=ReconstructionReport(
                status="failed",
                architecture=architecture,
                family="transformer_decoder",
                spec=spec,
                graph=definition.graph.to_dict(),
                graph_summary=definition.graph.summary(),
                tensor_mapping=mapping,
                missing=missing,
                extra=extra,
                shape_errors=shape_errors,
                unresolved=unresolved,
                warnings=warnings,
                validation=validation_doc,
                execution=None,
                source_tensor_count=len(arrays),
                mapped_tensor_count=len(canonical),
                progress=progress,
                reasons=reasons,
            )
        )

    execution = _execute_neox(definition, package, spec)
    step(
        "forward_execution",
        execution.get("status", "failed"),
        execution.get("message", "forward pass completed"),
    )
    if extra:
        unresolved.append(
            "source tensors not represented by the GPT-NeoX graph: "
            + ", ".join(extra[:12])
        )
    status = "validated" if execution.get("status") == "passed" and not extra and not unresolved else "partial"
    if execution.get("status") != "passed":
        status = "failed"
        reasons.append(
            "GPT-NeoX forward execution failed; the graph is not marked validated"
        )
    elif status == "partial":
        reasons.append("GPT-NeoX graph executes, but some source tensors remain unresolved")
    else:
        reasons.append(
            "all GPT-NeoX tensors mapped, graph validated, and forward execution produced finite logits"
        )
    report = ReconstructionReport(
        status=status,
        architecture=architecture,
        family="transformer_decoder",
        spec=spec,
        graph=definition.graph.to_dict(),
        graph_summary=definition.graph.summary(),
        tensor_mapping=mapping,
        missing=missing,
        extra=extra,
        shape_errors=shape_errors,
        unresolved=unresolved,
        warnings=warnings,
        validation=validation_doc,
        execution=execution,
        source_tensor_count=len(arrays),
        mapped_tensor_count=len(canonical),
        progress=progress,
        reasons=reasons,
    )
    if status == "failed":
        return ReconstructionResult(report=report, package=None, definition=definition)
    return ReconstructionResult(report=report, package=package, definition=definition)


def _neox_spec(config: Mapping[str, Any], warnings: list[str]) -> dict[str, Any]:
    try:
        d_model = int(config["hidden_size"])
        n_heads = int(config["num_attention_heads"])
        n_layers = int(config["num_hidden_layers"])
        d_ff = int(config["intermediate_size"])
        vocab_size = int(config["vocab_size"])
        max_seq = int(config["max_position_embeddings"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"GPT-NeoX configuration is missing a required integer field: {exc}"
        ) from exc
    if min(d_model, n_heads, n_layers, d_ff, vocab_size, max_seq) <= 0:
        raise ValueError(
            "GPT-NeoX dimensions, layers, vocabulary and context length must be positive"
        )
    if d_model % n_heads != 0:
        raise ValueError(
            f"hidden_size={d_model} is not divisible by num_attention_heads={n_heads}"
        )
    head_dim = d_model // n_heads
    # GPTNeoXConfig defaults a missing partial-rotary factor to 0.25, not 1.0.
    rotary_pct = float(
        config.get("partial_rotary_factor")
        or config.get("rotary_pct")
        or 0.25
    )
    rotary_dim = int(head_dim * rotary_pct)
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(
            f"partial rotary width {rotary_dim} (from {rotary_pct} of head_dim {head_dim}) "
            "must be positive, even and <= head_dim"
        )
    theta = float(config.get("rope_theta") or config.get("rotary_emb_base") or 10000.0)
    if theta <= 0:
        raise ValueError("rope_theta must be positive")
    if config.get("rope_scaling") is not None:
        raise ValueError("rope_scaling is not implemented; no scaled-RoPE graph was guessed")
    activation = str(config.get("hidden_act") or "gelu_fast")
    if activation in ("gelu_fast", "gelu_new"):
        nir_activation = "gelu"
    elif activation == "gelu":
        nir_activation = "gelu_exact"
    else:
        raise ValueError(
            f"hidden_act='{activation}' is not supported; this adapter implements "
            "exact GELU and the tanh-approximated gelu_fast/gelu_new"
        )
    eps = float(config.get("layer_norm_eps") or 1e-5)
    if eps <= 0:
        raise ValueError("layer_norm_eps must be positive")
    if config.get("attention_dropout"):
        warnings.append("attention_dropout is a training-time setting; inference omits it")
    if config.get("hidden_dropout"):
        warnings.append("hidden_dropout is a training-time setting; inference omits it")
    if config.get("type_vocab_size") not in (None, 0, 1):
        warnings.append("type_vocab_size is present but this GPT-NeoX adapter does not use token types")
    tie = config.get("tie_word_embeddings")
    return {
        "architecture": "gpt_neox",
        "d_model": d_model,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq,
        "rotary_dim": rotary_dim,
        "rope_theta": theta,
        "layer_norm_eps": eps,
        "activation": nir_activation,
        "parallel_residual": bool(config.get("use_parallel_residual", True)),
        "attention_bias": bool(config.get("attention_bias", False)),
        "tie_embeddings": False if tie is None else bool(tie),
    }


def _build_neox_definition(spec: Mapping[str, Any]) -> ArchitectureDefinition:
    d = int(spec["d_model"])
    heads = int(spec["n_heads"])
    head_dim = int(spec["head_dim"])
    layers = int(spec["n_layers"])
    ffn = int(spec["d_ff"])
    vocab = int(spec["vocab_size"])
    rotary_dim = int(spec["rotary_dim"])
    theta = float(spec["rope_theta"])
    eps = float(spec["layer_norm_eps"])
    parallel = bool(spec["parallel_residual"])
    attention_bias = bool(spec["attention_bias"])
    g = Graph(
        name=f"gpt_neox_L{layers}",
        metadata={
            "family": "transformer_decoder",
            "architecture": "gpt_neox",
            "spec": dict(spec),
        },
    )
    g.add_input("tokens", TensorType.of(("B", "T"), "int64"))
    g.add_input("positions", TensorType.of(("B", "T"), "int64"))
    sh: dict[str, tuple[int, ...]] = {
        "embed.weight": (vocab, d),
        "lm_head.weight": (vocab, d),
        "norm.weight": (d,),
        "norm.bias": (d,),
    }
    cur = g.add_node(
        "embed", "embedding", {"x": Ref("input:tokens")}, {"weight": "param:embed.weight"}
    )

    for layer in range(layers):
        p = f"layers.{layer}"
        sh[f"{p}.ln1.weight"] = (d,)
        sh[f"{p}.ln1.bias"] = (d,)
        sh[f"{p}.ln2.weight"] = (d,)
        sh[f"{p}.ln2.bias"] = (d,)
        sh[f"{p}.qkv.weight"] = (3 * d, d)
        sh[f"{p}.attn_out.weight"] = (d, d)
        sh[f"{p}.mlp_up.weight"] = (ffn, d)
        sh[f"{p}.mlp_up.bias"] = (ffn,)
        sh[f"{p}.mlp_down.weight"] = (d, ffn)
        sh[f"{p}.mlp_down.bias"] = (d,)
        if attention_bias:
            sh[f"{p}.qkv.bias"] = (3 * d,)
            sh[f"{p}.attn_out.bias"] = (d,)

        ln1 = g.add_node(
            f"l{layer}_ln1",
            "layernorm",
            {"x": cur},
            {
                "weight": f"param:{p}.ln1.weight",
                "bias": f"param:{p}.ln1.bias",
                "eps": eps,
            },
        )
        qkv_params = {"weight": f"param:{p}.qkv.weight"}
        if attention_bias:
            qkv_params["bias"] = f"param:{p}.qkv.bias"
        qkv = g.add_node(f"l{layer}_qkv", "linear", {"x": ln1}, qkv_params)
        split = g.add_node(
            f"l{layer}_split",
            "split_qkv",
            {"x": qkv},
            {"n_heads": heads, "layout": "interleaved"},
        )
        q = g.add_node(
            f"l{layer}_q_rope",
            "rope",
            {"x": split, "positions": Ref("input:positions")},
            {"theta": theta, "rotary_dim": rotary_dim},
        )
        k = g.add_node(
            f"l{layer}_k_rope",
            "rope",
            {"x": Ref(f"l{layer}_split", 1), "positions": Ref("input:positions")},
            {"theta": theta, "rotary_dim": rotary_dim},
        )
        attn = g.add_node(
            f"l{layer}_attn",
            "attention",
            {"q": q, "k": k, "v": Ref(f"l{layer}_split", 2)},
            {"causal": True, "scale": "auto"},
        )
        merged = g.add_node(
            f"l{layer}_merge", "merge_heads", {"x": attn}, {"n_heads": heads}
        )
        attn_params = {"weight": f"param:{p}.attn_out.weight"}
        if attention_bias:
            attn_params["bias"] = f"param:{p}.attn_out.bias"
        attn_out = g.add_node(
            f"l{layer}_attn_out", "linear", {"x": merged}, attn_params
        )

        if parallel:
            mlp_input = g.add_node(
                f"l{layer}_ln2",
                "layernorm",
                {"x": cur},
                {
                    "weight": f"param:{p}.ln2.weight",
                    "bias": f"param:{p}.ln2.bias",
                    "eps": eps,
                },
            )
        else:
            res1 = g.add_node(f"l{layer}_res1", "add", {"a": cur, "b": attn_out})
            mlp_input = g.add_node(
                f"l{layer}_ln2",
                "layernorm",
                {"x": res1},
                {
                    "weight": f"param:{p}.ln2.weight",
                    "bias": f"param:{p}.ln2.bias",
                    "eps": eps,
                },
            )
        up = g.add_node(
            f"l{layer}_mlp_up",
            "linear",
            {"x": mlp_input},
            {"weight": f"param:{p}.mlp_up.weight", "bias": f"param:{p}.mlp_up.bias"},
        )
        act = g.add_node(f"l{layer}_act", str(spec["activation"]), {"x": up})
        down = g.add_node(
            f"l{layer}_mlp_down",
            "linear",
            {"x": act},
            {"weight": f"param:{p}.mlp_down.weight", "bias": f"param:{p}.mlp_down.bias"},
        )
        if parallel:
            res1 = g.add_node(f"l{layer}_res1", "add", {"a": cur, "b": attn_out})
        cur = g.add_node(f"l{layer}_res2", "add", {"a": res1, "b": down})

    final = g.add_node(
        "norm",
        "layernorm",
        {"x": cur},
        {"weight": "param:norm.weight", "bias": "param:norm.bias", "eps": eps},
    )
    logits = g.add_node(
        "lm_head",
        "linear",
        {"x": final},
        {"weight": "param:lm_head.weight"},
    )
    g.set_output("logits", logits)
    g.set_output("hidden", final)
    return ArchitectureDefinition(
        family="transformer_decoder",
        spec=dict(spec),
        graph=g,
        parameter_shapes=sh,
        default_inputs={"tokens": ("B", "T"), "positions": ("B", "T")},
        input_dtypes={"tokens": "int64", "positions": "int64"},
    )


def _neox_tensors(
    arrays: Mapping[str, np.ndarray],
    spec: Mapping[str, Any],
    definition: ArchitectureDefinition,
    unresolved: list[str],
) -> tuple[
    dict[str, np.ndarray],
    list[dict[str, Any]],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    expected = dict(definition.parameter_shapes)
    canonical: dict[str, np.ndarray] = {}
    mapping: list[dict[str, Any]] = []
    missing: list[str] = []
    shape_errors: list[str] = []
    used: set[str] = set()
    tie = bool(spec["tie_embeddings"])
    attention_bias = bool(spec["attention_bias"])

    def add(target: str, source: str | None, transform: str = "identity", note: str = "") -> None:
        if target not in expected:
            shape_errors.append(f"adapter produced unexpected target '{target}'")
            return
        if source is None or source not in arrays:
            missing.append(target)
            mapping.append(
                {
                    "target": target,
                    "source": source,
                    "transform": transform,
                    "status": "missing",
                    "expected_shape": list(expected[target]),
                    "note": note,
                }
            )
            return
        arr = arrays[source]
        shape = [int(d) for d in arr.shape]
        status = "mapped"
        if tuple(shape) != tuple(expected[target]):
            status = "shape_error"
            shape_errors.append(
                f"{target}: source {source} shape {shape} != expected {list(expected[target])}"
            )
        if status == "mapped":
            canonical[target] = np.ascontiguousarray(arr)
            used.add(source)
        mapping.append(
            {
                "target": target,
                "source": source,
                "transform": transform,
                "status": status,
                "source_shape": shape,
                "expected_shape": list(expected[target]),
                "dtype": dtype_name(arr.dtype),
                "note": note,
            }
        )

    add("embed.weight", "gpt_neox.embed_in.weight")
    add("norm.weight", "gpt_neox.final_layer_norm.weight")
    add("norm.bias", "gpt_neox.final_layer_norm.bias")
    embed = arrays.get("gpt_neox.embed_in.weight")
    lm_head = arrays.get("embed_out.weight")
    if tie:
        if lm_head is not None:
            if embed is not None and lm_head.shape == embed.shape and not np.array_equal(lm_head, embed):
                unresolved.append(
                    "tied embed_in/embed_out tensors differ; no tied GPT-NeoX graph is claimed"
                )
            add("lm_head.weight", "embed_out.weight", note="tied LM head")
        else:
            add(
                "lm_head.weight",
                "gpt_neox.embed_in.weight",
                "tied_embedding",
                "LM head tied to the word embedding",
            )
    else:
        add("lm_head.weight", "embed_out.weight")

    for layer in range(int(spec["n_layers"])):
        src = f"gpt_neox.layers.{layer}"
        dst = f"layers.{layer}"
        add(f"{dst}.ln1.weight", f"{src}.input_layernorm.weight")
        add(f"{dst}.ln1.bias", f"{src}.input_layernorm.bias")
        add(f"{dst}.ln2.weight", f"{src}.post_attention_layernorm.weight")
        add(f"{dst}.ln2.bias", f"{src}.post_attention_layernorm.bias")
        add(
            f"{dst}.qkv.weight",
            f"{src}.attention.query_key_value.weight",
            note="per-head-interleaved QKV layout",
        )
        add(f"{dst}.attn_out.weight", f"{src}.attention.dense.weight")
        add(f"{dst}.mlp_up.weight", f"{src}.mlp.dense_h_to_4h.weight")
        add(f"{dst}.mlp_up.bias", f"{src}.mlp.dense_h_to_4h.bias")
        add(f"{dst}.mlp_down.weight", f"{src}.mlp.dense_4h_to_h.weight")
        add(f"{dst}.mlp_down.bias", f"{src}.mlp.dense_4h_to_h.bias")
        if attention_bias:
            add(f"{dst}.qkv.bias", f"{src}.attention.query_key_value.bias")
            add(f"{dst}.attn_out.bias", f"{src}.attention.dense.bias")

    extra = sorted(set(arrays) - used)
    for tensor_name in extra:
        unresolved.append(
            f"source tensor '{tensor_name}' is not represented by the GPT-NeoX graph"
        )
    return canonical, mapping, sorted(set(missing)), extra, shape_errors, unresolved


def _execute_neox(
    definition: ArchitectureDefinition, package: WeightPackage, spec: Mapping[str, Any]
) -> dict[str, Any]:
    seq = min(4, int(spec["max_seq_len"]))
    tokens = np.arange(seq, dtype=np.int64).reshape(1, seq)
    positions = np.arange(seq, dtype=np.int64).reshape(1, seq)
    inputs = {"tokens": tokens, "positions": positions}
    try:
        result = execute(definition.graph, inputs, package, check_finite=True, validate_types=True)
        logits = result.output("logits")
        expected = (1, seq, int(spec["vocab_size"]))
        if tuple(logits.shape) != expected:
            return {
                "status": "failed",
                "message": f"logits shape {tuple(logits.shape)} != expected {expected}",
                "inputs": {k: list(v.shape) for k, v in inputs.items()},
            }
        return {
            "status": "passed",
            "message": "GPT-NeoX forward pass produced finite logits with the expected shape",
            "inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in inputs.items()},
            "outputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in result.outputs.items()},
            "num_ops": result.num_ops,
            "sample_sequence_length": seq,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "message": str(exc),
            "inputs": {k: list(v.shape) for k, v in inputs.items()},
        }
