"""Architecture families: real graph builders plus analytic weight construction.

A "family" turns a small declarative spec into

  * a validated NIR graph, and
  * a concrete parameter binding plan (tensor names -> shapes),

so that weight engineering and architecture engineering can be exercised end to
end without any external checkpoint.

Weight initialization here is *structured random initialization* or *analytic
construction* — never learned. Every generated tensor is labelled with its true
origin (``randomly_initialized`` or ``analytically_derived``) in the package
metadata, and the spec records the RNG seed used.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

import numpy as np

from uzen.core.nir.graph import Graph
from uzen.core.nir.types import Ref, TensorType
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import InvalidOperationParameters
from uzen.policies.strict_plus.effects import Origin
from uzen.util.linalg import orthogonal_matrix
from uzen.util.rng import generator


@dataclass
class ArchitectureDefinition:
    family: str
    spec: dict[str, Any]
    graph: Graph
    parameter_shapes: dict[str, tuple[int, ...]]
    default_inputs: dict[str, tuple[int, ...]]
    input_dtypes: dict[str, str] = field(default_factory=dict)

    def build_package(self, *, seed: int | None = 0, dtype: str = "float32", name: str | None = None) -> WeightPackage:
        arrays, origins = initialize_weights(self.family, self.spec, self.parameter_shapes, seed=seed, dtype=dtype)
        pkg = WeightPackage.from_arrays(
            arrays,
            name=name or f"{self.family}",
            family=self.family,
            origin=Origin.RANDOMLY_INITIALIZED,
            family_spec=dict(self.spec),
            metadata={
                "initialization": "structured/analytic; NOT learned",
                "seed": seed,
                "parameter_origins": origins,
            },
        )
        # Per-tensor origin classification overrides the blanket label. The package
        # is immutable, so build a new one instead of assigning into ``metas``.
        metas = {
            name: replace(
                pkg.metas[name],
                origin=Origin(origins.get(name, Origin.RANDOMLY_INITIALIZED.value)),
            )
            for name in pkg.names()
        }
        return WeightPackage(
            tensors=dict(pkg.tensors),
            metas=metas,
            name=pkg.name,
            family=pkg.family,
            family_spec=dict(pkg.family_spec),
            metadata=dict(pkg.metadata),
            source_artifact_id=pkg.source_artifact_id,
        )

    def sample_inputs(self, batch: int = 2, seq: int = 8, seed: int = 0) -> dict[str, np.ndarray]:
        rng = generator(seed, "sample_inputs")
        out: dict[str, np.ndarray] = {}
        for iname, shape in self.default_inputs.items():
            if iname == "tokens" or iname.startswith("tokens_") or iname == "decoder_tokens":
                vocab = int(self.spec.get("vocab_size", 32))
                out[iname] = rng.integers(0, vocab, size=(batch, seq)).astype(np.int64)
            elif iname == "adj":
                size = tuple(batch if d == "B" else seq for d in shape)
                a = (rng.random(size) < 0.4).astype(np.float32)
                out[iname] = np.maximum(a, np.swapaxes(a, -1, -2))
            elif "image" in iname:
                out[iname] = rng.standard_normal((batch,) + tuple(shape[1:])).astype(np.float32)
            else:
                resolved = tuple(batch if isinstance(d, str) and d == "B" else seq if isinstance(d, str) else d for d in shape)
                out[iname] = rng.standard_normal(resolved).astype(np.float32)
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "spec": self.spec,
            "graph": self.graph.summary(),
            "num_parameters": len(self.parameter_shapes),
            "parameter_numel": int(
                sum(int(np.prod(shape)) for shape in self.parameter_shapes.values())
            ),
        }


# --------------------------------------------------------------------------- #
# initializers (analytic / structured random, explicitly labelled)
# --------------------------------------------------------------------------- #


def _orthogonal_rows(out_features: int, in_features: int, seed: int, gain: float = 1.0) -> np.ndarray:
    """Semi-orthogonal initialization scaled by ``gain / sqrt(in_features)``."""
    if out_features >= in_features:
        q = orthogonal_matrix(out_features, in_features, seed=seed, method="qr")
        return (q * (gain / np.sqrt(in_features))).astype(np.float32)
    # Wide matrix: build orthogonal on the transposed side, then transpose.
    q = orthogonal_matrix(in_features, out_features, seed=seed, method="qr")
    return (q.T * (gain / np.sqrt(in_features))).astype(np.float32)


def initialize_weights(
    family: str,
    spec: Mapping[str, Any],
    shapes: Mapping[str, tuple[int, ...]],
    *,
    seed: int | None = 0,
    dtype: str = "float32",
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Produce real initial tensors with explicit origin classification."""
    rng = generator(seed, "init", family)
    arrays: dict[str, np.ndarray] = {}
    origins: dict[str, str] = {}
    for index, (name, shape) in enumerate(sorted(shapes.items())):
        shape = tuple(int(d) for d in shape)
        low = name.lower()
        numel = int(np.prod(shape)) if shape else 1
        if low.endswith("bias") or "_bias" in low:
            arrays[name] = np.zeros(shape, dtype=np.float32)
            origins[name] = Origin.ANALYTICALLY_DERIVED.value
            continue
        if len(shape) == 2 and (low.endswith(".a") or low.endswith("_a")):
            # State-space decay A = -exp(U[log 1e-3, log 0.1]): stable by construction.
            decay = np.exp(rng.uniform(np.log(1e-3), np.log(0.1), size=shape))
            arrays[name] = (-decay).astype(np.float32)
            origins[name] = Origin.ANALYTICALLY_DERIVED.value
            continue
        if len(shape) == 1 and low.endswith(".d"):
            # Skip-connection gain for state-space layers.
            arrays[name] = np.ones(shape, dtype=np.float32)
            origins[name] = Origin.ANALYTICALLY_DERIVED.value
            continue
        if low.endswith("norm.weight") or low.endswith("ln.weight") or low.endswith("ln1.weight") or low.endswith("ln2.weight") or "layernorm.weight" in low or "rmsnorm.weight" in low:
            arrays[name] = np.ones(shape, dtype=np.float32)
            origins[name] = Origin.ANALYTICALLY_DERIVED.value
            continue
        if len(shape) == 1:
            arrays[name] = rng.standard_normal(shape).astype(np.float32) * 0.02
            origins[name] = Origin.RANDOMLY_INITIALIZED.value
            continue
        if len(shape) == 2:
            out_f, in_f = shape
            if "embed" in low or "lm_head" in low:
                values = rng.standard_normal(shape) * (1.0 / np.sqrt(max(1, in_f)))
                arrays[name] = values.astype(np.float32)
                origins[name] = Origin.RANDOMLY_INITIALIZED.value
                continue
            gain = 1.0
            if "qkv" in low or "up" in low or "fc1" in low:
                gain = 1.0
            elif "down" in low or "out" in low or "fc2" in low or "proj" in low:
                gain = 1.0
            arrays[name] = _orthogonal_rows(out_f, in_f, seed=seed + index if isinstance(seed, int) else index, gain=gain)
            origins[name] = Origin.RANDOMLY_INITIALIZED.value
            continue
        if len(shape) == 4:
            o, c, kh, kw = shape
            flat = _orthogonal_rows(o, c * kh * kw, seed=seed + index if isinstance(seed, int) else index)
            arrays[name] = flat.reshape(shape)
            origins[name] = Origin.RANDOMLY_INITIALIZED.value
            continue
        if len(shape) == 3 and (
            "expert" in low or low.endswith("up.weight") or low.endswith("down.weight")
        ):
            experts, out_f, in_f = shape
            base_seed = seed if isinstance(seed, int) else 0
            arrays[name] = np.stack(
                [
                    _orthogonal_rows(out_f, in_f, seed=base_seed + index * 97 + e)
                    for e in range(experts)
                ]
            ).astype(np.float32)
            origins[name] = Origin.RANDOMLY_INITIALIZED.value
            continue
        if len(shape) == 3:
            arrays[name] = (rng.standard_normal(shape) * 0.02).astype(np.float32)
            origins[name] = Origin.RANDOMLY_INITIALIZED.value
            continue
        raise InvalidOperationParameters(f"cannot initialize tensor '{name}' with shape {shape}")
    if dtype != "float32":
        arrays = {k: v.astype(dtype) for k, v in arrays.items()}
    return arrays, origins


# --------------------------------------------------------------------------- #
# family builders
# --------------------------------------------------------------------------- #


class FamilyBuilder:
    family: str = "generic"

    def build(self) -> ArchitectureDefinition:  # pragma: no cover - interface
        raise NotImplementedError

    def _require(self, spec: Mapping[str, Any], key: str, default: Any = None) -> Any:
        value = spec.get(key, default)
        if value is None:
            raise InvalidOperationParameters(f"family '{self.family}' requires spec field '{key}'")
        return value


@dataclass
class TransformerBuilder(FamilyBuilder):
    family: str = "transformer_decoder"

    def build(self) -> ArchitectureDefinition:
        spec = self.spec
        d_model = int(self._require(spec, "d_model"))
        n_heads = int(self._require(spec, "n_heads", 4))
        n_layers = int(self._require(spec, "n_layers", 2))
        d_ff = int(self._require(spec, "d_ff", 4 * d_model))
        vocab = int(self._require(spec, "vocab_size", 64))
        max_seq = int(self._require(spec, "max_seq_len", 32))
        norm = str(spec.get("norm", "layernorm"))
        activation = str(spec.get("activation", "gelu"))
        tie = bool(spec.get("tie_embeddings", False))
        if d_model % n_heads != 0:
            raise InvalidOperationParameters(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        if norm not in ("layernorm", "rmsnorm"):
            raise InvalidOperationParameters(f"unsupported norm '{norm}'")
        if activation not in ("gelu", "gelu_exact", "relu", "silu"):
            raise InvalidOperationParameters(f"unsupported activation '{activation}'")

        spec_clean = {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "vocab_size": vocab,
            "max_seq_len": max_seq,
            "norm": norm,
            "activation": activation,
            "tie_embeddings": tie,
        }
        g = Graph(name=f"transformer_decoder_L{n_layers}", metadata={"family": self.family, "spec": spec_clean})
        g.add_input("tokens", TensorType.of(("B", "T"), "int64"))
        sh: dict[str, tuple[int, ...]] = {}

        sh["embed.weight"] = (vocab, d_model)
        sh["pos_embed.weight"] = (max_seq, d_model)
        ref = g.add_node("embed", "embedding", {"x": Ref("input:tokens")}, {"weight": "param:embed.weight"})
        cur = g.add_node("pos", "pos_embedding", {"x": ref}, {"weight": "param:pos_embed.weight"})
        add_op = "add"
        ln_op = "layernorm" if norm == "layernorm" else "rmsnorm"

        for layer in range(n_layers):
            p = f"layers.{layer}"
            sh[f"{p}.ln1.weight"] = (d_model,)
            sh[f"{p}.qkv.weight"] = (3 * d_model, d_model)
            sh[f"{p}.qkv.bias"] = (3 * d_model,)
            sh[f"{p}.attn_out.weight"] = (d_model, d_model)
            sh[f"{p}.attn_out.bias"] = (d_model,)
            sh[f"{p}.ln2.weight"] = (d_model,)
            sh[f"{p}.mlp_up.weight"] = (d_ff, d_model)
            sh[f"{p}.mlp_up.bias"] = (d_ff,)
            sh[f"{p}.mlp_down.weight"] = (d_model, d_ff)
            sh[f"{p}.mlp_down.bias"] = (d_model,)
            if norm == "layernorm":
                sh[f"{p}.ln1.bias"] = (d_model,)
                sh[f"{p}.ln2.bias"] = (d_model,)
                ln1_params = {"weight": f"param:{p}.ln1.weight", "bias": f"param:{p}.ln1.bias", "eps": 1e-5}
                ln2_params = {"weight": f"param:{p}.ln2.weight", "bias": f"param:{p}.ln2.bias", "eps": 1e-5}
            else:
                ln1_params = {"weight": f"param:{p}.ln1.weight", "eps": 1e-6}
                ln2_params = {"weight": f"param:{p}.ln2.weight", "eps": 1e-6}

            ln1 = g.add_node(f"l{layer}_ln1", ln_op, {"x": cur}, ln1_params)
            qkv = g.add_node(
                f"l{layer}_qkv",
                "linear",
                {"x": ln1},
                {"weight": f"param:{p}.qkv.weight", "bias": f"param:{p}.qkv.bias"},
            )
            split = g.add_node(
                f"l{layer}_split", "split_qkv", {"x": qkv}, {"n_heads": n_heads}
            )
            attn = g.add_node(
                f"l{layer}_attn",
                "attention",
                {
                    "q": split,
                    "k": Ref(f"l{layer}_split", 1),
                    "v": Ref(f"l{layer}_split", 2),
                },
                {"causal": True, "scale": "auto"},
            )
            merged = g.add_node(f"l{layer}_merge", "merge_heads", {"x": attn}, {"n_heads": n_heads})
            proj = g.add_node(
                f"l{layer}_attn_out",
                "linear",
                {"x": merged},
                {"weight": f"param:{p}.attn_out.weight", "bias": f"param:{p}.attn_out.bias"},
            )
            cur = g.add_node(f"l{layer}_res1", add_op, {"a": cur, "b": proj})
            ln2 = g.add_node(f"l{layer}_ln2", ln_op, {"x": cur}, ln2_params)
            up = g.add_node(
                f"l{layer}_mlp_up",
                "linear",
                {"x": ln2},
                {"weight": f"param:{p}.mlp_up.weight", "bias": f"param:{p}.mlp_up.bias"},
            )
            act = g.add_node(f"l{layer}_act", activation, {"x": up})
            down = g.add_node(
                f"l{layer}_mlp_down",
                "linear",
                {"x": act},
                {"weight": f"param:{p}.mlp_down.weight", "bias": f"param:{p}.mlp_down.bias"},
            )
            cur = g.add_node(f"l{layer}_res2", add_op, {"a": cur, "b": down})

        if norm == "layernorm":
            sh["ln_f.weight"] = (d_model,)
            sh["ln_f.bias"] = (d_model,)
            final = g.add_node("ln_f", "layernorm", {"x": cur}, {"weight": "param:ln_f.weight", "bias": "param:ln_f.bias", "eps": 1e-5})
        else:
            sh["ln_f.weight"] = (d_model,)
            final = g.add_node("ln_f", "rmsnorm", {"x": cur}, {"weight": "param:ln_f.weight", "eps": 1e-6})
        if not tie:
            sh["lm_head.weight"] = (vocab, d_model)
            sh["lm_head.bias"] = (vocab,)
            logits = g.add_node(
                "lm_head",
                "linear",
                {"x": final},
                {"weight": "param:lm_head.weight", "bias": "param:lm_head.bias"},
            )
        else:
            sh["lm_head.weight"] = (vocab, d_model)
            logits = g.add_node(
                "lm_head", "linear", {"x": final}, {"weight": "param:lm_head.weight"}
            )
        g.set_output("logits", logits)
        g.set_output("hidden", final)
        definition = ArchitectureDefinition(
            family=self.family,
            spec=spec_clean,
            graph=g,
            parameter_shapes=sh,
            default_inputs={"tokens": ("B", "T")},
            input_dtypes={"tokens": "int64"},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition

    spec: dict[str, Any] = field(default_factory=dict)


@dataclass
class MLPBuilder(FamilyBuilder):
    family: str = "mlp"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        spec = self.spec
        input_dim = int(self._require(spec, "input_dim"))
        hidden = [int(h) for h in (spec.get("hidden_dims") or [64])]
        output_dim = int(self._require(spec, "output_dim"))
        activation = str(spec.get("activation", "relu"))
        spec_clean = {
            "input_dim": input_dim,
            "hidden_dims": hidden,
            "output_dim": output_dim,
            "activation": activation,
        }
        g = Graph(name="mlp", metadata={"family": self.family, "spec": spec_clean})
        g.add_input("x", TensorType.of(("B", input_dim), "float32"))
        sh: dict[str, tuple[int, ...]] = {}
        cur = Ref("input:x")
        prev = input_dim
        for i, width in enumerate(hidden):
            sh[f"fc{i}.weight"] = (width, prev)
            sh[f"fc{i}.bias"] = (width,)
            cur = g.add_node(
                f"fc{i}",
                "linear",
                {"x": cur},
                {"weight": f"param:fc{i}.weight", "bias": f"param:fc{i}.bias"},
            )
            cur = g.add_node(f"act{i}", activation, {"x": cur})
            prev = width
        sh["head.weight"] = (output_dim, prev)
        sh["head.bias"] = (output_dim,)
        out = g.add_node(
            "head",
            "linear",
            {"x": cur},
            {"weight": "param:head.weight", "bias": "param:head.bias"},
        )
        g.set_output("logits", out)
        g.set_output("hidden", cur)
        definition = ArchitectureDefinition(
            family=self.family,
            spec=spec_clean,
            graph=g,
            parameter_shapes=sh,
            default_inputs={"x": ("B", input_dim)},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class CNNBuilder(FamilyBuilder):
    family: str = "cnn"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        spec = self.spec
        in_channels = int(self._require(spec, "in_channels"))
        height = int(self._require(spec, "height"))
        width = int(self._require(spec, "width"))
        channels = [int(c) for c in (spec.get("channels") or [8])]
        kernel = int(spec.get("kernel", 3))
        stride = int(spec.get("stride", 1))
        padding = int(spec.get("padding", 1))
        num_classes = int(self._require(spec, "num_classes"))
        pool = int(spec.get("pool", 2))
        spec_clean = {
            "in_channels": in_channels,
            "height": height,
            "width": width,
            "channels": channels,
            "kernel": kernel,
            "stride": stride,
            "padding": padding,
            "num_classes": num_classes,
            "pool": pool,
        }
        g = Graph(name="cnn", metadata={"family": self.family, "spec": spec_clean})
        g.add_input("image", TensorType.of(("B", in_channels, height, width), "float32"))
        sh: dict[str, tuple[int, ...]] = {}
        cur = Ref("input:image")
        prev_c = in_channels
        h, w = height, width
        for i, out_c in enumerate(channels):
            sh[f"conv{i}.weight"] = (out_c, prev_c, kernel, kernel)
            sh[f"conv{i}.bias"] = (out_c,)
            cur = g.add_node(
                f"conv{i}",
                "conv2d",
                {"x": cur},
                {
                    "weight": f"param:conv{i}.weight",
                    "bias": f"param:conv{i}.bias",
                    "stride": stride,
                    "padding": padding,
                },
            )
            cur = g.add_node(f"relu{i}", "relu", {"x": cur})
            cur = g.add_node(f"pool{i}", "maxpool2d", {"x": cur}, {"kernel": pool, "stride": pool})
            h = (h + 2 * padding - kernel) // stride + 1
            h = (h - pool) // pool + 1
            w = (w + 2 * padding - kernel) // stride + 1
            w = (w - pool) // pool + 1
            if h <= 0 or w <= 0:
                raise InvalidOperationParameters(
                    f"cnn spatial dims collapsed to {h}x{w} after layer {i}; reduce pooling/depth"
                )
            prev_c = out_c
        flat = g.add_node("flatten", "flatten", {"x": cur}, {"start_dim": 1})
        sh["head.weight"] = (num_classes, prev_c * h * w)
        sh["head.bias"] = (num_classes,)
        out = g.add_node(
            "head",
            "linear",
            {"x": flat},
            {"weight": "param:head.weight", "bias": "param:head.bias"},
        )
        g.set_output("logits", out)
        g.set_output("features", flat)
        definition = ArchitectureDefinition(
            family=self.family,
            spec=spec_clean,
            graph=g,
            parameter_shapes=sh,
            default_inputs={"image": ("B", in_channels, height, width)},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class RecurrentBuilder(FamilyBuilder):
    family: str = "recurrent"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        spec = self.spec
        kind = str(spec.get("cell", "lstm"))
        if kind not in ("rnn", "lstm", "gru"):
            raise InvalidOperationParameters(f"unsupported recurrent cell '{kind}'")
        vocab = int(self._require(spec, "vocab_size"))
        d_model = int(self._require(spec, "d_model", 16))
        hidden = int(self._require(spec, "hidden_size"))
        num_classes = int(self._require(spec, "num_classes"))
        gates = 4 if kind == "lstm" else 3 if kind == "gru" else 1
        spec_clean = {
            "cell": kind,
            "vocab_size": vocab,
            "d_model": d_model,
            "hidden_size": hidden,
            "num_classes": num_classes,
        }
        g = Graph(name=f"{kind}_classifier", metadata={"family": self.family, "spec": spec_clean})
        g.add_input("tokens", TensorType.of(("B", "T"), "int64"))
        sh: dict[str, tuple[int, ...]] = {}
        sh["embed.weight"] = (vocab, d_model)
        emb = g.add_node("embed", "embedding", {"x": Ref("input:tokens")}, {"weight": "param:embed.weight"})
        sh["rnn.weight_ih"] = (gates * hidden, d_model)
        sh["rnn.weight_hh"] = (gates * hidden, hidden)
        sh["rnn.bias"] = (gates * hidden,)
        rec = g.add_node(
            "rnn",
            kind,
            {"x": emb},
            {
                "weight_ih": "param:rnn.weight_ih",
                "weight_hh": "param:rnn.weight_hh",
                "bias": "param:rnn.bias",
            },
        )
        last = g.add_node("last", "slice", {"x": rec}, {"axis": 1, "start": -1})
        squeezed = g.add_node("squeeze", "reshape", {"x": last}, {"shape": ["B", hidden]})
        sh["head.weight"] = (num_classes, hidden)
        sh["head.bias"] = (num_classes,)
        out = g.add_node(
            "head",
            "linear",
            {"x": squeezed},
            {"weight": "param:head.weight", "bias": "param:head.bias"},
        )
        g.set_output("logits", out)
        g.set_output("hidden", squeezed)
        definition = ArchitectureDefinition(
            family=self.family,
            spec=spec_clean,
            graph=g,
            parameter_shapes=sh,
            default_inputs={"tokens": ("B", "T")},
            input_dtypes={"tokens": "int64"},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class AutoencoderBuilder(FamilyBuilder):
    family: str = "autoencoder"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        spec = self.spec
        input_dim = int(self._require(spec, "input_dim"))
        latent = int(self._require(spec, "latent_dim"))
        hidden = int(spec.get("hidden_dim", max(latent * 2, 16)))
        spec_clean = {"input_dim": input_dim, "latent_dim": latent, "hidden_dim": hidden}
        g = Graph(name="autoencoder", metadata={"family": self.family, "spec": spec_clean})
        g.add_input("x", TensorType.of(("B", input_dim), "float32"))
        sh = {
            "enc.weight": (hidden, input_dim),
            "enc.bias": (hidden,),
            "mu.weight": (latent, hidden),
            "mu.bias": (latent,),
            "dec.weight": (input_dim, latent),
            "dec.bias": (input_dim,),
        }
        enc = g.add_node("enc", "linear", {"x": Ref("input:x")}, {"weight": "param:enc.weight", "bias": "param:enc.bias"})
        act = g.add_node("enc_act", "tanh", {"x": enc})
        mu = g.add_node("mu", "linear", {"x": act}, {"weight": "param:mu.weight", "bias": "param:mu.bias"})
        dec = g.add_node("dec", "linear", {"x": mu}, {"weight": "param:dec.weight", "bias": "param:dec.bias"})
        g.set_output("reconstruction", dec)
        g.set_output("latent", mu)
        definition = ArchitectureDefinition(
            family=self.family,
            spec=spec_clean,
            graph=g,
            parameter_shapes=sh,
            default_inputs={"x": ("B", input_dim)},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class MoEBuilder(FamilyBuilder):
    family: str = "moe"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        spec = self.spec
        vocab = int(self._require(spec, "vocab_size"))
        d_model = int(self._require(spec, "d_model", 16))
        n_experts = int(self._require(spec, "n_experts", 4))
        top_k = int(self._require(spec, "top_k", 2))
        d_ff = int(self._require(spec, "d_ff", 2 * d_model))
        num_classes = int(self._require(spec, "num_classes"))
        activation = str(spec.get("activation", "gelu"))
        if n_experts < 1:
            raise InvalidOperationParameters("moe requires n_experts >= 1")
        if not 1 <= top_k <= n_experts:
            raise InvalidOperationParameters(f"moe top_k must be in [1, {n_experts}], got {top_k}")
        if activation not in ("gelu", "silu", "relu", "tanh"):
            raise InvalidOperationParameters(f"unsupported moe activation '{activation}'")
        spec_clean = {
            "vocab_size": vocab,
            "d_model": d_model,
            "n_experts": n_experts,
            "top_k": top_k,
            "d_ff": d_ff,
            "num_classes": num_classes,
            "activation": activation,
        }
        g = Graph(name="moe_classifier", metadata={"family": self.family, "spec": spec_clean})
        g.add_input("tokens", TensorType.of(("B", "T"), "int64"))
        sh: dict[str, tuple[int, ...]] = {
            "embed.weight": (vocab, d_model),
            "moe.router.weight": (n_experts, d_model),
            "moe.up.weight": (n_experts, d_ff, d_model),
            "moe.up.bias": (n_experts, d_ff),
            "moe.down.weight": (n_experts, d_model, d_ff),
            "moe.down.bias": (n_experts, d_model),
            "head.weight": (num_classes, d_model),
            "head.bias": (num_classes,),
        }
        emb = g.add_node(
            "embed", "embedding", {"x": Ref("input:tokens")}, {"weight": "param:embed.weight"}
        )
        routed = g.add_node(
            "moe",
            "moe",
            {"x": emb},
            {
                "router": "param:moe.router.weight",
                "up_weight": "param:moe.up.weight",
                "up_bias": "param:moe.up.bias",
                "down_weight": "param:moe.down.weight",
                "down_bias": "param:moe.down.bias",
                "top_k": top_k,
                "activation": activation,
            },
        )
        pooled = g.add_node("pool", "mean", {"x": routed}, {"axis": 1})
        logits = g.add_node(
            "head",
            "linear",
            {"x": pooled},
            {"weight": "param:head.weight", "bias": "param:head.bias"},
        )
        g.set_output("logits", logits)
        g.set_output("hidden", pooled)
        definition = ArchitectureDefinition(
            family=self.family,
            spec=spec_clean,
            graph=g,
            parameter_shapes=sh,
            default_inputs={"tokens": ("B", "T")},
            input_dtypes={"tokens": "int64"},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class StateSpaceBuilder(FamilyBuilder):
    family: str = "state_space"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        spec = self.spec
        vocab = int(self._require(spec, "vocab_size"))
        d_model = int(self._require(spec, "d_model", 16))
        state_dim = int(self._require(spec, "state_dim", 8))
        num_classes = int(self._require(spec, "num_classes"))
        if state_dim < 1:
            raise InvalidOperationParameters("state_space requires state_dim >= 1")
        spec_clean = {
            "vocab_size": vocab,
            "d_model": d_model,
            "state_dim": state_dim,
            "num_classes": num_classes,
        }
        g = Graph(name="state_space_classifier", metadata={"family": self.family, "spec": spec_clean})
        g.add_input("tokens", TensorType.of(("B", "T"), "int64"))
        sh: dict[str, tuple[int, ...]] = {
            "embed.weight": (vocab, d_model),
            "ssm.A": (d_model, state_dim),
            "ssm.B": (d_model, state_dim),
            "ssm.C": (d_model, state_dim),
            "ssm.D": (d_model,),
            "head.weight": (num_classes, d_model),
            "head.bias": (num_classes,),
        }
        emb = g.add_node(
            "embed", "embedding", {"x": Ref("input:tokens")}, {"weight": "param:embed.weight"}
        )
        mixed = g.add_node(
            "ssm",
            "ssm",
            {"x": emb},
            {
                "A": "param:ssm.A",
                "B": "param:ssm.B",
                "C": "param:ssm.C",
                "D": "param:ssm.D",
            },
        )
        pooled = g.add_node("pool", "mean", {"x": mixed}, {"axis": 1})
        logits = g.add_node(
            "head",
            "linear",
            {"x": pooled},
            {"weight": "param:head.weight", "bias": "param:head.bias"},
        )
        g.set_output("logits", logits)
        g.set_output("hidden", pooled)
        definition = ArchitectureDefinition(
            family=self.family,
            spec=spec_clean,
            graph=g,
            parameter_shapes=sh,
            default_inputs={"tokens": ("B", "T")},
            input_dtypes={"tokens": "int64"},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class GNNBuilder(FamilyBuilder):
    family: str = "gnn"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        spec = self.spec
        in_features = int(self._require(spec, "in_features"))
        hidden = int(self._require(spec, "hidden_dim", 16))
        num_classes = int(self._require(spec, "num_classes"))
        n_layers = int(self._require(spec, "n_layers", 2))
        if n_layers < 1:
            raise InvalidOperationParameters("gnn requires n_layers >= 1")
        spec_clean = {
            "in_features": in_features,
            "hidden_dim": hidden,
            "num_classes": num_classes,
            "n_layers": n_layers,
        }
        g = Graph(name=f"gcn_classifier_L{n_layers}", metadata={"family": self.family, "spec": spec_clean})
        g.add_input("x", TensorType.of(("B", "N", in_features), "float32"))
        g.add_input("adj", TensorType.of(("B", "N", "N"), "float32"))
        sh: dict[str, tuple[int, ...]] = {}
        cur = Ref("input:x")
        prev = in_features
        for layer in range(n_layers):
            sh[f"gcn{layer}.weight"] = (hidden, prev)
            sh[f"gcn{layer}.bias"] = (hidden,)
            cur = g.add_node(
                f"gcn{layer}",
                "gcn",
                {"x": cur, "adj": Ref("input:adj")},
                {"weight": f"param:gcn{layer}.weight", "bias": f"param:gcn{layer}.bias"},
            )
            cur = g.add_node(f"act{layer}", "relu", {"x": cur})
            prev = hidden
        pooled = g.add_node("pool", "mean", {"x": cur}, {"axis": 1})
        sh["head.weight"] = (num_classes, hidden)
        sh["head.bias"] = (num_classes,)
        logits = g.add_node(
            "head",
            "linear",
            {"x": pooled},
            {"weight": "param:head.weight", "bias": "param:head.bias"},
        )
        g.set_output("logits", logits)
        g.set_output("hidden", pooled)
        g.set_output("node_embeddings", cur)
        definition = ArchitectureDefinition(
            family=self.family,
            spec=spec_clean,
            graph=g,
            parameter_shapes=sh,
            default_inputs={"x": ("B", "N", in_features), "adj": ("B", "N", "N")},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class VAEBuilder(FamilyBuilder):
    family: str = "vae"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        spec = self.spec
        input_dim = int(self._require(spec, "input_dim"))
        latent = int(self._require(spec, "latent_dim"))
        hidden = int(self._require(spec, "hidden_dim", max(2 * latent, 16)))
        sample_seed = int(spec.get("sample_seed", 0))
        spec_clean = {
            "input_dim": input_dim,
            "latent_dim": latent,
            "hidden_dim": hidden,
            "sample_seed": sample_seed,
        }
        g = Graph(name="variational_autoencoder", metadata={"family": self.family, "spec": spec_clean})
        g.add_input("x", TensorType.of(("B", input_dim), "float32"))
        sh: dict[str, tuple[int, ...]] = {
            "enc.weight": (hidden, input_dim),
            "enc.bias": (hidden,),
            "mu.weight": (latent, hidden),
            "mu.bias": (latent,),
            "logvar.weight": (latent, hidden),
            "logvar.bias": (latent,),
            "dec1.weight": (hidden, latent),
            "dec1.bias": (hidden,),
            "dec2.weight": (input_dim, hidden),
            "dec2.bias": (input_dim,),
        }
        enc = g.add_node(
            "enc", "linear", {"x": Ref("input:x")}, {"weight": "param:enc.weight", "bias": "param:enc.bias"}
        )
        act = g.add_node("enc_act", "tanh", {"x": enc})
        mu = g.add_node(
            "mu", "linear", {"x": act}, {"weight": "param:mu.weight", "bias": "param:mu.bias"}
        )
        logvar = g.add_node(
            "logvar",
            "linear",
            {"x": act},
            {"weight": "param:logvar.weight", "bias": "param:logvar.bias"},
        )
        z = g.add_node(
            "z",
            "sample_normal",
            {"mu": mu, "logvar": logvar},
            {"seed": sample_seed},
        )
        dec1 = g.add_node(
            "dec1", "linear", {"x": z}, {"weight": "param:dec1.weight", "bias": "param:dec1.bias"}
        )
        act2 = g.add_node("dec_act", "tanh", {"x": dec1})
        recon = g.add_node(
            "dec2",
            "linear",
            {"x": act2},
            {"weight": "param:dec2.weight", "bias": "param:dec2.bias"},
        )
        g.set_output("reconstruction", recon)
        g.set_output("latent", z)
        g.set_output("mu", mu)
        g.set_output("logvar", logvar)
        definition = ArchitectureDefinition(
            family=self.family,
            spec=spec_clean,
            graph=g,
            parameter_shapes=sh,
            default_inputs={"x": ("B", input_dim)},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class DiffusionBuilder(FamilyBuilder):
    family: str = "diffusion"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        spec = self.spec
        input_dim = int(self._require(spec, "input_dim"))
        hidden = int(self._require(spec, "hidden_dim", 2 * input_dim))
        alpha_bar = float(spec.get("alpha_bar", 0.5))
        noise_seed = int(spec.get("noise_seed", 0))
        if not 0.0 < alpha_bar <= 1.0:
            raise InvalidOperationParameters("diffusion alpha_bar must be in (0, 1]")
        spec_clean = {
            "input_dim": input_dim,
            "hidden_dim": hidden,
            "alpha_bar": alpha_bar,
            "noise_seed": noise_seed,
        }
        g = Graph(name="diffusion_denoiser", metadata={"family": self.family, "spec": spec_clean})
        g.add_input("x", TensorType.of(("B", input_dim), "float32"))
        g.add_input("t", TensorType.of(("B", 1), "float32"))
        sh: dict[str, tuple[int, ...]] = {
            "h1.weight": (hidden, input_dim + 1),
            "h1.bias": (hidden,),
            "h2.weight": (hidden, hidden),
            "h2.bias": (hidden,),
            "eps.weight": (input_dim, hidden),
            "eps.bias": (input_dim,),
        }
        noisy = g.add_node(
            "noise",
            "add_noise",
            {"x": Ref("input:x")},
            {"alpha_bar": alpha_bar, "seed": noise_seed},
        )
        fused = g.add_node(
            "concat", "concat", {"a": noisy, "b": Ref("input:t")}, {"axis": -1}
        )
        h1 = g.add_node(
            "h1", "linear", {"x": fused}, {"weight": "param:h1.weight", "bias": "param:h1.bias"}
        )
        act1 = g.add_node("act1", "silu", {"x": h1})
        h2 = g.add_node(
            "h2", "linear", {"x": act1}, {"weight": "param:h2.weight", "bias": "param:h2.bias"}
        )
        act2 = g.add_node("act2", "silu", {"x": h2})
        eps = g.add_node(
            "eps", "linear", {"x": act2}, {"weight": "param:eps.weight", "bias": "param:eps.bias"}
        )
        g.set_output("eps_pred", eps)
        g.set_output("x_t", noisy)
        definition = ArchitectureDefinition(
            family=self.family,
            spec=spec_clean,
            graph=g,
            parameter_shapes=sh,
            default_inputs={"x": ("B", input_dim), "t": ("B", 1)},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class MultimodalBuilder(FamilyBuilder):
    family: str = "multimodal"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        spec = self.spec
        in_channels = int(self._require(spec, "in_channels"))
        height = int(self._require(spec, "height"))
        width = int(self._require(spec, "width"))
        vocab = int(self._require(spec, "vocab_size"))
        d_model = int(self._require(spec, "d_model", 16))
        num_classes = int(self._require(spec, "num_classes"))
        spec_clean = {
            "in_channels": in_channels,
            "height": height,
            "width": width,
            "vocab_size": vocab,
            "d_model": d_model,
            "num_classes": num_classes,
        }
        g = Graph(name="multimodal_fusion", metadata={"family": self.family, "spec": spec_clean})
        g.add_input("image", TensorType.of(("B", in_channels, height, width), "float32"))
        g.add_input("tokens", TensorType.of(("B", "T"), "int64"))
        flat_dim = in_channels * height * width
        sh: dict[str, tuple[int, ...]] = {
            "image_proj.weight": (d_model, flat_dim),
            "image_proj.bias": (d_model,),
            "embed.weight": (vocab, d_model),
            "text_proj.weight": (d_model, d_model),
            "text_proj.bias": (d_model,),
            "head.weight": (num_classes, 2 * d_model),
            "head.bias": (num_classes,),
        }
        flat = g.add_node(
            "flatten", "flatten", {"x": Ref("input:image")}, {"start_dim": 1}
        )
        image_feat = g.add_node(
            "image_proj",
            "linear",
            {"x": flat},
            {"weight": "param:image_proj.weight", "bias": "param:image_proj.bias"},
        )
        embed = g.add_node(
            "embed", "embedding", {"x": Ref("input:tokens")}, {"weight": "param:embed.weight"}
        )
        pooled = g.add_node("text_pool", "mean", {"x": embed}, {"axis": 1})
        text_feat = g.add_node(
            "text_proj",
            "linear",
            {"x": pooled},
            {"weight": "param:text_proj.weight", "bias": "param:text_proj.bias"},
        )
        fused = g.add_node(
            "fuse", "concat", {"a": image_feat, "b": text_feat}, {"axis": -1}
        )
        logits = g.add_node(
            "head",
            "linear",
            {"x": fused},
            {"weight": "param:head.weight", "bias": "param:head.bias"},
        )
        g.set_output("logits", logits)
        g.set_output("image_features", image_feat)
        g.set_output("text_features", text_feat)
        definition = ArchitectureDefinition(
            family=self.family,
            spec=spec_clean,
            graph=g,
            parameter_shapes=sh,
            default_inputs={
                "image": ("B", in_channels, height, width),
                "tokens": ("B", "T"),
            },
            input_dtypes={"tokens": "int64"},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class HybridBuilder(FamilyBuilder):
    family: str = "hybrid"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        spec = self.spec
        in_channels = int(self._require(spec, "in_channels"))
        height = int(self._require(spec, "height"))
        width = int(self._require(spec, "width"))
        channels = [int(c) for c in (spec.get("channels") or [4])]
        kernel = int(spec.get("kernel", 3))
        stride = int(spec.get("stride", 1))
        padding = int(spec.get("padding", 1))
        pool = int(spec.get("pool", 2))
        hidden = int(self._require(spec, "hidden_size"))
        num_classes = int(self._require(spec, "num_classes"))
        spec_clean = {
            "in_channels": in_channels,
            "height": height,
            "width": width,
            "channels": channels,
            "kernel": kernel,
            "stride": stride,
            "padding": padding,
            "pool": pool,
            "hidden_size": hidden,
            "num_classes": num_classes,
        }
        g = Graph(name="conv_gru_hybrid", metadata={"family": self.family, "spec": spec_clean})
        g.add_input("image", TensorType.of(("B", in_channels, height, width), "float32"))
        sh: dict[str, tuple[int, ...]] = {}
        cur = Ref("input:image")
        prev_c = in_channels
        h, w = height, width
        for i, out_c in enumerate(channels):
            sh[f"conv{i}.weight"] = (out_c, prev_c, kernel, kernel)
            sh[f"conv{i}.bias"] = (out_c,)
            cur = g.add_node(
                f"conv{i}",
                "conv2d",
                {"x": cur},
                {
                    "weight": f"param:conv{i}.weight",
                    "bias": f"param:conv{i}.bias",
                    "stride": stride,
                    "padding": padding,
                },
            )
            cur = g.add_node(f"act{i}", "relu", {"x": cur})
            cur = g.add_node(f"pool{i}", "maxpool2d", {"x": cur}, {"kernel": pool, "stride": pool})
            h = (h + 2 * padding - kernel) // stride + 1
            h = (h - pool) // pool + 1
            w = (w + 2 * padding - kernel) // stride + 1
            w = (w - pool) // pool + 1
            if h <= 0 or w <= 0:
                raise InvalidOperationParameters(
                    f"hybrid spatial dims collapsed to {h}x{w} after layer {i}"
                )
            prev_c = out_c
        flat = g.add_node("flatten", "flatten", {"x": cur}, {"start_dim": 1})
        tokens = h * w
        sh["proj.weight"] = (tokens * hidden, prev_c * h * w)
        sh["proj.bias"] = (tokens * hidden,)
        proj = g.add_node(
            "proj",
            "linear",
            {"x": flat},
            {"weight": "param:proj.weight", "bias": "param:proj.bias"},
        )
        seq = g.add_node("seq", "reshape", {"x": proj}, {"shape": ["B", tokens, hidden]})
        sh["rnn.weight_ih"] = (3 * hidden, hidden)
        sh["rnn.weight_hh"] = (3 * hidden, hidden)  # GRU stores three gate blocks
        sh["rnn.bias"] = (3 * hidden,)
        rec = g.add_node(
            "rnn",
            "gru",
            {"x": seq},
            {
                "weight_ih": "param:rnn.weight_ih",
                "weight_hh": "param:rnn.weight_hh",
                "bias": "param:rnn.bias",
            },
        )
        last = g.add_node("last", "slice", {"x": rec}, {"axis": 1, "start": -1})
        squeezed = g.add_node("squeeze", "reshape", {"x": last}, {"shape": ["B", hidden]})
        sh["head.weight"] = (num_classes, hidden)
        sh["head.bias"] = (num_classes,)
        logits = g.add_node(
            "head",
            "linear",
            {"x": squeezed},
            {"weight": "param:head.weight", "bias": "param:head.bias"},
        )
        g.set_output("logits", logits)
        g.set_output("hidden", squeezed)
        definition = ArchitectureDefinition(
            family=self.family,
            spec=spec_clean,
            graph=g,
            parameter_shapes=sh,
            default_inputs={"image": ("B", in_channels, height, width)},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


# --------------------------------------------------------------------------- #
# additional families, ordered from simple to advanced in FAMILY_ORDER
# --------------------------------------------------------------------------- #


def _add_encoder_block(
    g: Graph,
    cur: Ref,
    sh: dict[str, tuple[int, ...]],
    *,
    prefix: str,
    d_model: int,
    n_heads: int,
    d_ff: int,
    eps: float,
    causal: bool,
    activation: str = "gelu",
    shared: bool = False,
) -> Ref:
    """Append one pre-norm attention + MLP block and return its residual output."""
    p = "shared" if shared else prefix

    def param(name: str) -> str:
        return f"param:{p}.{name}"

    sh.setdefault(f"{p}.ln1.weight", (d_model,))
    sh.setdefault(f"{p}.ln1.bias", (d_model,))
    sh.setdefault(f"{p}.qkv.weight", (3 * d_model, d_model))
    sh.setdefault(f"{p}.qkv.bias", (3 * d_model,))
    sh.setdefault(f"{p}.attn_out.weight", (d_model, d_model))
    sh.setdefault(f"{p}.attn_out.bias", (d_model,))
    sh.setdefault(f"{p}.ln2.weight", (d_model,))
    sh.setdefault(f"{p}.ln2.bias", (d_model,))
    sh.setdefault(f"{p}.mlp_up.weight", (d_ff, d_model))
    sh.setdefault(f"{p}.mlp_up.bias", (d_ff,))
    sh.setdefault(f"{p}.mlp_down.weight", (d_model, d_ff))
    sh.setdefault(f"{p}.mlp_down.bias", (d_model,))
    ln1 = g.add_node(
        f"{prefix}_ln1",
        "layernorm",
        {"x": cur},
        {"weight": param("ln1.weight"), "bias": param("ln1.bias"), "eps": eps},
    )
    qkv = g.add_node(
        f"{prefix}_qkv",
        "linear",
        {"x": ln1},
        {"weight": param("qkv.weight"), "bias": param("qkv.bias")},
    )
    split = g.add_node(f"{prefix}_split", "split_qkv", {"x": qkv}, {"n_heads": n_heads})
    attn = g.add_node(
        f"{prefix}_attn",
        "attention",
        {"q": split, "k": Ref(f"{prefix}_split", 1), "v": Ref(f"{prefix}_split", 2)},
        {"causal": causal, "scale": "auto"},
    )
    merged = g.add_node(f"{prefix}_merge", "merge_heads", {"x": attn}, {"n_heads": n_heads})
    proj = g.add_node(
        f"{prefix}_attn_out",
        "linear",
        {"x": merged},
        {"weight": param("attn_out.weight"), "bias": param("attn_out.bias")},
    )
    cur = g.add_node(f"{prefix}_res1", "add", {"a": cur, "b": proj})
    ln2 = g.add_node(
        f"{prefix}_ln2",
        "layernorm",
        {"x": cur},
        {"weight": param("ln2.weight"), "bias": param("ln2.bias"), "eps": eps},
    )
    up = g.add_node(
        f"{prefix}_mlp_up",
        "linear",
        {"x": ln2},
        {"weight": param("mlp_up.weight"), "bias": param("mlp_up.bias")},
    )
    act = g.add_node(f"{prefix}_act", activation, {"x": up})
    down = g.add_node(
        f"{prefix}_mlp_down",
        "linear",
        {"x": act},
        {"weight": param("mlp_down.weight"), "bias": param("mlp_down.bias")},
    )
    return g.add_node(f"{prefix}_res2", "add", {"a": cur, "b": down})


def _add_token_encoder(
    g: Graph,
    sh: dict[str, tuple[int, ...]],
    *,
    vocab: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    max_seq: int,
    eps: float = 1e-5,
    activation: str = "gelu",
    causal: bool = False,
    shared: bool = False,
    prefix: str = "enc",
) -> Ref:
    sh.setdefault(f"{prefix}.embed.weight", (vocab, d_model))
    sh.setdefault(f"{prefix}.pos.weight", (max_seq, d_model))
    cur = g.add_node(
        f"{prefix}_embed",
        "embedding",
        {"x": Ref("input:tokens")},
        {"weight": f"param:{prefix}.embed.weight"},
    )
    cur = g.add_node(
        f"{prefix}_pos",
        "pos_embedding",
        {"x": cur},
        {"weight": f"param:{prefix}.pos.weight"},
    )
    for layer in range(n_layers):
        cur = _add_encoder_block(
            g,
            cur,
            sh,
            prefix=f"{prefix}{layer}",
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            eps=eps,
            causal=causal,
            activation=activation,
            shared=shared,
        )
    sh.setdefault(f"{prefix}.ln_f.weight", (d_model,))
    sh.setdefault(f"{prefix}.ln_f.bias", (d_model,))
    return g.add_node(
        f"{prefix}_ln_f",
        "layernorm",
        {"x": cur},
        {
            "weight": f"param:{prefix}.ln_f.weight",
            "bias": f"param:{prefix}.ln_f.bias",
            "eps": eps,
        },
    )


@dataclass
class LinearResidualBuilder(FamilyBuilder):
    family: str = "linear_residual"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        inp = int(self._require(self.spec, "input_dim"))
        hidden = int(self._require(self.spec, "hidden_dim", 2 * inp))
        out = int(self._require(self.spec, "out_dim", inp))
        g = Graph(name="linear_residual", metadata={"family": self.family})
        g.add_input("x", TensorType.of(("B", inp)))
        sh = {
            "up.weight": (hidden, inp),
            "up.bias": (hidden,),
            "down.weight": (out, hidden),
            "down.bias": (out,),
            "skip.weight": (out, inp),
            "skip.bias": (out,),
        }
        up = g.add_node("up", "linear", {"x": Ref("input:x")}, {"weight": "param:up.weight", "bias": "param:up.bias"})
        act = g.add_node("act", "relu", {"x": up})
        down = g.add_node("down", "linear", {"x": act}, {"weight": "param:down.weight", "bias": "param:down.bias"})
        skip = g.add_node("skip", "linear", {"x": Ref("input:x")}, {"weight": "param:skip.weight", "bias": "param:skip.bias"})
        out_ref = g.add_node("out", "add", {"a": down, "b": skip})
        g.set_output("y", out_ref)
        definition = ArchitectureDefinition(
            family=self.family,
            spec={"input_dim": inp, "hidden_dim": hidden, "out_dim": out},
            graph=g,
            parameter_shapes=sh,
            default_inputs={"x": ("B", inp)},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class GatedMLPBuilder(FamilyBuilder):
    family: str = "gated_mlp"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        inp = int(self._require(self.spec, "input_dim"))
        hidden = int(self._require(self.spec, "hidden_dim", 2 * inp))
        out = int(self._require(self.spec, "out_dim", inp))
        g = Graph(name="gated_mlp", metadata={"family": self.family})
        g.add_input("x", TensorType.of(("B", inp)))
        sh = {
            "up.weight": (hidden, inp),
            "up.bias": (hidden,),
            "gate.weight": (hidden, inp),
            "gate.bias": (hidden,),
            "down.weight": (out, hidden),
            "down.bias": (out,),
            "skip.weight": (out, inp),
            "skip.bias": (out,),
        }
        up = g.add_node("up", "linear", {"x": Ref("input:x")}, {"weight": "param:up.weight", "bias": "param:up.bias"})
        gate = g.add_node("gate", "linear", {"x": Ref("input:x")}, {"weight": "param:gate.weight", "bias": "param:gate.bias"})
        gate = g.add_node("silu", "silu", {"x": gate})
        fused = g.add_node("fused", "mul", {"a": up, "b": gate})
        down = g.add_node("down", "linear", {"x": fused}, {"weight": "param:down.weight", "bias": "param:down.bias"})
        skip = g.add_node("skip", "linear", {"x": Ref("input:x")}, {"weight": "param:skip.weight", "bias": "param:skip.bias"})
        out_ref = g.add_node("out", "add", {"a": down, "b": skip})
        g.set_output("y", out_ref)
        definition = ArchitectureDefinition(
            family=self.family,
            spec={"input_dim": inp, "hidden_dim": hidden, "out_dim": out},
            graph=g,
            parameter_shapes=sh,
            default_inputs={"x": ("B", inp)},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class ConvResidualBuilder(FamilyBuilder):
    family: str = "conv_residual"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        channels = int(self._require(self.spec, "channels", 1))
        height = int(self._require(self.spec, "height", 8))
        width = int(self._require(self.spec, "width", 8))
        num_classes = int(self._require(self.spec, "num_classes", 2))
        if height % 2 or width % 2:
            raise InvalidOperationParameters("conv_residual height and width must be even for pooling")
        g = Graph(name="conv_residual", metadata={"family": self.family})
        g.add_input("image", TensorType.of(("B", channels, height, width)))
        sh = {
            "conv1.weight": (channels, channels, 3, 3),
            "conv1.bias": (channels,),
            "conv2.weight": (channels, channels, 3, 3),
            "conv2.bias": (channels,),
            "head.weight": (num_classes, channels * (height // 2) * (width // 2)),
            "head.bias": (num_classes,),
        }
        conv1 = g.add_node("conv1", "conv2d", {"x": Ref("input:image")}, {"weight": "param:conv1.weight", "bias": "param:conv1.bias", "padding": 1})
        act = g.add_node("act", "relu", {"x": conv1})
        conv2 = g.add_node("conv2", "conv2d", {"x": act}, {"weight": "param:conv2.weight", "bias": "param:conv2.bias", "padding": 1})
        res = g.add_node("res", "add", {"a": Ref("input:image"), "b": conv2})
        pool = g.add_node("pool", "maxpool2d", {"x": res}, {"kernel": 2, "stride": 2})
        flat = g.add_node("flat", "flatten", {"x": pool}, {"start_dim": 1})
        logits = g.add_node("head", "linear", {"x": flat}, {"weight": "param:head.weight", "bias": "param:head.bias"})
        g.set_output("logits", logits)
        definition = ArchitectureDefinition(
            family=self.family,
            spec={"channels": channels, "height": height, "width": width, "num_classes": num_classes},
            graph=g,
            parameter_shapes=sh,
            default_inputs={"image": ("B", channels, height, width)},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class TransformerEncoderBuilder(FamilyBuilder):
    family: str = "transformer_encoder"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        vocab = int(self._require(self.spec, "vocab_size"))
        d_model = int(self._require(self.spec, "d_model"))
        n_heads = int(self._require(self.spec, "n_heads"))
        n_layers = int(self._require(self.spec, "n_layers", 2))
        d_ff = int(self._require(self.spec, "d_ff", 2 * d_model))
        max_seq = int(self._require(self.spec, "max_seq_len", 16))
        num_classes = int(self._require(self.spec, "num_classes", 2))
        if d_model % n_heads:
            raise InvalidOperationParameters("d_model must be divisible by n_heads")
        g = Graph(name="transformer_encoder", metadata={"family": self.family})
        g.add_input("tokens", TensorType.of(("B", "T"), "int64"))
        sh: dict[str, tuple[int, ...]] = {}
        cur = _add_token_encoder(
            g, sh, vocab=vocab, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, max_seq=max_seq, causal=False, prefix="enc",
        )
        pooled = g.add_node("pool", "mean", {"x": cur}, {"axis": 1})
        sh["head.weight"] = (num_classes, d_model)
        sh["head.bias"] = (num_classes,)
        logits = g.add_node("head", "linear", {"x": pooled}, {"weight": "param:head.weight", "bias": "param:head.bias"})
        g.set_output("logits", logits)
        g.set_output("hidden", cur)
        definition = ArchitectureDefinition(
            family=self.family,
            spec={"vocab_size": vocab, "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers, "d_ff": d_ff, "max_seq_len": max_seq, "num_classes": num_classes},
            graph=g,
            parameter_shapes=sh,
            default_inputs={"tokens": ("B", "T")},
            input_dtypes={"tokens": "int64"},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class AlbertBuilder(FamilyBuilder):
    family: str = "albert"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        vocab = int(self._require(self.spec, "vocab_size"))
        d_model = int(self._require(self.spec, "d_model"))
        n_heads = int(self._require(self.spec, "n_heads"))
        n_layers = int(self._require(self.spec, "n_layers", 3))
        d_ff = int(self._require(self.spec, "d_ff", 2 * d_model))
        max_seq = int(self._require(self.spec, "max_seq_len", 16))
        num_classes = int(self._require(self.spec, "num_classes", 2))
        if d_model % n_heads:
            raise InvalidOperationParameters("d_model must be divisible by n_heads")
        g = Graph(name="albert_shared", metadata={"family": self.family})
        g.add_input("tokens", TensorType.of(("B", "T"), "int64"))
        sh: dict[str, tuple[int, ...]] = {}
        cur = _add_token_encoder(
            g, sh, vocab=vocab, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, max_seq=max_seq, causal=False, shared=True, prefix="shared",
        )
        pooled = g.add_node("pool", "mean", {"x": cur}, {"axis": 1})
        sh["head.weight"] = (num_classes, d_model)
        sh["head.bias"] = (num_classes,)
        logits = g.add_node("head", "linear", {"x": pooled}, {"weight": "param:head.weight", "bias": "param:head.bias"})
        g.set_output("logits", logits)
        g.set_output("hidden", cur)
        definition = ArchitectureDefinition(
            family=self.family,
            spec={"vocab_size": vocab, "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers, "d_ff": d_ff, "max_seq_len": max_seq, "num_classes": num_classes},
            graph=g,
            parameter_shapes=sh,
            default_inputs={"tokens": ("B", "T")},
            input_dtypes={"tokens": "int64"},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class BertBuilder(FamilyBuilder):
    family: str = "bert"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        vocab = int(self._require(self.spec, "vocab_size"))
        d_model = int(self._require(self.spec, "d_model"))
        n_heads = int(self._require(self.spec, "n_heads"))
        n_layers = int(self._require(self.spec, "n_layers", 2))
        d_ff = int(self._require(self.spec, "d_ff", 2 * d_model))
        max_seq = int(self._require(self.spec, "max_seq_len", 16))
        num_classes = int(self._require(self.spec, "num_classes", 2))
        if d_model % n_heads:
            raise InvalidOperationParameters("d_model must be divisible by n_heads")
        g = Graph(name="bert", metadata={"family": self.family})
        g.add_input("tokens", TensorType.of(("B", "T"), "int64"))
        sh: dict[str, tuple[int, ...]] = {}
        cur = _add_token_encoder(
            g, sh, vocab=vocab, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, max_seq=max_seq, causal=False, prefix="bert",
        )
        first = g.add_node("first", "slice", {"x": cur}, {"axis": 1, "start": 0, "stop": 1})
        flat = g.add_node("flat", "reshape", {"x": first}, {"shape": ["B", d_model]})
        sh["pooler.weight"] = (d_model, d_model)
        sh["pooler.bias"] = (d_model,)
        pooled = g.add_node("pooler", "linear", {"x": flat}, {"weight": "param:pooler.weight", "bias": "param:pooler.bias"})
        pooled = g.add_node("pooler_tanh", "tanh", {"x": pooled})
        sh["head.weight"] = (num_classes, d_model)
        sh["head.bias"] = (num_classes,)
        logits = g.add_node("head", "linear", {"x": pooled}, {"weight": "param:head.weight", "bias": "param:head.bias"})
        g.set_output("logits", logits)
        g.set_output("pooled", pooled)
        definition = ArchitectureDefinition(
            family=self.family,
            spec={"vocab_size": vocab, "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers, "d_ff": d_ff, "max_seq_len": max_seq, "num_classes": num_classes},
            graph=g,
            parameter_shapes=sh,
            default_inputs={"tokens": ("B", "T")},
            input_dtypes={"tokens": "int64"},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class ViTBuilder(FamilyBuilder):
    family: str = "vit"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        channels = int(self._require(self.spec, "in_channels", 1))
        height = int(self._require(self.spec, "height", 8))
        width = int(self._require(self.spec, "width", 8))
        patch = int(self._require(self.spec, "patch", 2))
        d_model = int(self._require(self.spec, "d_model"))
        n_heads = int(self._require(self.spec, "n_heads"))
        n_layers = int(self._require(self.spec, "n_layers", 1))
        d_ff = int(self._require(self.spec, "d_ff", 2 * d_model))
        num_classes = int(self._require(self.spec, "num_classes", 2))
        if height % patch or width % patch:
            raise InvalidOperationParameters("vit height and width must be divisible by patch")
        if d_model % n_heads:
            raise InvalidOperationParameters("d_model must be divisible by n_heads")
        tokens = (height // patch) * (width // patch)
        g = Graph(name="vit", metadata={"family": self.family})
        g.add_input("image", TensorType.of(("B", channels, height, width)))
        sh: dict[str, tuple[int, ...]] = {
            "patch.weight": (d_model, channels, patch, patch),
            "patch.bias": (d_model,),
            "pos.weight": (tokens, d_model),
        }
        patch_ref = g.add_node("patch", "conv2d", {"x": Ref("input:image")}, {"weight": "param:patch.weight", "bias": "param:patch.bias", "stride": patch})
        flat = g.add_node("flat", "reshape", {"x": patch_ref}, {"shape": ["B", tokens, d_model]})
        cur = g.add_node("pos", "pos_embedding", {"x": flat}, {"weight": "param:pos.weight"})
        for layer in range(n_layers):
            cur = _add_encoder_block(g, cur, sh, prefix=f"vit{layer}", d_model=d_model, n_heads=n_heads, d_ff=d_ff, eps=1e-5, causal=False)
        sh["ln_f.weight"] = (d_model,)
        sh["ln_f.bias"] = (d_model,)
        cur = g.add_node("ln_f", "layernorm", {"x": cur}, {"weight": "param:ln_f.weight", "bias": "param:ln_f.bias", "eps": 1e-5})
        pooled = g.add_node("pool", "mean", {"x": cur}, {"axis": 1})
        sh["head.weight"] = (num_classes, d_model)
        sh["head.bias"] = (num_classes,)
        logits = g.add_node("head", "linear", {"x": pooled}, {"weight": "param:head.weight", "bias": "param:head.bias"})
        g.set_output("logits", logits)
        definition = ArchitectureDefinition(
            family=self.family,
            spec={"in_channels": channels, "height": height, "width": width, "patch": patch, "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers, "d_ff": d_ff, "num_classes": num_classes},
            graph=g,
            parameter_shapes=sh,
            default_inputs={"image": ("B", channels, height, width)},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class SetTransformerBuilder(FamilyBuilder):
    family: str = "set_transformer"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        vocab = int(self._require(self.spec, "vocab_size"))
        d_model = int(self._require(self.spec, "d_model"))
        n_heads = int(self._require(self.spec, "n_heads"))
        n_layers = int(self._require(self.spec, "n_layers", 1))
        d_ff = int(self._require(self.spec, "d_ff", 2 * d_model))
        max_seq = int(self._require(self.spec, "max_seq_len", 16))
        num_classes = int(self._require(self.spec, "num_classes", 2))
        if d_model % n_heads:
            raise InvalidOperationParameters("d_model must be divisible by n_heads")
        g = Graph(name="set_transformer", metadata={"family": self.family})
        g.add_input("tokens", TensorType.of(("B", "T"), "int64"))
        sh: dict[str, tuple[int, ...]] = {}
        cur = _add_token_encoder(g, sh, vocab=vocab, d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, max_seq=max_seq, causal=False, prefix="set")
        pooled = g.add_node("pool", "mean", {"x": cur}, {"axis": 1})
        sh["head.weight"] = (num_classes, d_model)
        sh["head.bias"] = (num_classes,)
        logits = g.add_node("head", "linear", {"x": pooled}, {"weight": "param:head.weight", "bias": "param:head.bias"})
        g.set_output("logits", logits)
        definition = ArchitectureDefinition(
            family=self.family,
            spec={"vocab_size": vocab, "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers, "d_ff": d_ff, "max_seq_len": max_seq, "num_classes": num_classes},
            graph=g,
            parameter_shapes=sh,
            default_inputs={"tokens": ("B", "T")},
            input_dtypes={"tokens": "int64"},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class SiameseBuilder(FamilyBuilder):
    family: str = "siamese"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        inp = int(self._require(self.spec, "input_dim"))
        hidden = int(self._require(self.spec, "hidden_dim", 2 * inp))
        num_classes = int(self._require(self.spec, "num_classes", 2))
        g = Graph(name="siamese", metadata={"family": self.family})
        g.add_input("a", TensorType.of(("B", inp)))
        g.add_input("b", TensorType.of(("B", inp)))
        sh = {
            "up.weight": (hidden, inp), "up.bias": (hidden,),
            "down.weight": (hidden, hidden), "down.bias": (hidden,),
            "head.weight": (num_classes, 2 * hidden), "head.bias": (num_classes,),
        }

        def encode(name: str) -> Ref:
            up = g.add_node(f"{name}_up", "linear", {"x": Ref(f"input:{name}")}, {"weight": "param:up.weight", "bias": "param:up.bias"})
            act = g.add_node(f"{name}_act", "relu", {"x": up})
            return g.add_node(f"{name}_down", "linear", {"x": act}, {"weight": "param:down.weight", "bias": "param:down.bias"})

        left = encode("a")
        right = encode("b")
        joined = g.add_node("join", "concat", {"a": left, "b": right}, {"axis": -1})
        logits = g.add_node("head", "linear", {"x": joined}, {"weight": "param:head.weight", "bias": "param:head.bias"})
        g.set_output("logits", logits)
        definition = ArchitectureDefinition(
            family=self.family,
            spec={"input_dim": inp, "hidden_dim": hidden, "num_classes": num_classes},
            graph=g,
            parameter_shapes=sh,
            default_inputs={"a": ("B", inp), "b": ("B", inp)},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class RetrievalBuilder(FamilyBuilder):
    family: str = "retrieval"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        vocab = int(self._require(self.spec, "vocab_size"))
        d_model = int(self._require(self.spec, "d_model"))
        hidden = int(self._require(self.spec, "hidden_dim", 2 * d_model))
        embed_dim = int(self._require(self.spec, "embed_dim", d_model))
        max_seq = int(self._require(self.spec, "max_seq_len", 16))
        g = Graph(name="retrieval", metadata={"family": self.family})
        g.add_input("tokens_a", TensorType.of(("B", "T"), "int64"))
        g.add_input("tokens_b", TensorType.of(("B", "T"), "int64"))
        sh = {
            "embed.weight": (vocab, d_model),
            "up.weight": (hidden, d_model), "up.bias": (hidden,),
            "proj.weight": (embed_dim, hidden), "proj.bias": (embed_dim,),
        }

        def encode(name: str) -> Ref:
            emb = g.add_node(f"{name}_embed", "embedding", {"x": Ref(f"input:{name}")}, {"weight": "param:embed.weight"})
            pooled = g.add_node(f"{name}_pool", "mean", {"x": emb}, {"axis": 1})
            up = g.add_node(f"{name}_up", "linear", {"x": pooled}, {"weight": "param:up.weight", "bias": "param:up.bias"})
            act = g.add_node(f"{name}_act", "relu", {"x": up})
            return g.add_node(f"{name}_proj", "linear", {"x": act}, {"weight": "param:proj.weight", "bias": "param:proj.bias"})

        left = encode("tokens_a")
        right = encode("tokens_b")
        right_t = g.add_node("right_t", "transpose", {"x": right}, {"perm": [1, 0]})
        logits = g.add_node("logits", "matmul", {"a": left, "b": right_t})
        g.set_output("logits", logits)
        definition = ArchitectureDefinition(
            family=self.family,
            spec={"vocab_size": vocab, "d_model": d_model, "hidden_dim": hidden, "embed_dim": embed_dim, "max_seq_len": max_seq},
            graph=g,
            parameter_shapes=sh,
            default_inputs={"tokens_a": ("B", "T"), "tokens_b": ("B", "T")},
            input_dtypes={"tokens_a": "int64", "tokens_b": "int64"},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class PerceiverBuilder(FamilyBuilder):
    family: str = "perceiver"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        vocab = int(self._require(self.spec, "vocab_size"))
        d_model = int(self._require(self.spec, "d_model"))
        n_heads = int(self._require(self.spec, "n_heads"))
        latent_count = int(self._require(self.spec, "latent_count", 4))
        d_ff = int(self._require(self.spec, "d_ff", 2 * d_model))
        max_seq = int(self._require(self.spec, "max_seq_len", 16))
        num_classes = int(self._require(self.spec, "num_classes", 2))
        if d_model % n_heads:
            raise InvalidOperationParameters("d_model must be divisible by n_heads")
        head_dim = d_model // n_heads
        g = Graph(name="perceiver", metadata={"family": self.family})
        g.add_input("tokens", TensorType.of(("B", "T"), "int64"))
        g.add_input("latents", TensorType.of(("B", latent_count, d_model)))
        sh: dict[str, tuple[int, ...]] = {
            "embed.weight": (vocab, d_model),
            "cross.q.weight": (d_model, d_model), "cross.q.bias": (d_model,),
            "cross.k.weight": (d_model, d_model), "cross.k.bias": (d_model,),
            "cross.v.weight": (d_model, d_model), "cross.v.bias": (d_model,),
            "cross.out.weight": (d_model, d_model), "cross.out.bias": (d_model,),
        }
        embed = g.add_node("embed", "embedding", {"x": Ref("input:tokens")}, {"weight": "param:embed.weight"})

        def heads(ref: Ref, name: str, length: int) -> Ref:
            shaped = g.add_node(f"{name}_shape", "reshape", {"x": ref}, {"shape": ["B", length, n_heads, head_dim]})
            return g.add_node(f"{name}_heads", "transpose", {"x": shaped}, {"perm": [0, 2, 1, 3]})

        q = g.add_node("cross_q", "linear", {"x": Ref("input:latents")}, {"weight": "param:cross.q.weight", "bias": "param:cross.q.bias"})
        k = g.add_node("cross_k", "linear", {"x": embed}, {"weight": "param:cross.k.weight", "bias": "param:cross.k.bias"})
        v = g.add_node("cross_v", "linear", {"x": embed}, {"weight": "param:cross.v.weight", "bias": "param:cross.v.bias"})
        attn = g.add_node("cross_attn", "attention", {"q": heads(q, "q", latent_count), "k": heads(k, "k", "T"), "v": heads(v, "v", "T")}, {"causal": False, "scale": "auto"})
        merged = g.add_node("cross_merge", "merge_heads", {"x": attn}, {"n_heads": n_heads})
        projected = g.add_node("cross_out", "linear", {"x": merged}, {"weight": "param:cross.out.weight", "bias": "param:cross.out.bias"})
        cur = g.add_node("cross_res", "add", {"a": Ref("input:latents"), "b": projected})
        cur = _add_encoder_block(g, cur, sh, prefix="latent_self", d_model=d_model, n_heads=n_heads, d_ff=d_ff, eps=1e-5, causal=False)
        sh["ln_f.weight"] = (d_model,)
        sh["ln_f.bias"] = (d_model,)
        cur = g.add_node("ln_f", "layernorm", {"x": cur}, {"weight": "param:ln_f.weight", "bias": "param:ln_f.bias", "eps": 1e-5})
        pooled = g.add_node("pool", "mean", {"x": cur}, {"axis": 1})
        sh["head.weight"] = (num_classes, d_model)
        sh["head.bias"] = (num_classes,)
        logits = g.add_node("head", "linear", {"x": pooled}, {"weight": "param:head.weight", "bias": "param:head.bias"})
        g.set_output("logits", logits)
        definition = ArchitectureDefinition(
            family=self.family,
            spec={"vocab_size": vocab, "d_model": d_model, "n_heads": n_heads, "latent_count": latent_count, "d_ff": d_ff, "max_seq_len": max_seq, "num_classes": num_classes},
            graph=g,
            parameter_shapes=sh,
            default_inputs={"tokens": ("B", "T"), "latents": ("B", latent_count, d_model)},
            input_dtypes={"tokens": "int64"},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


@dataclass
class Seq2SeqBuilder(FamilyBuilder):
    family: str = "seq2seq"
    spec: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ArchitectureDefinition:
        vocab = int(self._require(self.spec, "vocab_size"))
        d_model = int(self._require(self.spec, "d_model"))
        n_heads = int(self._require(self.spec, "n_heads"))
        n_layers = int(self._require(self.spec, "n_layers", 1))
        d_ff = int(self._require(self.spec, "d_ff", 2 * d_model))
        max_seq = int(self._require(self.spec, "max_seq_len", 16))
        if d_model % n_heads:
            raise InvalidOperationParameters("d_model must be divisible by n_heads")
        head_dim = d_model // n_heads
        g = Graph(name="seq2seq", metadata={"family": self.family})
        g.add_input("tokens", TensorType.of(("B", "T"), "int64"))
        g.add_input("decoder_tokens", TensorType.of(("B", "T"), "int64"))
        sh: dict[str, tuple[int, ...]] = {
            "embed.weight": (vocab, d_model), "pos.weight": (max_seq, d_model),
            "out.weight": (vocab, d_model), "out.bias": (vocab,),
        }
        enc = g.add_node("enc_embed", "embedding", {"x": Ref("input:tokens")}, {"weight": "param:embed.weight"})
        enc = g.add_node("enc_pos", "pos_embedding", {"x": enc}, {"weight": "param:pos.weight"})
        for layer in range(n_layers):
            enc = _add_encoder_block(g, enc, sh, prefix=f"enc{layer}", d_model=d_model, n_heads=n_heads, d_ff=d_ff, eps=1e-5, causal=False)
        sh["enc_ln.weight"] = (d_model,)
        sh["enc_ln.bias"] = (d_model,)
        enc = g.add_node("enc_ln", "layernorm", {"x": enc}, {"weight": "param:enc_ln.weight", "bias": "param:enc_ln.bias", "eps": 1e-5})

        cur = g.add_node("dec_embed", "embedding", {"x": Ref("input:decoder_tokens")}, {"weight": "param:embed.weight"})
        cur = g.add_node("dec_pos", "pos_embedding", {"x": cur}, {"weight": "param:pos.weight"})
        sh["dec.ln1.weight"] = (d_model,); sh["dec.ln1.bias"] = (d_model,)
        sh["dec.qkv.weight"] = (3 * d_model, d_model); sh["dec.qkv.bias"] = (3 * d_model,)
        sh["dec.self_out.weight"] = (d_model, d_model); sh["dec.self_out.bias"] = (d_model,)
        sh["dec.ln2.weight"] = (d_model,); sh["dec.ln2.bias"] = (d_model,)
        sh["dec.cross_q.weight"] = (d_model, d_model); sh["dec.cross_q.bias"] = (d_model,)
        sh["dec.cross_k.weight"] = (d_model, d_model); sh["dec.cross_k.bias"] = (d_model,)
        sh["dec.cross_v.weight"] = (d_model, d_model); sh["dec.cross_v.bias"] = (d_model,)
        sh["dec.cross_out.weight"] = (d_model, d_model); sh["dec.cross_out.bias"] = (d_model,)
        sh["dec.ln3.weight"] = (d_model,); sh["dec.ln3.bias"] = (d_model,)
        sh["dec.up.weight"] = (d_ff, d_model); sh["dec.up.bias"] = (d_ff,)
        sh["dec.down.weight"] = (d_model, d_ff); sh["dec.down.bias"] = (d_model,)
        sh["dec.ln_f.weight"] = (d_model,); sh["dec.ln_f.bias"] = (d_model,)

        ln1 = g.add_node("dec_ln1", "layernorm", {"x": cur}, {"weight": "param:dec.ln1.weight", "bias": "param:dec.ln1.bias", "eps": 1e-5})
        qkv = g.add_node("dec_qkv", "linear", {"x": ln1}, {"weight": "param:dec.qkv.weight", "bias": "param:dec.qkv.bias"})
        split = g.add_node("dec_split", "split_qkv", {"x": qkv}, {"n_heads": n_heads})
        self_attn = g.add_node("dec_self_attn", "attention", {"q": split, "k": Ref("dec_split", 1), "v": Ref("dec_split", 2)}, {"causal": True, "scale": "auto"})
        merged = g.add_node("dec_self_merge", "merge_heads", {"x": self_attn}, {"n_heads": n_heads})
        self_out = g.add_node("dec_self_out", "linear", {"x": merged}, {"weight": "param:dec.self_out.weight", "bias": "param:dec.self_out.bias"})
        cur = g.add_node("dec_res1", "add", {"a": cur, "b": self_out})

        ln2 = g.add_node("dec_ln2", "layernorm", {"x": cur}, {"weight": "param:dec.ln2.weight", "bias": "param:dec.ln2.bias", "eps": 1e-5})
        cq = g.add_node("dec_cross_q", "linear", {"x": ln2}, {"weight": "param:dec.cross_q.weight", "bias": "param:dec.cross_q.bias"})
        ck = g.add_node("dec_cross_k", "linear", {"x": enc}, {"weight": "param:dec.cross_k.weight", "bias": "param:dec.cross_k.bias"})
        cv = g.add_node("dec_cross_v", "linear", {"x": enc}, {"weight": "param:dec.cross_v.weight", "bias": "param:dec.cross_v.bias"})

        def heads(ref: Ref, name: str, length_symbol: str) -> Ref:
            shaped = g.add_node(f"{name}_shape", "reshape", {"x": ref}, {"shape": ["B", length_symbol, n_heads, head_dim]})
            return g.add_node(f"{name}_heads", "transpose", {"x": shaped}, {"perm": [0, 2, 1, 3]})

        cross_attn = g.add_node("dec_cross_attn", "attention", {"q": heads(cq, "cq", "T"), "k": heads(ck, "ck", "T"), "v": heads(cv, "cv", "T")}, {"causal": False, "scale": "auto"})
        cross_merged = g.add_node("dec_cross_merge", "merge_heads", {"x": cross_attn}, {"n_heads": n_heads})
        cross_out = g.add_node("dec_cross_out", "linear", {"x": cross_merged}, {"weight": "param:dec.cross_out.weight", "bias": "param:dec.cross_out.bias"})
        cur = g.add_node("dec_res2", "add", {"a": cur, "b": cross_out})

        ln3 = g.add_node("dec_ln3", "layernorm", {"x": cur}, {"weight": "param:dec.ln3.weight", "bias": "param:dec.ln3.bias", "eps": 1e-5})
        up = g.add_node("dec_up", "linear", {"x": ln3}, {"weight": "param:dec.up.weight", "bias": "param:dec.up.bias"})
        act = g.add_node("dec_act", "gelu", {"x": up})
        down = g.add_node("dec_down", "linear", {"x": act}, {"weight": "param:dec.down.weight", "bias": "param:dec.down.bias"})
        cur = g.add_node("dec_res3", "add", {"a": cur, "b": down})
        cur = g.add_node("dec_ln_f", "layernorm", {"x": cur}, {"weight": "param:dec.ln_f.weight", "bias": "param:dec.ln_f.bias", "eps": 1e-5})
        logits = g.add_node("out", "linear", {"x": cur}, {"weight": "param:out.weight", "bias": "param:out.bias"})
        g.set_output("logits", logits)
        g.set_output("encoder_hidden", enc)
        definition = ArchitectureDefinition(
            family=self.family,
            spec={"vocab_size": vocab, "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers, "d_ff": d_ff, "max_seq_len": max_seq},
            graph=g,
            parameter_shapes=sh,
            default_inputs={"tokens": ("B", "T"), "decoder_tokens": ("B", "T")},
            input_dtypes={"tokens": "int64", "decoder_tokens": "int64"},
        )
        definition.graph.require_valid(definition.build_package(seed=0))
        return definition


FAMILY_BUILDERS: dict[str, type[FamilyBuilder]] = {
    "transformer_decoder": TransformerBuilder,
    "mlp": MLPBuilder,
    "cnn": CNNBuilder,
    "recurrent": RecurrentBuilder,
    "autoencoder": AutoencoderBuilder,
    "moe": MoEBuilder,
    "state_space": StateSpaceBuilder,
    "gnn": GNNBuilder,
    "vae": VAEBuilder,
    "diffusion": DiffusionBuilder,
    "multimodal": MultimodalBuilder,
    "hybrid": HybridBuilder,
    "linear_residual": LinearResidualBuilder,
    "gated_mlp": GatedMLPBuilder,
    "conv_residual": ConvResidualBuilder,
    "transformer_encoder": TransformerEncoderBuilder,
    "albert": AlbertBuilder,
    "bert": BertBuilder,
    "vit": ViTBuilder,
    "set_transformer": SetTransformerBuilder,
    "siamese": SiameseBuilder,
    "retrieval": RetrievalBuilder,
    "perceiver": PerceiverBuilder,
    "seq2seq": Seq2SeqBuilder,
}


# Catalogue order is deliberate: simple feed-forward blocks first, then vision /
# sequence encoders, then dual-tower, latent-attention and encoder-decoder models.
FAMILY_ORDER: tuple[str, ...] = (
    "linear_residual",
    "gated_mlp",
    "mlp",
    "conv_residual",
    "cnn",
    "recurrent",
    "autoencoder",
    "siamese",
    "retrieval",
    "set_transformer",
    "transformer_encoder",
    "albert",
    "bert",
    "vit",
    "vae",
    "state_space",
    "gnn",
    "moe",
    "diffusion",
    "multimodal",
    "hybrid",
    "perceiver",
    "seq2seq",
    "transformer_decoder",
)


def build_architecture(family: str, spec: Mapping[str, Any] | None = None) -> ArchitectureDefinition:
    try:
        builder_cls = FAMILY_BUILDERS[family]
    except KeyError as exc:
        raise InvalidOperationParameters(
            f"unknown architecture family '{family}'; available: {sorted(FAMILY_BUILDERS)}"
        ) from exc
    builder = builder_cls()
    builder.spec = dict(spec or {})
    return builder.build()


def family_catalogue() -> list[dict[str, Any]]:
    notes = {
        "transformer_decoder": "pre-norm decoder-only transformer with fused QKV and causal attention",
        "mlp": "stacked linear + activation network",
        "cnn": "conv2d/relu/maxpool stack with a linear head",
        "recurrent": "embedding + rnn|lstm|gru + last-step linear head",
        "autoencoder": "linear encoder/decoder with tanh bottleneck",
        "moe": "token embedding + top-k routed mixture-of-experts + pooled linear head",
        "state_space": "embedding + diagonal state-space recurrence + pooled linear head",
        "gnn": "degree-normalized graph convolutions over dense adjacency + mean readout",
        "vae": "encoder/mu/logvar + seeded reparameterized sample + decoder",
        "diffusion": "seeded forward-noising step + time-conditioned MLP epsilon predictor",
        "multimodal": "image projection fused with pooled text embedding + linear head",
        "hybrid": "conv/pool trunk + GRU over spatial tokens + linear head",
        "linear_residual": "linear up/relu/down with a learned residual projection",
        "gated_mlp": "SwiGLU-style up/gate/down MLP with residual",
        "conv_residual": "residual 3x3 convolution block + pooling + linear head",
        "transformer_encoder": "bidirectional pre-norm transformer encoder with mean pooling",
        "albert": "encoder with one shared transformer block repeated across depth",
        "bert": "transformer encoder + first-token pooler + classifier",
        "vit": "patch convolution + positional tokens + transformer encoder",
        "set_transformer": "permutation-invariant self-attention over a token set",
        "siamese": "weight-shared twin MLP encoders + concatenation head",
        "retrieval": "shared text encoder with query/document dot-product logits",
        "perceiver": "latent cross-attention into token features + latent self-attention",
        "seq2seq": "bidirectional encoder + causal decoder with cross-attention",
    }
    return [
        {
            "family": name,
            "builder": FAMILY_BUILDERS[name].__name__,
            "supported": True,
            "notes": notes.get(name, ""),
        }
        for name in FAMILY_ORDER
        if name in FAMILY_BUILDERS
    ]
