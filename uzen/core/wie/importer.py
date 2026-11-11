"""Hugging Face and safetensors checkpoint import.

The importer is deliberately conservative. It can establish that every tensor
was loaded, that a configuration was parsed and that roles were inferred; it
does **not** claim that an NIR graph was reconstructed or that the model was
executed. Those stages are reported separately and remain ``not_implemented`` /
``not_run`` until they exist.

Supported sources
-----------------
* a Hugging Face-style local directory (``config.json`` + ``model.safetensors``);
* a sharded safetensors checkpoint (``model.safetensors.index.json``);
* a single ``.safetensors`` file with optional adjacent metadata;
* a Hugging Face repository id through the optional ``huggingface_hub`` extra.

No pickle is read, no remote code is executed and no missing shard is silently
ignored. Source file hashes, revision/license metadata and tokenizer/vocabulary
metadata are recorded in the report and in the artifact manifest.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.nws.dtypes import dtype_name
from uzen.core.nws.roles import infer_role
from uzen.core.nws.tensors import WeightPackage
from uzen.core.wie.inspector import WeightIntelligenceEngine
from uzen.errors import ArtifactIntegrityError, RuntimeUnavailable, UnsupportedFormat
from uzen.infrastructure.storage.safetensors_io import read_safetensors
from uzen.util.canonical import ContentHasher, sha256_file, utc_now_iso

HF_ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.txt",
    "*.model",
    "*.jinja",
    "README.md",
]
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
]
KNOWN_FAMILIES = {
    "transformer_decoder",
    "mlp",
    "cnn",
    "recurrent",
    "autoencoder",
    "moe",
    "state_space",
    "gnn",
    "vae",
    "diffusion",
    "multimodal",
    "hybrid",
}


# --------------------------------------------------------------------------- #
# report types
# --------------------------------------------------------------------------- #


@dataclass
class SourceFile:
    path: str
    role: str
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }


@dataclass
class TensorRecord:
    name: str
    shape: list[int]
    dtype: str
    stored_dtype: str
    role: str
    source_file: str
    nbytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
            "stored_dtype": self.stored_dtype,
            "role": self.role,
            "source_file": self.source_file,
            "nbytes": self.nbytes,
        }


@dataclass
class Compatibility:
    weights: str
    config: str
    tokenizer: str
    architecture: str
    reconstruction: str
    execution: str
    overall: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "config": self.config,
            "tokenizer": self.tokenizer,
            "architecture": self.architecture,
            "reconstruction": self.reconstruction,
            "execution": self.execution,
            "overall": self.overall,
            "reasons": list(self.reasons),
        }


@dataclass
class ModelImportReport:
    source: str
    source_kind: str
    ok: bool
    family: str
    config: dict[str, Any]
    architecture: dict[str, Any]
    tokenizer: dict[str, Any]
    license: dict[str, Any]
    tensors: list[dict[str, Any]]
    role_counts: dict[str, int]
    compatibility: Compatibility
    files: list[dict[str, Any]]
    source_hash: str
    repo_id: str | None = None
    revision: str | None = None
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    understanding: dict[str, Any] = field(default_factory=dict)
    imported_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_kind": self.source_kind,
            "ok": self.ok,
            "family": self.family,
            "config": self.config,
            "architecture": self.architecture,
            "tokenizer": self.tokenizer,
            "license": self.license,
            "tensors": self.tensors,
            "role_counts": self.role_counts,
            "compatibility": self.compatibility.to_dict(),
            "files": self.files,
            "source_hash": self.source_hash,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "problems": self.problems,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "understanding": self.understanding,
            "imported_at": self.imported_at,
        }


@dataclass
class ModelImportResult:
    report: ModelImportReport
    package: WeightPackage | None = None

    @property
    def ok(self) -> bool:
        return self.report.ok and self.package is not None

    def require_ok(self) -> WeightPackage:
        if not self.ok or self.package is None:
            raise ArtifactIntegrityError(
                "model import did not produce a complete weight package",
                report=self.report.to_dict(),
            )
        return self.package


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _load_huggingface_hub():
    try:
        import huggingface_hub  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeUnavailable(
            "Hugging Face repository import requires the optional 'hf' extra; "
            "install it with `pip install 'uzen[hf]'` or provide a local model directory",
            dependency="huggingface_hub",
        ) from exc
    return huggingface_hub


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"invalid: {exc}"
    if not isinstance(value, dict):
        return None, "invalid: top-level JSON value is not an object"
    return value, None


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    out: dict[str, str] = {}
    for line in text[3:end].splitlines():
        if not line or line.lstrip().startswith(("-", "#")) or ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip().lower()] = value.strip().strip("\"'")
    return out


def _license_metadata(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    readme = root / "README.md"
    if readme.exists():
        try:
            front = _frontmatter(readme.read_text(encoding="utf-8", errors="replace")[:20000])
        except OSError:
            front = {}
        declared = front.get("license") or front.get("license_name")
        if declared:
            return {
                "declared": declared,
                "source": "model_card",
                "status": "declared",
            }
    declared = config.get("license")
    if declared:
        return {
            "declared": str(declared),
            "source": "config",
            "status": "declared",
        }
    return {
        "declared": None,
        "source": "unknown",
        "status": "unknown",
        "note": "no license was declared in README.md frontmatter or config.json",
    }


def _family_from_config(config: Mapping[str, Any]) -> str | None:
    archs = " ".join(str(a) for a in (config.get("architectures") or []))
    model_type = str(config.get("model_type") or "")
    text = f"{archs} {model_type}".lower()
    if any(
        token in text
        for token in (
            "causallm",
            "causal_lm",
            "llama",
            "mistral",
            "qwen",
            "gpt",
            "phi",
            "gemma",
            "falcon",
            "bloom",
            "opt",
            "mpt",
        )
    ):
        return "transformer_decoder"
    if any(token in text for token in ("resnet", "convnext", "efficientnet", "cnn")):
        return "cnn"
    if any(token in text for token in ("lstm", "gru", "rnn")):
        return "recurrent"
    if "autoencoder" in text:
        return "autoencoder"
    if "mlp" in text or "perceptron" in text:
        return "mlp"
    if "moe" in text or "mixture" in text:
        return "moe"
    return None


def _architecture_metadata(config: Mapping[str, Any], family: str | None) -> dict[str, Any]:
    keys = {
        "model_type": "model_type",
        "architectures": "architectures",
        "hidden_size": "hidden_size",
        "d_model": "hidden_size",
        "n_embd": "hidden_size",
        "num_hidden_layers": "num_layers",
        "n_layer": "num_layers",
        "num_attention_heads": "num_attention_heads",
        "n_head": "num_attention_heads",
        "num_key_value_heads": "num_key_value_heads",
        "intermediate_size": "intermediate_size",
        "vocab_size": "vocab_size",
        "max_position_embeddings": "max_position_embeddings",
        "max_seq_len": "max_position_embeddings",
        "tie_word_embeddings": "tie_word_embeddings",
        "torch_dtype": "torch_dtype",
        "dtype": "torch_dtype",
    }
    out: dict[str, Any] = {"family": family, "nir_family": family if family in KNOWN_FAMILIES else None}
    for source, target in keys.items():
        if source in config and target not in out:
            out[target] = config[source]
    out["note"] = (
        "configuration metadata only; NIR graph reconstruction is a separate step"
    )
    return out


def _tokenizer_metadata(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    files: list[SourceFile] = []
    details: dict[str, Any] = {}
    invalid = False
    for name in TOKENIZER_FILES:
        path = root / name
        if not path.is_file():
            continue
        files.append(
            SourceFile(
                path=name,
                role="tokenizer",
                sha256=sha256_file(path),
                bytes=path.stat().st_size,
            )
        )
        if name == "tokenizer.json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("tokenizer.json must contain a JSON object")
                model = data.get("model") or {}
                if isinstance(model, dict):
                    vocab = model.get("vocab")
                    details["tokenizer_model"] = model.get("type")
                    if isinstance(vocab, dict):
                        details["vocab_size"] = len(vocab)
                    elif isinstance(vocab, list):
                        details["vocab_size"] = len(vocab)
                    details["added_tokens"] = len(data.get("added_tokens") or [])
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                details["tokenizer_json_error"] = str(exc)
                invalid = True
        elif name == "vocab.json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    details.setdefault("vocab_size", len(data))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                pass
        elif name == "tokenizer_config.json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for key in ("model_max_length", "padding_side", "truncation_side"):
                        if key in data:
                            details[key] = data[key]
                    if "chat_template" in data:
                        details["chat_template_present"] = True
                    sp = data.get("special_tokens_map")
                    if isinstance(sp, dict):
                        details["special_tokens"] = sorted(str(k) for k in sp)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                pass
    if "vocab_size" not in details and config.get("vocab_size") is not None:
        details["vocab_size"] = config.get("vocab_size")
        details["vocab_size_source"] = "config.json"
    if invalid:
        status = "invalid"
    elif files:
        status = "present" if any(f.path in ("tokenizer.json", "vocab.json", "tokenizer.model") for f in files) else "config_only"
    else:
        status = "missing"
    return {
        "status": status,
        "files": [f.to_dict() for f in files],
        **details,
    }


def _source_hash(files: Sequence[SourceFile], *, repo_id: str | None, revision: str | None) -> str:
    h = ContentHasher()
    h.field("repo_id", repo_id or "")
    h.field("revision", revision or "")
    for item in sorted(files, key=lambda f: f.path):
        h.field(f"file:{item.path}", f"{item.sha256}|{item.role}|{item.bytes}")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# importer
# --------------------------------------------------------------------------- #


class ModelImporter:
    """Import Hugging Face/safetensors checkpoints with an honest report."""

    def __init__(self, *, hub: Any | None = None, engine: WeightIntelligenceEngine | None = None) -> None:
        self._hub = hub
        self.engine = engine or WeightIntelligenceEngine()

    def import_source(
        self,
        source: str | Path,
        *,
        name: str | None = None,
        family: str | None = None,
        revision: str | None = None,
        repo_id: str | None = None,
        local_files_only: bool = False,
        cache_dir: str | None = None,
    ) -> ModelImportResult:
        if repo_id is None and not Path(source).exists():
            text = str(source)
            if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text):
                repo_id = text
        if repo_id is not None:
            return self.import_huggingface_repo(
                repo_id,
                name=name,
                family=family,
                revision=revision,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(str(path))
        if path.is_dir():
            return self._import_directory(path, name=name, family=family, revision=revision)
        if path.suffix.lower() != ".safetensors":
            raise UnsupportedFormat(
                f"the HF/safetensors importer handles directories and .safetensors files; "
                f"got '{path.name}'. Use the legacy artifact importer for .npz/.gguf.",
                path=str(path),
            )
        return self._import_directory(
            path.parent,
            name=name,
            family=family,
            revision=revision,
            single_file=path,
        )

    def import_huggingface_repo(
        self,
        repo_id: str,
        *,
        name: str | None = None,
        family: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        cache_dir: str | None = None,
    ) -> ModelImportResult:
        hub = self._hub or _load_huggingface_hub()
        kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "revision": revision,
            "allow_patterns": HF_ALLOW_PATTERNS,
            "local_files_only": local_files_only,
        }
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        try:
            snapshot = hub.snapshot_download(**kwargs)
        except Exception as exc:
            raise RuntimeUnavailable(
                f"could not download Hugging Face repository '{repo_id}': {exc}",
                repo_id=repo_id,
                revision=revision,
            ) from exc
        result = self._import_directory(
            Path(snapshot),
            name=name,
            family=family,
            revision=revision,
            source_kind="huggingface_repo",
            repo_id=repo_id,
        )
        return result

    # ------------------------------------------------------------------ #
    # directory / file loading
    # ------------------------------------------------------------------ #

    def _import_directory(
        self,
        root: Path,
        *,
        name: str | None,
        family: str | None,
        revision: str | None,
        single_file: Path | None = None,
        source_kind: str | None = None,
        repo_id: str | None = None,
    ) -> ModelImportResult:
        problems: list[str] = []
        warnings: list[str] = []
        files: list[SourceFile] = []

        config, config_error = _read_json(root / "config.json")
        config_status = "parsed"
        if config is None:
            config = {}
            config_status = "missing" if config_error == "missing" else "invalid"
            if config_status == "missing":
                warnings.append("config.json is missing; the model can only be inspected as weights")
            else:
                warnings.append(
                    f"config.json is {config_error}; architecture metadata is unavailable, "
                    "but the weight payload can still be imported as weights_only"
                )

        family_from_config = _family_from_config(config) if config_status == "parsed" else None
        selected_family = family or family_from_config

        if (root / "config.json").exists():
            files.append(
                SourceFile(
                    path="config.json",
                    role="config",
                    sha256=sha256_file(root / "config.json"),
                    bytes=(root / "config.json").stat().st_size,
                )
            )
        index_path = root / "model.safetensors.index.json"
        if index_path.exists():
            files.append(
                SourceFile(
                    path=index_path.name,
                    role="shard_index",
                    sha256=sha256_file(index_path),
                    bytes=index_path.stat().st_size,
                )
            )

        tokenizer = _tokenizer_metadata(root, config)
        for entry in tokenizer.get("files", []):
            files.append(SourceFile(**entry))

        arrays: dict[str, np.ndarray] = {}
        records: list[TensorRecord] = []
        source_dtypes: dict[str, str] = {}
        directory_kind = source_kind or (
            "huggingface_directory" if (root / "config.json").exists()
            else "safetensors_directory"
        )
        loaded_kind = directory_kind

        if single_file is not None:
            loaded_kind = source_kind or "safetensors_file"
            try:
                shard_arrays, header = read_safetensors(single_file)
            except (ArtifactIntegrityError, UnsupportedFormat) as exc:
                problems.append(str(exc))
            else:
                arrays.update(shard_arrays)
                for tensor_name, entry in header.entries.items():
                    records.append(self._record(tensor_name, entry, single_file.name, shard_arrays[tensor_name]))
                    if entry.dtype_code == "BF16":
                        source_dtypes[tensor_name] = "bfloat16"
                files.append(
                    SourceFile(
                        path=single_file.name,
                        role="weights",
                        sha256=header.sha256 or sha256_file(single_file),
                        bytes=header.file_bytes,
                    )
                )
        elif index_path.exists():
            loaded_kind = directory_kind
            try:
                shard_arrays, shard_records, shard_files = self._load_sharded(root, index_path)
            except (ArtifactIntegrityError, UnsupportedFormat, OSError, ValueError) as exc:
                problems.append(str(exc))
            else:
                arrays.update(shard_arrays)
                records.extend(shard_records)
                files.extend(shard_files)
                source_dtypes.update(
                    {rec.name: "bfloat16" for rec in shard_records if rec.dtype == "BF16"}
                )
        else:
            candidates = sorted(root.glob("*.safetensors"))
            if not candidates:
                problems.append(
                    "no safetensors payload found; expected model.safetensors, "
                    "model.safetensors.index.json or *.safetensors"
                )
                loaded_kind = directory_kind
            elif len(candidates) == 1:
                loaded_kind = source_kind or directory_kind
                file = candidates[0]
                try:
                    shard_arrays, header = read_safetensors(file)
                except (ArtifactIntegrityError, UnsupportedFormat) as exc:
                    problems.append(str(exc))
                else:
                    arrays.update(shard_arrays)
                    for tensor_name, entry in header.entries.items():
                        records.append(self._record(tensor_name, entry, file.name, shard_arrays[tensor_name]))
                        if entry.dtype_code == "BF16":
                            source_dtypes[tensor_name] = "bfloat16"
                    files.append(
                        SourceFile(
                            path=file.name,
                            role="weights",
                            sha256=header.sha256 or sha256_file(file),
                            bytes=header.file_bytes,
                        )
                    )
            else:
                loaded_kind = directory_kind
                warnings.append(
                    f"no model.safetensors.index.json; combined {len(candidates)} shards by filename"
                )
                try:
                    shard_arrays, shard_records, shard_files = self._load_unindexed_shards(candidates)
                except (ArtifactIntegrityError, UnsupportedFormat, OSError) as exc:
                    problems.append(str(exc))
                else:
                    arrays.update(shard_arrays)
                    records.extend(shard_records)
                    files.extend(shard_files)
                    source_dtypes.update(
                        {r.name: "bfloat16" for r in shard_records if r.dtype == "BF16"}
                    )

        license_info = _license_metadata(root, config)
        readme = root / "README.md"
        if readme.exists():
            files.append(
                SourceFile(
                    path="README.md",
                    role="model_card",
                    sha256=sha256_file(readme),
                    bytes=readme.stat().st_size,
                )
            )

        if not selected_family and records:
            selected_family = self.engine.guess_family_from_names([r.name for r in records]).get("family")
        if not selected_family or selected_family == "unknown":
            selected_family = "generic"

        weights_status = "complete" if arrays and not problems else ("partial" if arrays else "failed")
        architecture_status = (
            "recognized"
            if family_from_config in KNOWN_FAMILIES
            else ("inferred" if selected_family in KNOWN_FAMILIES else "unknown")
        )
        reconstruction = "not_implemented"
        reasons = [
            "weight loading and metadata inspection are implemented",
            "NIR graph reconstruction from imported HF configurations is not implemented in this build",
            "no forward pass has been executed or validated",
        ]
        if problems:
            overall = "unsupported"
        elif (
            weights_status == "complete"
            and config_status == "parsed"
            and architecture_status == "recognized"
            and tokenizer.get("status") == "present"
        ):
            overall = "ready_for_reconstruction"
        elif weights_status == "complete":
            overall = "weights_only"
        else:
            overall = "inspect_only"

        compatibility = Compatibility(
            weights=weights_status,
            config=config_status,
            tokenizer=str(tokenizer.get("status")),
            architecture=architecture_status,
            reconstruction=reconstruction,
            execution="not_run",
            overall=overall,
            reasons=reasons,
        )

        if revision is None:
            commit_hash = config.get("_commit_hash")
            revision = str(commit_hash) if commit_hash else None

        architecture = _architecture_metadata(config, family_from_config or selected_family)
        understanding = {
            "weights_loaded": weights_status == "complete",
            "config_parsed": config_status == "parsed",
            "tokenizer_available": tokenizer.get("status") == "present",
            "architecture_family_recognized": architecture_status == "recognized",
            "graph_reconstructed": False,
            "execution_validated": False,
            "classification": overall,
            "note": "weights metadata can be complete while graph reconstruction is still unavailable",
        }
        role_counts: dict[str, int] = {}
        for rec in records:
            role_counts[rec.role] = role_counts.get(rec.role, 0) + 1
        role_counts["unclassified"] = sum(1 for r in records if r.role == "unclassified")

        display_source = str(single_file) if single_file is not None else str(root)
        default_name = single_file.stem if single_file is not None else root.name
        report = ModelImportReport(
            source=display_source,
            source_kind=loaded_kind or "safetensors_directory",
            ok=bool(arrays) and not problems,
            family=selected_family,
            config=config,
            architecture=architecture,
            tokenizer=tokenizer,
            license=license_info,
            tensors=[r.to_dict() for r in records],
            role_counts=role_counts,
            compatibility=compatibility,
            files=[f.to_dict() for f in files],
            source_hash=_source_hash(files, repo_id=repo_id, revision=revision),
            repo_id=repo_id,
            revision=revision,
            problems=problems,
            warnings=warnings,
            understanding=understanding,
            metadata={
                "importer": "hf/safetensors",
                "config_status": config_status,
                "sharded": index_path.exists() or len(list(root.glob("*.safetensors"))) > 1,
                "tensor_count": len(records),
                "numel": int(sum(int(np.prod(r.shape)) for r in records)),
                "note": "architecture reconstruction and execution are separate, unimplemented stages",
            },
        )

        if report.ok and arrays:
            package = WeightPackage.from_arrays(
                arrays,
                name=name or default_name,
                family=selected_family,
                metadata={
                    "import_source": display_source,
                    "source_kind": report.source_kind,
                    "source_hash": report.source_hash,
                    "repo_id": repo_id,
                    "revision": revision,
                    "license": license_info,
                    "architecture": architecture,
                    "tokenizer": {
                        "status": tokenizer.get("status"),
                        "vocab_size": tokenizer.get("vocab_size"),
                        "files": tokenizer.get("files", []),
                    },
                    "compatibility": compatibility.to_dict(),
                },
                source_dtypes=source_dtypes,
            )
            report.metadata["stored_family"] = package.family
            return ModelImportResult(report=report, package=package)
        return ModelImportResult(report=report, package=None)

    # ------------------------------------------------------------------ #
    # shard loading
    # ------------------------------------------------------------------ #

    def _load_sharded(
        self, root: Path, index_path: Path
    ) -> tuple[dict[str, np.ndarray], list[TensorRecord], list[SourceFile]]:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(f"invalid safetensors index: {exc}") from exc
        if not isinstance(index, dict):
            raise ArtifactIntegrityError("safetensors index must be a JSON object")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ArtifactIntegrityError("safetensors index has no non-empty weight_map")
        by_shard: dict[str, set[str]] = {}
        for tensor_name, shard_name in weight_map.items():
            if not isinstance(tensor_name, str) or not isinstance(shard_name, str):
                raise ArtifactIntegrityError("weight_map entries must map tensor names to shard filenames")
            shard_path = (root / shard_name).resolve()
            if root.resolve() not in shard_path.parents:
                raise ArtifactIntegrityError(
                    f"shard '{shard_name}' resolves outside the model directory"
                )
            by_shard.setdefault(shard_name, set()).add(tensor_name)

        arrays: dict[str, np.ndarray] = {}
        records: list[TensorRecord] = []
        files: list[SourceFile] = []
        seen_tensors: set[str] = set()
        for shard_name in sorted(by_shard):
            shard_path = root / shard_name
            if not shard_path.exists():
                raise ArtifactIntegrityError(f"missing shard '{shard_name}' declared in weight_map")
            shard_arrays, header = read_safetensors(shard_path)
            expected = by_shard[shard_name]
            actual = set(shard_arrays)
            missing = expected - actual
            extra = actual - expected
            if missing:
                raise ArtifactIntegrityError(
                    f"shard '{shard_name}' is missing declared tensors: {sorted(missing)[:10]}"
                )
            if extra:
                raise ArtifactIntegrityError(
                    f"shard '{shard_name}' contains undeclared tensors: {sorted(extra)[:10]}"
                )
            for tensor_name in sorted(expected):
                if tensor_name in seen_tensors:
                    raise ArtifactIntegrityError(
                        f"tensor '{tensor_name}' appears in more than one shard"
                    )
                seen_tensors.add(tensor_name)
                arrays[tensor_name] = shard_arrays[tensor_name]
                records.append(self._record(tensor_name, header.entries[tensor_name], shard_name, shard_arrays[tensor_name]))
            files.append(
                SourceFile(
                    path=shard_name,
                    role="weights",
                    sha256=header.sha256 or sha256_file(shard_path),
                    bytes=header.file_bytes,
                )
            )
        return arrays, records, files

    def _load_unindexed_shards(
        self, paths: Sequence[Path]
    ) -> tuple[dict[str, np.ndarray], list[TensorRecord], list[SourceFile]]:
        arrays: dict[str, np.ndarray] = {}
        records: list[TensorRecord] = []
        files: list[SourceFile] = []
        for shard_path in paths:
            shard_arrays, header = read_safetensors(shard_path)
            for tensor_name, entry in sorted(header.entries.items()):
                if tensor_name in arrays:
                    raise ArtifactIntegrityError(
                        f"tensor '{tensor_name}' appears in more than one shard"
                    )
                arrays[tensor_name] = shard_arrays[tensor_name]
                records.append(self._record(tensor_name, entry, shard_path.name, shard_arrays[tensor_name]))
            files.append(
                SourceFile(
                    path=shard_path.name,
                    role="weights",
                    sha256=header.sha256 or sha256_file(shard_path),
                    bytes=header.file_bytes,
                )
            )
        return arrays, records, files

    @staticmethod
    def _record(
        name: str,
        entry: Any,
        source_file: str,
        array: np.ndarray,
    ) -> TensorRecord:
        return TensorRecord(
            name=name,
            shape=[int(d) for d in entry.shape],
            dtype=entry.dtype_code,
            stored_dtype=dtype_name(array.dtype),
            role=infer_role(name, entry.shape).value,
            source_file=source_file,
            nbytes=int(array.nbytes),
        )


def import_model(
    source: str | Path,
    *,
    name: str | None = None,
    family: str | None = None,
    revision: str | None = None,
    repo_id: str | None = None,
) -> ModelImportResult:
    """Convenience wrapper around :class:`ModelImporter`."""
    return ModelImporter().import_source(
        source, name=name, family=family, revision=revision, repo_id=repo_id
    )
