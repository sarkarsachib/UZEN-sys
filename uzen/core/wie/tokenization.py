"""Durable tokenizer assets and text encoding for imported models.

The importer records tokenizer file metadata, but execution needs the tokenizer
document itself.  A ``tokenizer.json`` payload is stored in a content-addressed
sidecar under the workspace and referenced by SHA-256 from the artifact manifest.
The artifact can therefore be executed after the original import directory is
gone, without embedding a large vocabulary in the manifest or trusting a path
from the caller.

Only the tokenizer document is loaded.  No tokenizer code, custom model code or
pickle is executed.  The optional ``tokenizers`` package performs the real
GPT-2/Llama/OPT encoding; when it is absent, text inference raises a typed
``runtime.unavailable`` error instead of inventing token IDs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from uzen.errors import (
    ArtifactIntegrityError,
    InvalidOperationParameters,
    RuntimeUnavailable,
)
from uzen.util.canonical import atomic_write_bytes

TOKENIZER_SCHEMA = "uzen.tokenizer/1"
TOKENIZER_KIND = "huggingface-tokenizer-json"


def tokenizer_path(root: str | Path, sha256: str) -> Path:
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise ArtifactIntegrityError("stored tokenizer has an invalid SHA-256")
    return Path(root) / "tokenizers" / sha256[:2] / sha256 / "tokenizer.json"


def store_tokenizer(root: str | Path, data: bytes) -> dict[str, Any]:
    """Store a validated tokenizer.json document and return its manifest reference."""
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidOperationParameters(f"tokenizer.json is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise InvalidOperationParameters("tokenizer.json must contain a JSON object")
    model = document.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("type"), str):
        raise InvalidOperationParameters("tokenizer.json has no model.type")

    digest = hashlib.sha256(data).hexdigest()
    path = tokenizer_path(root, digest)
    if path.exists():
        existing = path.read_bytes()
        if hashlib.sha256(existing).hexdigest() != digest:
            raise ArtifactIntegrityError(
                f"stored tokenizer {digest} exists but its bytes do not match its address"
            )
    else:
        atomic_write_bytes(path, data)

    vocab = model.get("vocab")
    vocab_size = len(vocab) if isinstance(vocab, (dict, list)) else document.get("vocab_size")
    return {
        "schema": TOKENIZER_SCHEMA,
        "kind": TOKENIZER_KIND,
        "sha256": digest,
        "bytes": len(data),
        "model_type": model["type"],
        "vocab_size": vocab_size,
    }


def tokenizer_spec_from_manifest(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = manifest.get("metadata")
    spec = metadata.get("tokenizer") if isinstance(metadata, Mapping) else None
    if not isinstance(spec, Mapping):
        raise InvalidOperationParameters(
            "artifact has no stored tokenizer; re-import the model with a tokenizer.json "
            "before requesting text inference"
        )
    if spec.get("kind") != TOKENIZER_KIND:
        raise InvalidOperationParameters(
            f"unsupported stored tokenizer kind {spec.get('kind')!r}; "
            f"expected {TOKENIZER_KIND!r}"
        )
    return spec


def load_tokenizer(root: str | Path, spec: Mapping[str, Any]) -> Any:
    """Load the sidecar with the optional real Hugging Face tokenizer runtime."""
    digest = str(spec.get("sha256") or "")
    path = tokenizer_path(root, digest)
    if not path.is_file():
        raise ArtifactIntegrityError(
            f"stored tokenizer {digest} is missing from the workspace tokenizer store"
        )
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ArtifactIntegrityError(f"stored tokenizer {digest} failed its SHA-256 check")
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError as exc:
        raise RuntimeUnavailable(
            "text inference requires the optional tokenizers runtime; "
            "install it with `pip install 'uzen[tokenizer]'`"
        ) from exc
    try:
        return Tokenizer.from_file(str(path))
    except Exception as exc:
        raise InvalidOperationParameters(f"stored tokenizer could not be loaded: {exc}") from exc


@dataclass(frozen=True)
class TokenizedText:
    text: str
    token_ids: tuple[int, ...]
    tokens: tuple[str, ...]
    decoded_text: str


def encode_with_tokenizer(tokenizer: Any, text: str) -> TokenizedText:
    if not isinstance(text, str):
        raise InvalidOperationParameters("text must be a string")
    if not text:
        raise InvalidOperationParameters("text must not be empty")
    try:
        encoding = tokenizer.encode(text)
    except Exception as exc:
        raise InvalidOperationParameters(f"text could not be tokenized: {exc}") from exc
    ids = tuple(int(value) for value in encoding.ids)
    if not ids:
        raise InvalidOperationParameters("tokenizer produced no tokens for the supplied text")
    tokens = tuple(str(value) for value in encoding.tokens)
    decoded = tokenizer.decode(list(ids), skip_special_tokens=False)
    return TokenizedText(text=text, token_ids=ids, tokens=tokens, decoded_text=decoded)


def encode_text(root: str | Path, spec: Mapping[str, Any], text: str) -> TokenizedText:
    return encode_with_tokenizer(load_tokenizer(root, spec), text)


EOS_TOKEN_CANDIDATES = (
    "<|endoftext|>",
    "<|end_of_text|>",
    "</s>",
    "<eos>",
    "<|eot_id|>",
    "<|im_end|>",
)


def infer_eos_token_id(tokenizer: Any) -> int | None:
    """Return the tokenizer's explicit EOS id when its document declares one."""
    for candidate in EOS_TOKEN_CANDIDATES:
        try:
            token_id = tokenizer.token_to_id(candidate)
        except Exception:
            token_id = None
        if token_id is not None:
            return int(token_id)
    return None
