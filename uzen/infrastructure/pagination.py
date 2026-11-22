"""Cursor pagination for control-plane list endpoints.

Lists are ordered by ``(created_at DESC, id DESC)`` and paged with a keyset cursor:
the cursor carries the last row's sort key, and the next page is
``WHERE (created_at, id) < cursor``. Unlike ``OFFSET`` this is stable while rows are
being inserted (a job finishing between two page requests cannot shift items across
pages) and it uses the ``created_at`` index instead of scanning skipped rows.

The cursor is opaque to clients: base64url of a small canonical JSON object. It is
signed by nothing and carries no secrets, so it is safe to hand out; tampering only
produces :class:`~uzen.errors.InvalidCursor` (a 400), never a wider query.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Iterable, Mapping, Sequence

from uzen.errors import InvalidCursor

DEFAULT_LIMIT = 50
MAX_LIMIT = 500
_CURSOR_VERSION = 1


def clamp_limit(limit: int, *, default: int = DEFAULT_LIMIT, maximum: int = MAX_LIMIT) -> int:
    """Bound a caller-supplied page size (a negative or zero limit is a 400)."""
    try:
        value = int(limit)
    except (TypeError, ValueError) as exc:
        raise InvalidCursor(f"limit must be an integer, got {limit!r}") from exc
    if value <= 0:
        raise InvalidCursor("limit must be a positive integer", limit=value)
    return min(value, int(maximum)) if maximum else value


def encode_cursor(values: Mapping[str, str]) -> str:
    payload = {"v": _CURSOR_VERSION, **{str(k): str(v) for k, v in values.items()}}
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str, *, keys: Sequence[str] = ("created_at", "id")) -> dict[str, str]:
    """Decode a cursor, refusing anything that is not exactly this page's shape."""
    if not isinstance(cursor, str) or not cursor:
        raise InvalidCursor("cursor must be a non-empty string")
    padded = cursor + "=" * (-len(cursor) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
        raise InvalidCursor("cursor is not valid base64-encoded JSON") from exc
    if not isinstance(payload, dict) or payload.get("v") != _CURSOR_VERSION:
        raise InvalidCursor("unsupported cursor version")
    missing = [key for key in keys if not isinstance(payload.get(key), str) or not payload[key]]
    if missing:
        raise InvalidCursor("cursor is missing its sort key", missing=missing)
    return {key: payload[key] for key in keys}


def split_page(items: Iterable[Any], limit: int) -> tuple[list[Any], bool]:
    """Split ``limit + 1`` fetched rows into one page and a ``has_more`` flag."""
    rows = list(items)
    return rows[:limit], len(rows) > limit


def page_payload(
    items: Sequence[Any],
    *,
    limit: int,
    cursor_of: Any,
    key: str = "items",
) -> dict[str, Any]:
    """Build ``{<key>, next_cursor, has_more, limit}`` from an over-fetched row list.

    ``cursor_of`` maps a row to the ``{"created_at": ..., "id": ...}`` sort key of
    that row. The next cursor points at the last row of the page, so the following
    request continues strictly after it.
    """
    page, has_more = split_page(items, limit)
    next_cursor = None
    if has_more and page:
        next_cursor = encode_cursor(cursor_of(page[-1]))
    return {
        key: list(page),
        "next_cursor": next_cursor,
        "has_more": bool(has_more),
        "limit": int(limit),
    }
