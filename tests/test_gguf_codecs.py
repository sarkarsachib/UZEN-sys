"""P13: new GGUF codecs (Q2_K, Q3_K, IQ4_NL) checked against the ``gguf`` reference.

Randomised blocks are dequantized by UZEN and by the reference ``gguf`` package; the
comparison covers every scale, high-bit mask and nibble position, which a hand-built
single block cannot. The fp16 scale fields are pinned to finite values so the test
compares numbers rather than NaN payloads.
"""

from __future__ import annotations

import numpy as np
import pytest

from uzen.errors import UnsupportedFormat
from uzen.infrastructure.storage.ggml_quant import (
    GGML_IQ4_NL,
    GGML_Q2_K,
    GGML_Q3_K,
    decode_iq4_nl,
    decode_q2_k,
    decode_q3_k,
    dequantize,
    type_info,
)

gguf = pytest.importorskip("gguf")
from gguf import GGMLQuantizationType, quants  # noqa: E402

CASES = [
    (GGML_Q2_K, GGMLQuantizationType.Q2_K, 84, 256, (80, 82)),
    (GGML_Q3_K, GGMLQuantizationType.Q3_K, 110, 256, (108,)),
    (GGML_IQ4_NL, GGMLQuantizationType.IQ4_NL, 18, 32, (0,)),
]


@pytest.mark.parametrize("type_id,reference_type,block_bytes,block_size,scale_offsets", CASES)
def test_decoder_matches_the_reference_implementation(
    type_id: int, reference_type: object, block_bytes: int, block_size: int, scale_offsets: tuple[int, ...]
) -> None:
    rng = np.random.default_rng(1234)
    blocks = 6
    raw = rng.integers(0, 256, size=(blocks, block_bytes), dtype=np.uint8)
    for table in (raw,):
        for offset in scale_offsets:
            # Pin fp16 d/dmin to 0.5 so no randomised block can produce inf/NaN.
            table[:, offset : offset + 2] = np.frombuffer(np.float16(0.5).tobytes(), dtype=np.uint8)
    ours = dequantize(raw.tobytes(), type_id, blocks * block_size)
    reference = np.asarray(quants.dequantize(raw, reference_type)).reshape(-1)
    np.testing.assert_allclose(ours, reference, rtol=1e-5, atol=1e-6)


def test_codecs_are_registered_and_lengths_are_enforced() -> None:
    assert type_info(GGML_Q2_K)["decoder_available"] is True
    assert type_info(GGML_Q3_K)["decoder_available"] is True
    assert type_info(GGML_IQ4_NL)["decoder_available"] is True
    with pytest.raises(Exception):
        dequantize(b"\x00" * 10, GGML_Q2_K, 256)
    with pytest.raises(Exception):
        dequantize(b"\x00" * 84, GGML_Q2_K, 255)
    with pytest.raises(UnsupportedFormat):
        dequantize(b"\x00" * 66, 16, 256)  # iq2_xxs is still metadata-only


def test_decoders_return_the_requested_element_count() -> None:
    for decoder, block_bytes, block_size in (
        (decode_q2_k, 84, 256),
        (decode_q3_k, 110, 256),
        (decode_iq4_nl, 18, 32),
    ):
        raw = bytes(block_bytes * 3)
        values = decoder(raw, block_size * 3)
        assert values.shape == (block_size * 3,)
        assert np.isfinite(values).all()
