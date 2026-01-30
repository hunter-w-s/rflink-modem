# tests/unit/test_framing.py
from __future__ import annotations

import pytest

from rflink_modem.protocol.framing import (
    SYNC_WORD,
    VERSION,
    HEADER_LEN,
    pack_frame,
    unpack_frame,
    frame_total_len,
)


def test_pack_unpack_roundtrip_whitened():
    from rflink_modem.protocol.framing import FLAG_WHITENED

    payload = b"hello whiten"
    frame = pack_frame(payload, flags=0x20, seq=0x1234, whiten_payload=True)
    hdr, out = unpack_frame(frame)

    assert out == payload
    assert (hdr.flags & FLAG_WHITENED) != 0
    # original user flag bit should still be present
    assert (hdr.flags & 0x20) != 0
    assert hdr.seq == 0x1234



def test_frame_len_helper():
    assert frame_total_len(0) == HEADER_LEN + 4
    assert frame_total_len(10) == HEADER_LEN + 10 + 4


def test_bad_payload_crc_raises():
    payload = b"abc123"
    frame = bytearray(pack_frame(payload))
    # flip a payload bit
    frame[HEADER_LEN + 1] ^= 0x01
    with pytest.raises(ValueError, match="bad payload crc32"):
        unpack_frame(bytes(frame))


def test_bad_header_crc_raises():
    payload = b"abc123"
    frame = bytearray(pack_frame(payload))
    # flip a header-core bit (inside first 8 bytes)
    frame[2] ^= 0x01
    with pytest.raises(ValueError, match="bad header crc16"):
        unpack_frame(bytes(frame))


def test_length_mismatch_raises():
    payload = b"abcdef"
    frame = pack_frame(payload)
    with pytest.raises(ValueError, match="frame length mismatch"):
        unpack_frame(frame[:-1])
