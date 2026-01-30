# tests/integration/test_end_to_end_frame_loopback.py
from __future__ import annotations

import random

from rflink_modem.protocol.framing import (
    HEADER_LEN,
    pack_frame,
    unpack_frame,
    frame_total_len,
    FLAG_WHITENED,
)
from rflink_modem.verify.sync_search import bytes_to_bits, bits_to_bytes, find_sync_bits


def _randbytes(rng: random.Random, n: int) -> bytes:
    return rng.randbytes(n) if hasattr(rng, "randbytes") else bytes(rng.randrange(256) for _ in range(n))


def _loop_once(*, rng: random.Random, whiten_payload: bool) -> None:
    payload_len = rng.randint(0, 200)
    payload = _randbytes(rng, payload_len)

    # Avoid accidental collision with FLAG_WHITENED when we assert flags behavior.
    flags = rng.randrange(256) & (~FLAG_WHITENED & 0xFF)
    seq = rng.randrange(65536)

    frame = pack_frame(payload, flags=flags, seq=seq, whiten_payload=whiten_payload)

    # Convert to bits and add a random 0..7 bit prefix to force misalignment.
    prefix_bits = [rng.randrange(2) for _ in range(rng.randrange(8))]
    bitstream = prefix_bits + bytes_to_bits(frame)

    hits = find_sync_bits(bitstream)
    assert hits, "sync not found in bitstream"
    start = hits[0]

    # Rebuild the header bytes from that bit offset (sanity; also exercises misaligned reconstruction)
    header = bits_to_bytes(bitstream, bit_offset=start, n_bytes=HEADER_LEN)
    assert len(header) == HEADER_LEN

    # Rebuild the entire frame (we know exact expected size from payload_len used to build it)
    total_len = frame_total_len(payload_len)
    rebuilt = bits_to_bytes(bitstream, bit_offset=start, n_bytes=total_len)

    hdr, out = unpack_frame(rebuilt)

    # Payload should always roundtrip to the original (unpack dewhitens if needed)
    assert out == payload
    assert hdr.payload_len == payload_len
    assert hdr.seq == seq

    if whiten_payload:
        assert (hdr.flags & FLAG_WHITENED) != 0
        assert (hdr.flags & (~FLAG_WHITENED & 0xFF)) == flags
    else:
        assert (hdr.flags & FLAG_WHITENED) == 0
        assert hdr.flags == flags


def test_end_to_end_frame_loopback_bitstream_scan_unwhitened():
    rng = random.Random(1337)
    for _ in range(50):
        _loop_once(rng=rng, whiten_payload=False)


def test_end_to_end_frame_loopback_bitstream_scan_whitened():
    rng = random.Random(1338)
    for _ in range(50):
        _loop_once(rng=rng, whiten_payload=True)
