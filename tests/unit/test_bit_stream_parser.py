# tests/unit/test_bit_stream_parser.py
from __future__ import annotations

import os
import random

from rflink_modem.protocol.framing import pack_frame
from rflink_modem.verify.bit_stream_parser import BitFrameStreamParser
from rflink_modem.verify.sync_search import bytes_to_bits


def _chunk(bits, rng: random.Random, min_sz: int = 1, max_sz: int = 97):
    i = 0
    n = len(bits)
    while i < n:
        sz = rng.randint(min_sz, max_sz)
        yield bits[i : i + sz]
        i += sz


def test_bit_stream_parser_roundtrip_misaligned_chunked():
    rng = random.Random(2026)
    p = BitFrameStreamParser()

    payload = os.urandom(120)
    frame = pack_frame(payload, flags=0x5A, seq=0x1234)

    prefix = [rng.randrange(2) for _ in range(rng.randrange(8))]  # 0..7 bits
    bitstream = prefix + bytes_to_bits(frame)

    out = []
    for part in _chunk(bitstream, rng):
        out.extend(p.feed_bits(part))

    assert len(out) == 1
    hdr, pl = out[0]
    assert pl == payload
    assert hdr.flags == 0x5A
    assert hdr.seq == 0x1234


def test_bit_stream_parser_corrupt_first_recovers_second():
    rng = random.Random(777)
    p = BitFrameStreamParser()

    f1 = bytearray(pack_frame(b"first", flags=1, seq=1))
    f2 = pack_frame(b"second", flags=2, seq=2)

    # Flip one payload bit in first frame (payload begins after 10-byte header)
    f1[10 + 1] ^= 0x01

    bits = [rng.randrange(2) for _ in range(5)] + bytes_to_bits(bytes(f1) + f2)
    out = []
    for part in _chunk(bits, rng, min_sz=1, max_sz=31):
        out.extend(p.feed_bits(part))

    assert [pl for _hdr, pl in out] == [b"second"]
    assert p.stats.frames_ok == 1
    assert p.stats.candidates_tried >= 1
