# tests/unit/test_stream_parser.py
from __future__ import annotations

import os

from rflink_modem.protocol.framing import pack_frame
from rflink_modem.verify.stream_parser import FrameStreamParser


def test_stream_parser_single_frame_one_chunk():
    p = FrameStreamParser()
    payload = b"hello"
    frame = pack_frame(payload, flags=0x01, seq=123)

    out = p.feed(frame)
    assert len(out) == 1
    hdr, pl = out[0]
    assert pl == payload
    assert hdr.flags == 0x01
    assert hdr.seq == 123


def test_stream_parser_frame_split_across_chunks():
    p = FrameStreamParser()
    payload = os.urandom(50)
    frame = pack_frame(payload, flags=0xAA, seq=0xBEEF)

    a = frame[:7]
    b = frame[7:25]
    c = frame[25:]

    assert p.feed(a) == []
    assert p.feed(b) == []
    out = p.feed(c)
    assert len(out) == 1
    hdr, pl = out[0]
    assert pl == payload
    assert hdr.flags == 0xAA
    assert hdr.seq == 0xBEEF


def test_stream_parser_garbage_between_frames():
    p = FrameStreamParser()
    f1 = pack_frame(b"one", flags=1, seq=1)
    f2 = pack_frame(b"two", flags=2, seq=2)

    stream = b"\x00\x01\x02junkjunk" + f1 + b"\xFF\xFEgarbage" + f2 + b"tail"
    out = p.feed(stream)
    assert [pl for _hdr, pl in out] == [b"one", b"two"]


def test_stream_parser_corrupt_first_frame_recovers_for_second():
    p = FrameStreamParser()
    f1 = bytearray(pack_frame(b"first", flags=3, seq=3))
    f2 = pack_frame(b"second", flags=4, seq=4)

    # Corrupt a payload byte (will break CRC32)
    # payload starts after 10-byte header; flip one byte
    f1[10 + 1] ^= 0x01

    stream = bytes(f1) + f2
    out = p.feed(stream)
    # should eventually recover and decode the second frame
    assert [pl for _hdr, pl in out] == [b"second"]
    assert p.stats.frames_bad_payload_crc >= 1
    assert p.stats.frames_ok == 1
