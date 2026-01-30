# tests/unit/test_sync_search.py
from __future__ import annotations

from rflink_modem.protocol.framing import SYNC_WORD
from rflink_modem.verify.sync_search import bytes_to_bits, bits_to_bytes, find_sync_bits


def test_find_sync_aligned():
    sync_bytes = SYNC_WORD.to_bytes(2, "big")
    bits = bytes_to_bits(b"\x00\xFF" + sync_bytes + b"\xAA")
    hits = find_sync_bits(bits)
    # aligned => starts after 2 bytes = 16 bits
    assert hits == [16]


def test_find_sync_misaligned():
    sync_bytes = SYNC_WORD.to_bytes(2, "big")
    # Put 3 prefix bits before sync, then append sync bits.
    bits = [1, 0, 1] + bytes_to_bits(sync_bytes) + [0, 0, 1, 1]
    hits = find_sync_bits(bits)
    assert hits == [3]


def test_bits_to_bytes_from_offset_roundtrip():
    data = b"\xDE\xAD\xBE\xEF"
    bits = [1, 1, 0] + bytes_to_bits(data) + [0, 1]
    rebuilt = bits_to_bytes(bits, bit_offset=3, n_bytes=len(data))
    assert rebuilt == data
