# tests/unit/test_whitening.py
from __future__ import annotations

from rflink_modem.protocol.whitening import whiten


def test_whiten_roundtrip():
    data = b"\x00\x01\x02\x03\xFE\xFFhello world\x00"
    w = whiten(data, seed=0xACE1)
    assert w != data
    assert whiten(w, seed=0xACE1) == data


def test_whiten_known_vector_seed_ace1():
    data = b"123456789"
    w = whiten(data, seed=0xACE1)
    assert w.hex() == "84a9efb7c7007546da"

