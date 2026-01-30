# tests/unit/test_fec_rs.py
from __future__ import annotations

import random
import pytest

from rflink_modem.protocol.fec.fec_rs import rs_encode, rs_decode


def test_rs_encode_shape_and_decodes_clean():
    msg = b"hello"
    nsym = 10
    cw = rs_encode(msg, nsym=nsym)
    assert len(cw) == len(msg) + nsym
    assert rs_decode(cw, nsym=nsym) == msg



def test_rs_roundtrip_with_correctable_errors():
    rng = random.Random(1234)
    nsym = 10
    t = nsym // 2  # correct up to t byte errors

    msg = bytes(rng.randrange(256) for _ in range(50))
    cw = bytearray(rs_encode(msg, nsym=nsym))

    # introduce t random byte errors
    positions = rng.sample(range(len(cw)), t)
    for p in positions:
        cw[p] ^= rng.randrange(1, 256)

    

    out = rs_decode(bytes(cw), nsym=nsym)

    assert out == msg



def test_rs_raises_when_too_many_errors():
    rng = random.Random(5678)
    nsym = 10
    t = nsym // 2

    msg = bytes(rng.randrange(256) for _ in range(40))
    cw = bytearray(rs_encode(msg, nsym=nsym))

    # introduce t+1 errors -> should be uncorrectable
    positions = rng.sample(range(len(cw)), t + 1)
    for p in positions:
        cw[p] ^= rng.randrange(1, 256)

    with pytest.raises(ValueError, match="uncorrectable|could not locate"):
        rs_decode(bytes(cw), nsym=nsym)
