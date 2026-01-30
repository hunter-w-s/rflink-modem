import random
import pytest

from rflink_modem.protocol.fec.interleave import interleave_bytes, deinterleave_bytes


@pytest.mark.parametrize("depth", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("n", [0, 1, 2, 7, 8, 9, 31, 32, 33, 100, 255])
def test_interleave_roundtrip(depth, n):
    rng = random.Random(0xC0FFEE + depth * 1000 + n)
    data = bytes(rng.randrange(256) for _ in range(n))
    x = interleave_bytes(data, depth=depth)
    y = deinterleave_bytes(x, depth=depth, original_len=len(data))
    assert y == data


def test_interleave_spreads_burst():
    # demonstrate a contiguous burst in interleaved domain becomes dispersed after deinterleave
    data = bytes(range(64))
    depth = 8
    x = interleave_bytes(data, depth=depth)

    # corrupt 8 contiguous bytes in the interleaved domain
    start = 10
    burst = 8
    xb = bytearray(x)
    for i in range(burst):
        xb[start + i] ^= 0xFF

    yb = deinterleave_bytes(bytes(xb), depth=depth, original_len=len(data))

    # count how many distinct positions were affected; should be > burst/depth typically
    affected = [i for i, (a, b) in enumerate(zip(data, yb)) if a != b]
    assert len(affected) >= 8  # dispersed; typically equals burst but spread across rows
