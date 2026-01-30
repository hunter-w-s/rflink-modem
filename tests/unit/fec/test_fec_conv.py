# tests/unit/test_fec_conv.py
from __future__ import annotations

import random

from rflink_modem.protocol.fec.fec_conv import ConvParams, conv_encode_bits, viterbi_decode_hard


def test_conv_encode_known_vector():
    params = ConvParams(K=7, g1=0o171, g2=0o133)
    bits = [1, 0, 1, 1, 0, 0, 1]
    enc = conv_encode_bits(bits, params=params, tail=True)
    # locked output bits (includes tail flush of K-1=6 zeros)
    assert enc == [
        1, 1,
        0, 1,
        1, 1,
        0, 1,
        1, 0,
        0, 1,
        0, 0,
        0, 0,
        0, 1,
        0, 0,
        1, 1,
        1, 0,
        1, 1,
    ]


def test_viterbi_roundtrip_with_bit_flips():
    rng = random.Random(2026)
    params = ConvParams()

    bits = [rng.randrange(2) for _ in range(80)]
    enc = conv_encode_bits(bits, params=params, tail=True)

    # flip a small number of received bits
    rx = enc[:]
    flip_positions = rng.sample(range(len(rx)), 12)
    for p in flip_positions:
        rx[p] ^= 1

    dec = viterbi_decode_hard(rx, params=params, tail=True)
    assert dec == bits
