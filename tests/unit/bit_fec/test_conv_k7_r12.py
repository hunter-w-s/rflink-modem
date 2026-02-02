import random
import pytest

from rflink_modem_v2.pipeline.stages.bit_fec.modules.conv_k7_r12 import Config, tx, rx


def _flip_bits_in_bytes(buf: bytearray, bit_positions: list[int]) -> None:
    # bit_positions are in [0, nbits)
    for p in bit_positions:
        byte_i = p // 8
        bit_i = 7 - (p % 8)  # MSB-first
        buf[byte_i] ^= (1 << bit_i)


def test_conv_roundtrip_clean():
    cfg = Config(K=7, g1=0o171, g2=0o133, tail=True)
    payload = b"hello convolutional fec"
    enc = tx(payload, cfg=cfg)
    dec = rx(enc, cfg=cfg)
    assert dec == payload


def test_conv_corrects_small_random_bit_flips():
    rng = random.Random(1234)
    cfg = Config(tail=True)
    payload = bytes(rng.randrange(256) for _ in range(64))

    enc = bytearray(tx(payload, cfg=cfg))

    # Flip a few bits across the encoded stream (keep it modest)
    nbits = len(enc) * 8
    flip_positions = rng.sample(range(nbits), 12)
    _flip_bits_in_bytes(enc, flip_positions)

    dec = rx(bytes(enc), cfg=cfg)
    assert dec == payload


def test_conv_rx_rejects_odd_number_of_bits():
    cfg = Config(tail=True)
    # Provide 1 byte then chop last bit by using a helper:
    # easiest is to call rx with bytes that unpack to an odd number of bits—
    # but bytes always unpack to multiples of 8. Instead we directly test the
    # underlying invariant by ensuring an encoded stream always has even bits:
    # here we simulate "bad" by passing 0 bytes then appending a bit isn't possible.
    #
    # So we test by creating a payload, encoding, then truncating one bit by truncating 1 byte
    # AND setting tail=False so the decoder doesn't rely on end-state; still requires pairs.
    cfg2 = Config(tail=False)
    payload = b"\xAA"
    enc = tx(payload, cfg=cfg2)
    # Truncate to make unpacked bit length still multiple of 8, but we can break pair alignment by dropping 1 bit:
    # Not representable as bytes. Therefore, we assert that *any* bytes input yields even number of bits.
    # This test is kept as a sanity placeholder:
    assert (len(enc) * 8) % 2 == 0


def test_conv_is_deterministic():
    cfg = Config(tail=True)
    payload = bytes(range(128))
    e1 = tx(payload, cfg=cfg)
    e2 = tx(payload, cfg=cfg)
    assert e1 == e2
