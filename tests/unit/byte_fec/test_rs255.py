import random
import pytest

from rflink_modem_v2.pipeline.stages.byte_fec.modules.rs255 import Config, tx, rx


def _corrupt_bytes(buf: bytearray, positions: list[int], rng: random.Random) -> None:
    for p in positions:
        # ensure change
        old = buf[p]
        new = rng.randrange(256)
        while new == old:
            new = rng.randrange(256)
        buf[p] = new


def test_rs_roundtrip_single_block_no_errors():
    cfg = Config(nsym=32, pad=0x00)
    k = 255 - cfg.nsym
    payload = bytes(range(k))  # exactly one full message block
    enc = tx(payload, cfg=cfg)
    dec = rx(enc, cfg=cfg)
    assert dec == payload


def test_rs_corrects_up_to_t_errors_in_one_block():
    rng = random.Random(1234)
    nsym = 20
    t = nsym // 2
    cfg = Config(nsym=nsym, pad=0x00)
    k = 255 - nsym

    payload = bytes(rng.randrange(256) for _ in range(k))
    enc = bytearray(tx(payload, cfg=cfg))

    # corrupt t random byte positions within the 255-byte codeword
    positions = rng.sample(range(255), t)
    _corrupt_bytes(enc, positions, rng)

    dec = rx(bytes(enc), cfg=cfg)
    assert dec == payload


def test_rs_fails_when_exceeding_t_errors_in_one_block():
    rng = random.Random(2025)
    nsym = 20
    t = nsym // 2
    cfg = Config(nsym=nsym, pad=0x00)
    k = 255 - nsym

    payload = bytes(rng.randrange(256) for _ in range(k))
    enc = bytearray(tx(payload, cfg=cfg))

    # corrupt t+1 positions -> uncorrectable (in general)
    positions = rng.sample(range(255), t + 1)
    _corrupt_bytes(enc, positions, rng)

    with pytest.raises(ValueError):
        rx(bytes(enc), cfg=cfg)


def test_rs_roundtrip_multi_block_with_padding_prefix_matches():
    cfg = Config(nsym=32, pad=0xAA)
    payload = b"hello" * 100  # arbitrary length, spans multiple blocks
    enc = tx(payload, cfg=cfg)
    dec = rx(enc, cfg=cfg)

    # decoder returns padded length; original is prefix
    assert dec[:len(payload)] == payload

    k = 255 - cfg.nsym
    assert len(dec) % k == 0
    assert len(enc) % 255 == 0
