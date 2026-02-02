import pytest

from rflink_modem_v2.pipeline.stages.whiten.modules.lfsr16 import Config, tx, rx


def test_whiten_roundtrip():
    cfg = Config(seed=0xACE1)
    payload = b"\x00\x01\x02hello\xff\x10\x20"
    w = tx(payload, cfg=cfg)
    out = rx(w, cfg=cfg)
    assert out == payload
    assert w != payload  # should actually change most inputs


def test_whiten_is_deterministic_for_same_seed():
    cfg = Config(seed=0xBEEF)
    payload = bytes(range(256))
    w1 = tx(payload, cfg=cfg)
    w2 = tx(payload, cfg=cfg)
    assert w1 == w2


def test_whiten_diff_seed_diff_output():
    payload = b"same input"
    w1 = tx(payload, cfg=Config(seed=0xACE1))
    w2 = tx(payload, cfg=Config(seed=0xACE2))
    assert w1 != w2


def test_seed_validation_rejects_zero():
    with pytest.raises(ValueError):
        tx(b"abc", cfg=Config(seed=0))


def test_seed_validation_rejects_out_of_range():
    with pytest.raises(ValueError):
        tx(b"abc", cfg=Config(seed=0x1_0000))
