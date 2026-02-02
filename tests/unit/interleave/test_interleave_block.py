import pytest

from rflink_modem_v2.pipeline.stages.interleave.modules.block import Config, tx, rx


def test_block_roundtrip_exact_multiple():
    cfg = Config(depth=8, pad=0x00)
    payload = bytes(range(64))  # 64 is multiple of 8
    y = tx(payload, cfg=cfg)
    out = rx(y, cfg=cfg)
    assert out == payload


def test_block_roundtrip_with_padding():
    cfg = Config(depth=8, pad=0xAA)
    payload = b"hello world"  # not multiple of 8
    y = tx(payload, cfg=cfg)
    out = rx(y, cfg=cfg)

    # deinterleaver returns padded length; original is prefix
    assert out[:len(payload)] == payload
    assert len(out) % cfg.depth == 0


def test_block_depth_1_identity():
    cfg = Config(depth=1, pad=0x00)
    payload = b"abcdef"
    assert tx(payload, cfg=cfg) == payload
    assert rx(payload, cfg=cfg) == payload


def test_block_rx_rejects_bad_length():
    cfg = Config(depth=8, pad=0x00)
    with pytest.raises(ValueError):
        rx(b"\x00" * 7, cfg=cfg)
