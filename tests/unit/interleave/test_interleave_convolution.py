import pytest

from rflink_modem_v2.pipeline.stages.interleave.modules.convolution import Config, tx, rx


def test_conv_roundtrip_exact_with_padding_trim():
    cfg = Config(depth=8, step=4, pad=0x00)
    payload = b"convolutional interleave test payload"  # len=37 (will pad to 40)

    y = tx(payload, cfg=cfg)
    out = rx(y, cfg=cfg)

    assert out[:len(payload)] == payload
    assert len(out) % cfg.depth == 0


def test_conv_deterministic_same_cfg_and_roundtrip():
    # depth=6, step=4 => gcd(5,6)=1 (invertible)
    cfg = Config(depth=6, step=4, pad=0x55)
    payload = bytes(range(200))  # will pad to multiple of 6

    y1 = tx(payload, cfg=cfg)
    y2 = tx(payload, cfg=cfg)
    assert y1 == y2

    out = rx(y1, cfg=cfg)
    assert out[:len(payload)] == payload
    assert len(out) % cfg.depth == 0


def test_conv_rejects_noninvertible_params():
    # depth=6, step=3 => gcd(4,6)=2 (non-invertible)
    cfg = Config(depth=6, step=3, pad=0x00)
    with pytest.raises(ValueError, match="gcd"):
        tx(b"abc", cfg=cfg)
