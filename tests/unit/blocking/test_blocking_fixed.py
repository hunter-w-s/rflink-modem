import pytest

from rflink_modem_v2.pipeline.stages.blocking import stage as blocking_stage
from rflink_modem_v2.pipeline.stages.blocking.modules.fixed import Config as FixedConfig


def test_blocking_fixed_pads_to_multiple():
    cfg = blocking_stage.Config(module="fixed", module_cfg=FixedConfig(block_size=16, pad=0xAA))
    payload = b"123456789"  # len 9 -> pad to 16
    out = blocking_stage.tx(payload, cfg=cfg)

    assert len(out) == 16
    assert out[:len(payload)] == payload
    assert out[len(payload):] == bytes([0xAA]) * (16 - len(payload))


def test_blocking_fixed_strict_rx_rejects_non_multiple():
    cfg = blocking_stage.Config(module="fixed", module_cfg=FixedConfig(block_size=16, strict_rx=True))
    with pytest.raises(ValueError):
        blocking_stage.rx(b"\x00" * 17, cfg=cfg)
