import pytest

from rflink_modem_v2.pipeline.stages.crypto import stage as crypto_stage
from rflink_modem_v2.pipeline.stages.crypto.modules.chacha20poly1305 import Config as Cfg


def test_chacha20poly1305_auth_failure():
    cfg = crypto_stage.Config(
        module="chacha20poly1305",
        module_cfg=Cfg(key=b"\xAA" * 32, fixed_nonce=b"\xBB" * 12),
    )

    payload = b"attack at dawn"
    enc = bytearray(crypto_stage.tx(payload, cfg=cfg))
    enc[-1] ^= 0x01  # flip one bit

    with pytest.raises(ValueError, match="authentication failed"):
        crypto_stage.rx(bytes(enc), cfg=cfg)
