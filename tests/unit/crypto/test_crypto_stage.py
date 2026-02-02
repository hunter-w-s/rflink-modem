import pytest

from rflink_modem_v2.pipeline.stages.crypto import stage as crypto_stage


@pytest.mark.parametrize("module_name", crypto_stage.available_modules())
def test_crypto_stage_roundtrip_all_modules(module_name: str):
    mod = crypto_stage._import_crypto_module(module_name)
    module_cfg = mod.Config(key=b"\x11" * 32, fixed_nonce=b"\x22" * 12)
    cfg = crypto_stage.Config(module=module_name, module_cfg=module_cfg)

    payload = b"crypto stage test payload" * 10
    enc = crypto_stage.tx(payload, cfg=cfg)
    dec = crypto_stage.rx(enc, cfg=cfg)
    assert dec == payload
