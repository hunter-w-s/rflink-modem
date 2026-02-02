import pytest

from rflink_modem_v2.pipeline.stages.bit_fec import stage as bit_fec_stage


@pytest.mark.parametrize("module_name", bit_fec_stage.available_modules())
def test_bit_fec_stage_roundtrip_all_modules(module_name: str):
    mod = bit_fec_stage._import_bit_fec_module(module_name)

    # Project invariant: module Config must be default-constructible
    try:
        module_cfg = mod.Config()
    except TypeError as e:
        pytest.fail(
            f"bit_fec module '{module_name}' Config() must be default-constructible. Error: {e}"
        )

    cfg = bit_fec_stage.Config(module=module_name, module_cfg=module_cfg)

    payload = b"bit fec stage test payload" * 10
    enc = bit_fec_stage.tx(payload, cfg=cfg)
    dec = bit_fec_stage.rx(enc, cfg=cfg)

    # Bit-FEC module is designed to be exact for byte input after pad-bit handling.
    assert dec == payload
