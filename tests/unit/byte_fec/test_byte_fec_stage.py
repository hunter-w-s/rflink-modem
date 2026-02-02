import pytest

from rflink_modem_v2.pipeline.stages.byte_fec import stage as byte_fec_stage


@pytest.mark.parametrize("module_name", byte_fec_stage.available_modules())
def test_byte_fec_stage_roundtrip_all_modules(module_name: str):
    mod = byte_fec_stage._import_byte_fec_module(module_name)

    # Project invariant: module Config must be default-constructible
    try:
        module_cfg = mod.Config()
    except TypeError as e:
        pytest.fail(
            f"byte_fec module '{module_name}' Config() must be default-constructible. Error: {e}"
        )

    cfg = byte_fec_stage.Config(module=module_name, module_cfg=module_cfg)

    payload = b"byte fec stage test payload" * 10
    enc = byte_fec_stage.tx(payload, cfg=cfg)
    dec = byte_fec_stage.rx(enc, cfg=cfg)

    # Many byte-FECs return padded length; enforce prefix correctness.
    assert dec[:len(payload)] == payload
