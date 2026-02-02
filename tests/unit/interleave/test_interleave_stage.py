import pytest

from rflink_modem_v2.pipeline.stages.interleave import stage as interleave_stage


@pytest.mark.parametrize("module_name", interleave_stage.available_modules())
def test_interleave_stage_roundtrip_all_modules(module_name: str):
    mod = interleave_stage._import_interleave_module(module_name)

    # Project invariant: module Config must be default-constructible
    try:
        module_cfg = mod.Config()
    except TypeError as e:
        pytest.fail(
            f"interleave module '{module_name}' Config() must be default-constructible. Error: {e}"
        )

    cfg = interleave_stage.Config(module=module_name, module_cfg=module_cfg)

    payload = b"interleave stage test payload" * 25
    enc = interleave_stage.tx(payload, cfg=cfg)
    dec = interleave_stage.rx(enc, cfg=cfg)

    # Both interleavers return padded length; original must be prefix.
    assert dec[:len(payload)] == payload
