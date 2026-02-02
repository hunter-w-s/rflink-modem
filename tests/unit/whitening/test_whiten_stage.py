import pytest

from rflink_modem_v2.pipeline.stages.whiten import stage as whiten_stage


@pytest.mark.parametrize("module_name", whiten_stage.available_modules())
def test_whiten_stage_roundtrip_all_modules(module_name: str):
    mod = whiten_stage._import_whiten_module(module_name)

    # Project invariant: module Config must be default-constructible
    try:
        module_cfg = mod.Config()
    except TypeError as e:
        pytest.fail(
            f"whiten module '{module_name}' Config() must be default-constructible. Error: {e}"
        )

    cfg = whiten_stage.Config(module=module_name, module_cfg=module_cfg)

    payload = b"\x00\x01\x02hello\xff\x10\x20" * 10
    w = whiten_stage.tx(payload, cfg=cfg)
    out = whiten_stage.rx(w, cfg=cfg)

    assert out == payload
