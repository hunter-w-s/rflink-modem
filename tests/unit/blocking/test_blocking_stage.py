import pytest

from rflink_modem_v2.pipeline.stages.blocking import stage as blocking_stage


@pytest.mark.parametrize("module_name", blocking_stage.available_modules())
def test_blocking_stage_roundtrip_all_modules(module_name: str):
    mod = blocking_stage._import_blocking_module(module_name)

    # project invariant
    module_cfg = mod.Config()

    cfg = blocking_stage.Config(module=module_name, module_cfg=module_cfg)

    payload = b"blocking stage test payload" * 31
    out = blocking_stage.tx(payload, cfg=cfg)

    # Must be padded to some multiple, but original must be prefix
    assert out[:len(payload)] == payload

    back = blocking_stage.rx(out, cfg=cfg)
    assert back == out
