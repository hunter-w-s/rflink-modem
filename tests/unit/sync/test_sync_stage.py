import pytest

from rflink_modem_v2.pipeline.stages.sync import stage as sync_stage


@pytest.mark.parametrize("module_name", sync_stage.available_modules())
def test_sync_stage_roundtrip_all_modules(module_name: str):
    mod = sync_stage._import_sync_module(module_name)

    # Project invariant: module Config must be default-constructible
    try:
        module_cfg = mod.Config()
    except TypeError as e:
        pytest.fail(f"sync module '{module_name}' Config() must be default-constructible. Error: {e}")

    cfg = sync_stage.Config(module=module_name, module_cfg=module_cfg)

    payload = b"SYNC_PAYLOAD_123" * 10
    framed = sync_stage.tx(payload, cfg=cfg)

    # Add garbage in front; RX must align and recover payload stream
    aligned, rem = sync_stage.rx(b"\x00\x01garbage" + framed, cfg=cfg)
    assert rem == b""
    assert aligned == payload


def test_sync_stage_split_sync_across_chunks():
    cfg = sync_stage.Config(module="word16", module_cfg=None)

    payload = b"abcdef" * 20
    framed = sync_stage.tx(payload, cfg=cfg)

    # Split so first chunk ends with the first byte of the 2-byte sync
    first = framed[:1]
    second = framed[1:]

    aligned1, rem1 = sync_stage.rx(first, cfg=cfg)
    assert aligned1 == b""
    assert rem1 == first  # keeps last sync_len-1 bytes

    aligned2, rem2 = sync_stage.rx(rem1 + second, cfg=cfg)
    assert rem2 == b""
    assert aligned2 == payload
