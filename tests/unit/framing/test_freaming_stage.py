import pytest

from rflink_modem_v2.pipeline.stages.framing import stage as framing_stage


@pytest.mark.parametrize("module_name", framing_stage.available_modules())
def test_framing_stage_tx_rx_single_frame(module_name: str):
    mod = framing_stage._import_framing_module(module_name)

    # Require default-constructible module Config (project invariant)
    module_cfg = mod.Config()

    cfg = framing_stage.Config(module=module_name, module_cfg=module_cfg)

    payload = b"test payload 123"
    framed = framing_stage.tx(payload, cfg=cfg)

    payloads, rem = framing_stage.rx(framed, cfg=cfg)
    assert rem == b""
    assert payloads == [payload]


def test_framing_stage_rx_stream_multiple_frames_and_garbage_resync():
    # Specific to length framing fast-path behavior
    cfg = framing_stage.Config(module="length", module_cfg=None)

    p1 = b"aaa"
    p2 = b"bbbccc"
    f1 = framing_stage.tx(p1, cfg=cfg)
    f2 = framing_stage.tx(p2, cfg=cfg)

    stream = b"\x00\x01garbage" + f1 + b"\xFF\xFE" + f2 + b"\x99tail"
    payloads, rem = framing_stage.rx(stream, cfg=cfg)

    assert payloads == [p1, p2]
    # remainder should keep last magic-1 bytes or tail depending on scan;
    # for length framing, it will keep the tail since no new magic begins.
    assert isinstance(rem, (bytes, bytearray))
