from rflink_modem_v2.pipeline.stages.sync import stage as sync_stage
from rflink_modem_v2.pipeline.stages.sync.modules.word16 import Config as Word16Config


def test_sync_word16_basic_roundtrip():
    """
    Explicit test for the word16 sync module.

    This mirrors the 'default module' tests used in other stages and ensures
    word16 remains stable even as additional sync modules are added.
    """
    module_cfg = Word16Config(
        sync_word=0x2DD4,
        keep_sync=False,
    )
    cfg = sync_stage.Config(module="word16", module_cfg=module_cfg)

    payload = b"HELLO_SYNC_WORLD" * 8
    tx_stream = sync_stage.tx(payload, cfg=cfg)

    # RX should strip sync and return payload
    aligned, rem = sync_stage.rx(tx_stream, cfg=cfg)

    assert rem == b""
    assert aligned == payload


def test_sync_word16_resync_with_leading_garbage():
    """
    word16 must resynchronize correctly when garbage precedes the sync word.
    """
    cfg = sync_stage.Config(module="word16", module_cfg=Word16Config())

    payload = b"PAYLOAD_AFTER_SYNC"
    tx_stream = sync_stage.tx(payload, cfg=cfg)

    rx_stream = b"\x00\xff\xaaGARBAGE" + tx_stream
    aligned, rem = sync_stage.rx(rx_stream, cfg=cfg)

    assert rem == b""
    assert aligned == payload


def test_sync_word16_split_across_chunks():
    """
    Verify correct behavior when the 16-bit sync word is split across RX calls.
    """
    cfg = sync_stage.Config(module="word16", module_cfg=Word16Config())

    payload = b"SPLIT_SYNC_TEST" * 5
    tx_stream = sync_stage.tx(payload, cfg=cfg)

    # Split stream so sync spans the boundary
    first_chunk = tx_stream[:1]
    second_chunk = tx_stream[1:]

    aligned1, rem1 = sync_stage.rx(first_chunk, cfg=cfg)
    assert aligned1 == b""
    assert rem1 == first_chunk  # keep last sync_len-1 bytes

    aligned2, rem2 = sync_stage.rx(rem1 + second_chunk, cfg=cfg)
    assert rem2 == b""
    assert aligned2 == payload


def test_sync_word16_keep_sync_flag():
    """
    If keep_sync=True, RX should return the stream starting at the sync word.
    """
    module_cfg = Word16Config(keep_sync=True)
    cfg = sync_stage.Config(module="word16", module_cfg=module_cfg)

    payload = b"KEEP_SYNC_TEST"
    tx_stream = sync_stage.tx(payload, cfg=cfg)

    aligned, rem = sync_stage.rx(tx_stream, cfg=cfg)

    sync_bytes = module_cfg.sync_word.to_bytes(2, "big")
    assert rem == b""
    assert aligned.startswith(sync_bytes)
    assert aligned[2:] == payload
