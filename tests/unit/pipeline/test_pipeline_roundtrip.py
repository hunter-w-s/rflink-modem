import numpy as np

from rflink_modem_v2.pipeline.config import PipelineConfig
from rflink_modem_v2.pipeline.pipeline import Pipeline

from rflink_modem_v2.pipeline.stages.sync.stage import Config as SyncStageConfig
from rflink_modem_v2.pipeline.stages.sync.modules.word16 import Config as Word16Config

from rflink_modem_v2.pipeline.stages.crypto.stage import Config as CryptoStageConfig
from rflink_modem_v2.pipeline.stages.crypto.modules.chacha20poly1305 import Config as CryptoModCfg

from rflink_modem_v2.pipeline.stages.framing.stage import Config as FramingStageConfig
from rflink_modem_v2.pipeline.stages.whiten.stage import Config as WhitenStageConfig
from rflink_modem_v2.pipeline.stages.interleave.stage import Config as InterleaveStageConfig
from rflink_modem_v2.pipeline.stages.blocking.stage import Config as BlockingStageConfig
from rflink_modem_v2.pipeline.stages.byte_fec.stage import Config as ByteFECStageConfig
from rflink_modem_v2.pipeline.stages.bit_fec.stage import Config as BitFECStageConfig
from rflink_modem_v2.pipeline.stages.phy.stage import Config as PhyStageConfig
from rflink_modem_v2.pipeline.stages.interleave.modules.block import Config as BlockInterleaveCfg
from rflink_modem_v2.pipeline.stages.phy.modules.afsk import Config as AFSKCfg


def _make_pipeline() -> Pipeline:
    # NOTE: fixed_nonce makes test deterministic; in real use fixed_nonce must be None.
    crypto_cfg = CryptoStageConfig(
        module="chacha20poly1305",
        module_cfg=CryptoModCfg(key=b"\x11" * 32, fixed_nonce=b"\x22" * 12, aad=b""),
    )

    phy_cfg = PhyStageConfig(
        module="afsk",
        module_cfg=AFSKCfg(
            sample_rate=48000,
            symbol_rate=1200,
            mark_hz=2200.0,
            space_hz=1200.0,
            amplitude=0.8,
            lead_silence_s=0.0,
            trail_silence_s=0.0,
        ),
        use_cached_demod=True,
    )

    cfg = PipelineConfig(
        sync=SyncStageConfig(module="word16", module_cfg=Word16Config(sync_word=0x2DD4, keep_sync=False)),
        crypto=crypto_cfg,
        framing=FramingStageConfig(module="length", module_cfg=None),
        whiten=WhitenStageConfig(module="lfsr16", module_cfg=None),
        interleave=InterleaveStageConfig(module="block",module_cfg=BlockInterleaveCfg(depth=5, pad=0x00)),
        blocking=BlockingStageConfig(module="fixed", module_cfg=None),
        byte_fec=ByteFECStageConfig(module="rs255", module_cfg=None),
        bit_fec=BitFECStageConfig(module="conv_k7_r12", module_cfg=None),
        phy=phy_cfg,
        keep_partial_bits=True,
    )

    return Pipeline(cfg)


def test_pipeline_roundtrip_golden():
    pipe = _make_pipeline()

    payload = b"PIPELINE_GOLDEN_TEST_PAYLOAD_" * 5
    pcm = pipe.tx(payload)

    out = pipe.rx(pcm)
    assert out == [payload]


def test_pipeline_roundtrip_chunked_pcm():
    pipe = _make_pipeline()

    payload = b"PIPELINE_CHUNKED_PCM_TEST_" * 10
    pcm = pipe.tx(payload)

    # Feed RX in chunks
    outs = []
    step = max(1, len(pcm) // 7)
    for i in range(0, len(pcm), step):
        outs.extend(pipe.rx(pcm[i:i + step]))

    assert outs == [payload]
