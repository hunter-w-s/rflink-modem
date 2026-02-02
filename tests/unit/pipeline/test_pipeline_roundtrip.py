import importlib
from typing import Any

import numpy as np
import pytest

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


def _load_phy_stage_config(module_path: str) -> PhyStageConfig:
    """
    Import a PHY config module and return a PhyStageConfig.

    Supported export patterns (tries in order):
      1) A ready-to-use PhyStageConfig object: PHY_STAGE_CONFIG / phy_stage_config / PHY_CFG / phy_cfg / CONFIG / config
      2) A factory function returning PhyStageConfig: make_phy_stage_config / build_phy_stage_config / make / build / create

    If your module uses a different symbol name, add it below.
    """
    m = importlib.import_module(module_path)

    # Pattern 1: exported object
    candidate_objs = [
        "PHY_STAGE_CONFIG",
        "phy_stage_config",
        "PHY_CFG",
        "phy_cfg",
        "CONFIG",
        "config",
    ]
    for name in candidate_objs:
        if hasattr(m, name):
            cfg = getattr(m, name)
            if isinstance(cfg, PhyStageConfig):
                return cfg
            # Sometimes people export a dict-like or dataclass not typed as PhyStageConfig
            # but still meant to be passed in; fail loudly with a helpful message.
            raise TypeError(
                f"{module_path}.{name} exists but is {type(cfg)!r}, expected PhyStageConfig."
            )

    # Pattern 2: exported factory
    candidate_fns = [
        "make_phy_stage_config",
        "build_phy_stage_config",
        "create_phy_stage_config",
        "make",
        "build",
        "create",
        "get",
    ]
    for name in candidate_fns:
        if hasattr(m, name) and callable(getattr(m, name)):
            cfg = getattr(m, name)()
            if isinstance(cfg, PhyStageConfig):
                return cfg
            raise TypeError(
                f"{module_path}.{name}() returned {type(cfg)!r}, expected PhyStageConfig."
            )

    raise ImportError(
        f"Could not find a PhyStageConfig export in {module_path}. "
        f"Expected one of {candidate_objs} or one of factory functions {candidate_fns}."
    )


def _make_pipeline(phy_cfg: PhyStageConfig) -> Pipeline:
    # NOTE: fixed_nonce makes test deterministic; in real use fixed_nonce must be None.
    crypto_cfg = CryptoStageConfig(
        module="chacha20poly1305",
        module_cfg=CryptoModCfg(key=b"\x11" * 32, fixed_nonce=b"\x22" * 12, aad=b""),
    )

    cfg = PipelineConfig(
        sync=SyncStageConfig(
            module="word16",
            module_cfg=Word16Config(sync_word=0x2DD4, keep_sync=False),
        ),
        crypto=crypto_cfg,
        framing=FramingStageConfig(module="length", module_cfg=None),
        whiten=WhitenStageConfig(module="lfsr16", module_cfg=None),
        interleave=InterleaveStageConfig(
            module="block",
            module_cfg=BlockInterleaveCfg(depth=5, pad=0x00),
        ),
        blocking=BlockingStageConfig(module="fixed", module_cfg=None),
        byte_fec=ByteFECStageConfig(module="rs255", module_cfg=None),
        bit_fec=BitFECStageConfig(module="conv_k7_r12", module_cfg=None),
        phy=phy_cfg,
        keep_partial_bits=True,
    )

    return Pipeline(cfg)





@pytest.mark.parametrize(
    "phy_module_path",
    [
        "rflink_modem_v2.pipeline.stages.phy.config.afsk_config",
        "rflink_modem_v2.pipeline.stages.phy.config.mfsk_config",
    ],
)
def test_pipeline_roundtrip_golden(phy_module_path: str):
    phy_cfg = _load_phy_stage_config(phy_module_path)
    pipe = _make_pipeline(phy_cfg)

    payload = b"PIPELINE_CHUNKED_PCM_TEST_" * 10
    pcm = pipe.tx(payload)

    # Feed RX in chunks
    outs = []
    step = max(1, len(pcm) // 7)
    for i in range(0, len(pcm), step):
        outs.extend(pipe.rx(pcm[i : i + step]))

    assert outs == [payload]
