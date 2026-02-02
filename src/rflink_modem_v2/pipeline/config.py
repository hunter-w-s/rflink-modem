from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rflink_modem_v2.pipeline.stages.sync.stage import Config as SyncStageConfig
from rflink_modem_v2.pipeline.stages.crypto.stage import Config as CryptoStageConfig
from rflink_modem_v2.pipeline.stages.framing.stage import Config as FramingStageConfig
from rflink_modem_v2.pipeline.stages.whiten.stage import Config as WhitenStageConfig
from rflink_modem_v2.pipeline.stages.interleave.stage import Config as InterleaveStageConfig
from rflink_modem_v2.pipeline.stages.blocking.stage import Config as BlockingStageConfig
from rflink_modem_v2.pipeline.stages.byte_fec.stage import Config as ByteFECStageConfig
from rflink_modem_v2.pipeline.stages.bit_fec.stage import Config as BitFECStageConfig
from rflink_modem_v2.pipeline.stages.phy.stage import Config as PhyStageConfig


@dataclass(frozen=True)
class PipelineConfig:
    """
    End-to-end modem pipeline configuration.

    Stages are composed in this order on TX:
      crypto -> framing -> sync -> whiten -> interleave -> blocking -> byte_fec -> bit_fec -> phy

    RX is the inverse:
      phy -> bit_fec -> byte_fec -> blocking -> deinterleave -> dewhiten -> sync -> framing -> crypto
    """
    sync: SyncStageConfig
    crypto: CryptoStageConfig
    framing: FramingStageConfig
    whiten: WhitenStageConfig
    interleave: InterleaveStageConfig
    blocking: BlockingStageConfig
    byte_fec: ByteFECStageConfig
    bit_fec: BitFECStageConfig
    phy: PhyStageConfig

    # If you later add bit-offset sync, this becomes useful.
    keep_partial_bits: bool = True
