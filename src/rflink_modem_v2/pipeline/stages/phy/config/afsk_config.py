"""
Best-known AFSK PHY stage config.

Selected from phy_limits_report.json using the lowest finite snr_threshold_db_est.
"""

from rflink_modem_v2.pipeline.stages.phy.stage import Config as PhyStageConfig
from rflink_modem_v2.pipeline.stages.phy.modules.afsk import Config as AFSKConfig

PHY_STAGE_CONFIG = PhyStageConfig(
    module="afsk",
    module_cfg=AFSKConfig(
        sample_rate=96000,
        symbol_rate=9600,
        mark_hz=18000.0,
        space_hz=6000.0,
        amplitude=0.8,
        lead_silence_s=0.0,
        trail_silence_s=0.0,
    ),
    # Keep this if your afsk PHY supports demod caching for faster tests.
    use_cached_demod=True,
)

# Optional alias (handy if your loader looks for multiple names)
phy_stage_config = PHY_STAGE_CONFIG
