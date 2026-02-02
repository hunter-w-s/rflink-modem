"""
Best-known M=4 MFSK PHY stage config ("M4").

Selected from phy_limits_msfk_report.json using the lowest finite snr_threshold_db_est among M=4 entries.
NOTE: the report uses module name "msfk". If your registered module is "mfsk", change it below.
"""

from rflink_modem_v2.pipeline.stages.phy.stage import Config as PhyStageConfig

# Adjust import path/name to your actual module implementation.
# If your module is at ...phy.modules.mfsk, change the import accordingly.
from rflink_modem_v2.pipeline.stages.phy.modules.msfk import Config as MFSKConfig  # or .mfsk

PHY_STAGE_CONFIG = PhyStageConfig(
    module="msfk",  # change to "mfsk" if that's what your registry uses
    module_cfg=MFSKConfig(
        sample_rate=96000,
        symbol_rate=9600,
        tones_hz=(6000.0, 10000.0, 14000.0, 18000.0),
        amplitude=0.8,
        lead_silence_s=0.0,
        trail_silence_s=0.0,
    ),
    use_cached_demod=True,
)

phy_stage_config = PHY_STAGE_CONFIG
