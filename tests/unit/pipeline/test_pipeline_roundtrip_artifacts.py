import hashlib
import importlib
import json
from pathlib import Path

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

# You already have a wav container stage/module in-tree; easiest is to just write wav directly.
# If your project already has a helper like wav_container.write_wav, use that.
import wave


def _write_wav_mono_16bit(path: Path, pcm_f32: np.ndarray, sample_rate: int) -> None:
    """Write float32 PCM in [-1,1] to 16-bit PCM WAV."""
    pcm = np.asarray(pcm_f32).reshape(-1)
    pcm_i16 = np.clip(pcm, -1.0, 1.0)
    pcm_i16 = (pcm_i16 * 32767.0).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm_i16.tobytes())


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _load_phy_stage_config(module_path: str) -> PhyStageConfig:
    m = importlib.import_module(module_path)
    for name in ("PHY_STAGE_CONFIG", "phy_stage_config", "CONFIG", "config"):
        if hasattr(m, name):
            cfg = getattr(m, name)
            if isinstance(cfg, PhyStageConfig):
                return cfg
    raise ImportError(f"No PhyStageConfig export found in {module_path}")


def _make_pipeline(phy_cfg: PhyStageConfig) -> Pipeline:
    crypto_cfg = CryptoStageConfig(
        module="chacha20poly1305",
        # deterministic for test artifacts; do NOT use fixed_nonce in real life
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


def _artifact_root(test_file: str, test_name: str, case_name: str) -> Path:
    # Mirrors your existing "test_results_unrev" style directory.
    # This path exists in your tree under tests/outputs/test_results_unrev/... :contentReference[oaicite:1]{index=1}
    safe_test_file = test_file.replace("/", "_").replace("\\", "_").replace(".", "_")
    safe_case = case_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    return Path("tests") / "outputs" / "test_results_unrev" / f"{safe_test_file}__{test_name}{safe_case}"


@pytest.mark.parametrize(
    "phy_module_path",
    [
        "rflink_modem_v2.pipeline.stages.phy.config.afsk_config",
        "rflink_modem_v2.pipeline.stages.phy.config.mfsk_config",
    ],
)
def test_roundtrip_artifacts_cwg(phy_module_path: str):
    phy_cfg = _load_phy_stage_config(phy_module_path)
    pipe = _make_pipeline(phy_cfg)

    # --- Load payload (cwg.png) ---
    payload_path = Path("tests") / "sample_assets" / "cwg.png"
    payload = payload_path.read_bytes()

    # --- TX ---
    pcm = pipe.tx(payload)
    sample_rate = int(phy_cfg.module_cfg.sample_rate)

    # --- RX (feed whole buffer; you can chunk this too) ---
    outs = pipe.rx(pcm)
    assert outs == [payload]
    reconstructed = outs[0]

    # --- Write artifacts ---
    case_name = f"[{phy_cfg.module}]"
    out_root = _artifact_root(__file__, "test_roundtrip_artifacts_cwg", case_name)
    tx_dir = out_root / "tx"
    rx_dir = out_root / "rx"
    tx_dir.mkdir(parents=True, exist_ok=True)
    rx_dir.mkdir(parents=True, exist_ok=True)

    # WAV + raw arrays
    _write_wav_mono_16bit(tx_dir / "00_pcm_tx.wav", pcm, sample_rate)
    np.save(tx_dir / "00_pcm_tx.npy", np.asarray(pcm, dtype=np.float32))

    _write_wav_mono_16bit(rx_dir / "00_pcm_rx.wav", pcm, sample_rate)
    np.save(rx_dir / "00_pcm_rx.npy", np.asarray(pcm, dtype=np.float32))

    # Payloads + hashes (viewable)
    (tx_dir / "original.png").write_bytes(payload)
    (rx_dir / "reconstructed.png").write_bytes(reconstructed)
    (tx_dir / "original.sha256").write_text(_sha256(payload) + "\n", encoding="utf-8")
    (rx_dir / "reconstructed.sha256").write_text(_sha256(reconstructed) + "\n", encoding="utf-8")

    # Manifest
    manifest = {
        "phy_module_path": phy_module_path,
        "phy_module": phy_cfg.module,
        "sample_rate": sample_rate,
        "pcm_len": int(len(pcm)),
        "payload_len": int(len(payload)),
        "payload_sha256": _sha256(payload),
        "reconstructed_sha256": _sha256(reconstructed),
        "paths": {
            "tx_wav": str((tx_dir / "00_pcm_tx.wav").as_posix()),
            "rx_wav": str((rx_dir / "00_pcm_rx.wav").as_posix()),
            "original_png": str((tx_dir / "original.png").as_posix()),
            "reconstructed_png": str((rx_dir / "reconstructed.png").as_posix()),
        },
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Helpful console line when run with -s
    print(f"[ARTIFACTS] wrote: {out_root}")
