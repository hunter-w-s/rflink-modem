from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import pytest

from rflink_modem_v2.pipeline.stages.phy import stage as phy_stage


def _snr_db_to_sigma(signal: np.ndarray, snr_db: float) -> float:
    x = np.asarray(signal, dtype=np.float32)
    p_sig = float(np.mean(x * x)) + 1e-12
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    return float(np.sqrt(p_noise))


def _awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    sigma = _snr_db_to_sigma(x, snr_db)
    n = rng.normal(loc=0.0, scale=sigma, size=x.shape).astype(np.float32)
    return x + n


def _symbols_to_bits_msb(sym: np.ndarray, k_bits: int) -> np.ndarray:
    sym = np.asarray(sym, dtype=np.uint16).reshape(-1)
    out = np.empty(sym.size * k_bits, dtype=np.uint8)
    o = 0
    for v in sym:
        for shift in range(k_bits - 1, -1, -1):
            out[o] = (v >> shift) & 1
            o += 1
    return out


def make_blocky_symbols_m4(block_syms: int = 120) -> np.ndarray:
    """
    Listening-oriented M=4 dataset:
      - Long steady blocks: 0,1,2,3 (easy to hear each tone)
      - Then structured transitions: pairs and ramps
    """
    seq: List[int] = []

    # long steady blocks per tone (very audible)
    for s in range(4):
        seq.extend([s] * block_syms)

    # pair alternations (audible switching)
    seq.extend([0, 1] * (block_syms // 2))
    seq.extend([0, 2] * (block_syms // 2))
    seq.extend([0, 3] * (block_syms // 2))
    seq.extend([1, 2] * (block_syms // 2))
    seq.extend([2, 3] * (block_syms // 2))

    # 0-1-2-3 ramps (clearly cycles through tones)
    seq.extend([0, 1, 2, 3] * (block_syms // 2))

    return np.asarray(seq, dtype=np.uint16)


def make_blocky_symbols_m8(block_syms: int = 120) -> np.ndarray:
    """
    Listening-oriented M=8 dataset:
      - Long steady blocks: 0..7 (easy to hear each tone)
      - Pair alternations (audible switching)
      - Ramps / cycles across all tones
    """
    seq: List[int] = []

    for s in range(8):
        seq.extend([s] * block_syms)

    pairs = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (4, 5), (5, 6), (6, 7),
        (0, 7), (0, 4), (2, 6),
    ]
    for a, b in pairs:
        seq.extend([a, b] * (block_syms // 2))

    seq.extend(list(range(8)) * (block_syms // 2))
    seq.extend(list(range(7, -1, -1)) * (block_syms // 2))

    return np.asarray(seq, dtype=np.uint16)



@pytest.mark.characterization
def test_msfk_listen_m8_fixed_settings(tmp_path: Path):
    """
    8-MFSK listening test that is ALSO decodable with current RX.

    Important: with your current correlator RX, M=8 at SPS=10 and tight spacing
    often fails even at high SNR due to leakage + more hypotheses.

    So for this listen test we lower symbol_rate (increase SPS) so we can use
    ~orthogonal spacing (Δf ≈ 1/Tsym = symbol_rate) within an audible band.
    """
    if os.getenv("RUN_CHAR", "0") != "1":
        pytest.skip("Set RUN_CHAR=1 to run characterization/listening tests")

    module_name = "msfk"
    mod = phy_stage._import_phy_module(module_name)

    # ---- Fixed "listening" settings that decode reliably ----
    # Fs=96 kHz, Rsym=1200 => SPS=80
    # Choose spacing ≈ symbol_rate (orthogonal-ish over one symbol)
    sample_rate = 96_000
    symbol_rate = 1_200         # <- changed from 9600 (SPS goes 10 -> 80)
    spacing_hz = 1_200.0        # Δf ~ 1/Tsym

    # 8 tones in a comfortable audible band: 4.2k .. 12.6k
    tones_hz = tuple(4200.0 + spacing_hz * i for i in range(8))

    cfg = mod.Config(
        sample_rate=sample_rate,
        symbol_rate=symbol_rate,
        tones_hz=tones_hz,
        amplitude=0.8,
        lead_silence_s=0.25,
        trail_silence_s=0.25,
    )

    # Deterministic “audible” symbol sequence for M=8
    sym = make_blocky_symbols_m8(block_syms=200)     # length knob
    bits = _symbols_to_bits_msb(sym, k_bits=3)       # M=8 => k=3

    # Modulate
    stage_cfg = phy_stage.Config(module=module_name, module_cfg=cfg, use_cached_demod=False)
    pcm = phy_stage.tx(bits, cfg=stage_cfg)

    # Add noise (keep it clean to the ear)
    snr_db = 40.0
    noisy = _awgn(pcm, snr_db, rng=np.random.default_rng(999))

    # Sanity check: should decode perfectly at this SNR
    decoded = phy_stage.rx(noisy, cfg=stage_cfg)
    assert np.array_equal(decoded, bits), "Listen WAV should decode cleanly at high SNR"

    # Write WAV
    listen_dir = Path("tests/output/listen")
    listen_dir.mkdir(parents=True, exist_ok=True)

    wav_path = listen_dir / "msfk_listen_m8_fs96000_sr1200_tones4p2-12p6k_step1p2k_snr40.wav"

    try:
        from rflink_modem_v2.pipeline.stages.containers.stage import tx as containers_tx
        from rflink_modem_v2.pipeline.stages.containers.config.stage_config import Config as ContainersCfg

        c_cfg = ContainersCfg(enabled=True, tx_path=str(wav_path), sample_rate=int(cfg.sample_rate))
        containers_tx(noisy, c_cfg)
    except Exception:
        import wave

        x = np.clip(noisy, -1.0, 1.0)
        i16 = (x * 32767.0).astype(np.int16)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(cfg.sample_rate))
            wf.writeframes(i16.tobytes())

    print(f"\nWrote listen WAV: {wav_path}\n")



@pytest.mark.characterization
def test_msfk_listen_m4_fixed_settings(tmp_path: Path):
    """
    Generates a WAV for MSFK M=4 using the user's specific settings:
      Fs=96000, Rsym=9600 (SPS=10), tones=[6000,10000,14000,18000], amp=0.8

    Skips unless RUN_CHAR=1 so it doesn't run in CI.
    Writes:
      tests/output/listen/msfk_listen_m4_fixed_*.wav
    """
    if os.getenv("RUN_CHAR", "0") != "1":
        pytest.skip("Set RUN_CHAR=1 to run characterization/listening tests")

    module_name = "msfk"
    mod = phy_stage._import_phy_module(module_name)

    # ---- Fixed settings (from your JSON) ----
    cfg = mod.Config(
        sample_rate=96_000,
        symbol_rate=9_600,  # SPS=10
        tones_hz=(6000.0, 10_000.0, 14_000.0, 18_000.0),
        amplitude=0.8,
        lead_silence_s=0.25,   # small lead/trail makes playback nicer
        trail_silence_s=0.25,
    )

    # Deterministic “audible” symbol sequence for M=4
    sym = make_blocky_symbols_m4(block_syms=2000)  # change this to alter length
    bits = _symbols_to_bits_msb(sym, k_bits=2)    # M=4 => k=2

    # Modulate
    stage_cfg = phy_stage.Config(module=module_name, module_cfg=cfg, use_cached_demod=False)
    pcm = phy_stage.tx(bits, cfg=stage_cfg)

    # Add noise (for listening, use a high SNR floor so tones are obvious)
    # If you want “just decodable” instead, set snr_db ~ 16-18.
    snr_db = 40.0
    noisy = _awgn(pcm, snr_db, rng=np.random.default_rng(999))

    # Optional sanity check: it should decode perfectly at this SNR
    decoded = phy_stage.rx(noisy, cfg=stage_cfg)
    assert np.array_equal(decoded, bits), "Listen WAV should decode cleanly at high SNR"

    # Write WAV
    listen_dir = Path("tests/output/listen")
    listen_dir.mkdir(parents=True, exist_ok=True)

    wav_path = listen_dir / "msfk_listen_m4_fixed_fs96000_sr9600_tones6-10-14-18k_snr40.wav"

    # Prefer your container stage WAV writer; fall back to stdlib wave if import differs
    try:
        from rflink_modem_v2.pipeline.stages.containers.stage import tx as containers_tx
        from rflink_modem_v2.pipeline.stages.containers.config.stage_config import Config as ContainersCfg

        c_cfg = ContainersCfg(enabled=True, tx_path=str(wav_path), sample_rate=int(cfg.sample_rate))
        containers_tx(noisy, c_cfg)
    except Exception:
        import wave

        x = np.clip(noisy, -1.0, 1.0)
        i16 = (x * 32767.0).astype(np.int16)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(cfg.sample_rate))
            wf.writeframes(i16.tobytes())

    print(f"\nWrote listen WAV: {wav_path}\n")
