from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from rflink_modem.containers.wav import write_wav_mono


@dataclass(frozen=True)
class AFSKTxConfig:
    sample_rate: int = 48000
    symbol_rate: int = 1200
    mark_hz: float = 2200.0
    space_hz: float = 1200.0
    amplitude: float = 0.8
    lead_silence_s: float = 0.25
    trail_silence_s: float = 0.25


def samples_per_symbol(cfg: AFSKTxConfig) -> int:
    if cfg.sample_rate % cfg.symbol_rate != 0:
        raise ValueError(
            f"sample_rate must be divisible by symbol_rate for v0.1. "
            f"Got {cfg.sample_rate=} and {cfg.symbol_rate=}."
        )
    return cfg.sample_rate // cfg.symbol_rate


def _phase_inc(freq_hz: float, fs: int) -> float:
    return 2.0 * np.pi * float(freq_hz) / float(fs)


def _osc_block(phase0: float, phase_inc: float, n: int) -> tuple[np.ndarray, float]:
    # Phase-continuous oscillator block.
    idx = np.arange(n, dtype=np.float64)
    phase = phase0 + phase_inc * idx
    block = np.sin(phase).astype(np.float32)

    # Keep phase bounded to avoid float growth over long runs.
    phase1 = float((phase0 + phase_inc * n) % (2.0 * np.pi))
    return block, phase1


def bits_to_pcm(bits: list[int] | np.ndarray, cfg: AFSKTxConfig) -> np.ndarray:
    """
    Convert a 0/1 bitstream into a phase-continuous BFSK/AFSK waveform (mono PCM float32).
    """
    sps = samples_per_symbol(cfg)

    bits_arr = np.asarray(bits, dtype=np.uint8)
    if bits_arr.size == 0:
        raise ValueError("bits is empty")
    if np.any((bits_arr != 0) & (bits_arr != 1)):
        raise ValueError("bits must contain only 0/1")

    mark_inc = _phase_inc(cfg.mark_hz, cfg.sample_rate)
    space_inc = _phase_inc(cfg.space_hz, cfg.sample_rate)

    out = np.empty(bits_arr.size * sps, dtype=np.float32)

    phase = 0.0
    w = 0
    for b in bits_arr:
        inc = mark_inc if b == 1 else space_inc
        block, phase = _osc_block(phase, inc, sps)
        out[w : w + sps] = block
        w += sps

    # Apply amplitude scaling (keep headroom for int16 conversion)
    out *= float(cfg.amplitude)

    # Add lead/trail silence
    lead_n = int(round(cfg.lead_silence_s * cfg.sample_rate))
    trail_n = int(round(cfg.trail_silence_s * cfg.sample_rate))
    if lead_n > 0 or trail_n > 0:
        out = np.concatenate(
            [np.zeros(lead_n, dtype=np.float32), out, np.zeros(trail_n, dtype=np.float32)]
        )

    return out


def bits_to_wav(bits: list[int] | np.ndarray, wav_path: str, cfg: AFSKTxConfig) -> None:
    pcm = bits_to_pcm(bits, cfg)
    write_wav_mono(wav_path, pcm, cfg.sample_rate)
