from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ============================================================
# Uniform module API (canonical external functions)
#   - Config
#   - tx(bits, cfg) -> pcm
#   - rx(pcm, cfg) -> bits
#   - RX(cfg).rx(pcm) -> bits   (optional cache)
# ============================================================

@dataclass(frozen=True)
class Config:
    """
    AFSK/BFSK modem config.

    Constraints (v2 baseline):
      - sample_rate must be divisible by symbol_rate (perfect timing assumption).
      - mark/space are in Hz.
      - waveform is float32 mono PCM in ~[-amplitude, +amplitude].
    """
    sample_rate: int = 48_000
    symbol_rate: int = 1_200
    mark_hz: float = 2_200.0
    space_hz: float = 1_200.0
    amplitude: float = 0.8
    lead_silence_s: float = 0.0
    trail_silence_s: float = 0.0


def samples_per_symbol(cfg: Config) -> int:
    if cfg.sample_rate % cfg.symbol_rate != 0:
        raise ValueError(
            "sample_rate must be divisible by symbol_rate for the current AFSK modem. "
            f"Got sample_rate={cfg.sample_rate}, symbol_rate={cfg.symbol_rate}."
        )
    return cfg.sample_rate // cfg.symbol_rate


def _validate_bits(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8)
    if b.size == 0:
        raise ValueError("bits is empty")
    if np.any((b != 0) & (b != 1)):
        raise ValueError("bits must contain only 0/1")
    return b


# ----------------------------
# TX (canonical entrypoint)
# ----------------------------

def tx(bits: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Vectorized, phase-continuous AFSK/BFSK modulator.

    Input:  bits (0/1) as array-like
    Output: pcm float32 mono
    """
    b = _validate_bits(bits)
    fs = int(cfg.sample_rate)
    sps = samples_per_symbol(cfg)

    inc_mark = (2.0 * np.pi * float(cfg.mark_hz)) / float(fs)
    inc_space = (2.0 * np.pi * float(cfg.space_hz)) / float(fs)

    # Map each symbol to its phase increment and repeat for SPS samples.
    inc_sym = np.where(b == 1, inc_mark, inc_space).astype(np.float64)
    inc = np.repeat(inc_sym, sps)

    # Phase-continuous waveform by cumulative sum of increments.
    phase = np.cumsum(inc)
    if phase.size:
        phase -= inc[0]

    pcm = (np.sin(phase).astype(np.float32) * float(cfg.amplitude))

    lead_n = int(round(float(cfg.lead_silence_s) * fs))
    trail_n = int(round(float(cfg.trail_silence_s) * fs))
    if lead_n > 0 or trail_n > 0:
        pcm = np.concatenate(
            [np.zeros(lead_n, dtype=np.float32), pcm, np.zeros(trail_n, dtype=np.float32)]
        )

    return pcm


# ----------------------------
# RX internals
# ----------------------------

def _tone_refs(freq_hz: float, fs: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    t = np.arange(n, dtype=np.float32) / float(fs)
    w = 2.0 * np.pi * float(freq_hz)
    c = np.cos(w * t).astype(np.float32)
    s = np.sin(w * t).astype(np.float32)
    return c, s


def _strip_silence(pcm: np.ndarray, cfg: Config) -> np.ndarray:
    fs = int(cfg.sample_rate)
    lead_n = int(round(float(cfg.lead_silence_s) * fs))
    trail_n = int(round(float(cfg.trail_silence_s) * fs))

    if lead_n + trail_n >= pcm.size:
        raise ValueError("PCM too short after removing lead/trail silence")

    return pcm[lead_n : pcm.size - trail_n]


def _demod_body_to_bits(
    body: np.ndarray,
    cfg: Config,
    c_mark: np.ndarray,
    s_mark: np.ndarray,
    c_space: np.ndarray,
    s_space: np.ndarray,
) -> np.ndarray:
    sps = samples_per_symbol(cfg)

    n_sym = body.size // sps
    if n_sym == 0:
        raise ValueError("No full symbols to demodulate")

    body = body[: n_sym * sps]
    X = body.reshape(n_sym, sps)

    # Quadrature energy for mark
    I_m = X @ c_mark
    Q_m = X @ s_mark
    E_m = I_m * I_m + Q_m * Q_m

    # Quadrature energy for space
    I_s = X @ c_space
    Q_s = X @ s_space
    E_s = I_s * I_s + Q_s * Q_s

    return (E_m > E_s).astype(np.uint8)


# ----------------------------
# RX (canonical entrypoint)
# ----------------------------

def rx(pcm: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Demodulate AFSK/BFSK PCM into bits using per-symbol quadrature energy.
    Assumes perfect symbol timing and known lead/trail silence durations from cfg.
    Returns uint8 array of 0/1.
    """
    pcm = np.asarray(pcm, dtype=np.float32)
    fs = int(cfg.sample_rate)
    sps = samples_per_symbol(cfg)

    body = _strip_silence(pcm, cfg)

    c_mark, s_mark = _tone_refs(cfg.mark_hz, fs, sps)
    c_space, s_space = _tone_refs(cfg.space_hz, fs, sps)

    return _demod_body_to_bits(body, cfg, c_mark, s_mark, c_space, s_space)


# ----------------------------
# Cached RX (optional)
# ----------------------------

class RX:
    """
    Cached receiver to avoid rebuilding tone references every call.
    Uniform pattern: RX(cfg).rx(pcm) -> bits
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.fs = int(cfg.sample_rate)
        self.sps = samples_per_symbol(cfg)

        self.c_mark, self.s_mark = _tone_refs(cfg.mark_hz, self.fs, self.sps)
        self.c_space, self.s_space = _tone_refs(cfg.space_hz, self.fs, self.sps)

    def rx(self, pcm: np.ndarray) -> np.ndarray:
        pcm = np.asarray(pcm, dtype=np.float32)
        body = _strip_silence(pcm, self.cfg)
        return _demod_body_to_bits(body, self.cfg, self.c_mark, self.s_mark, self.c_space, self.s_space)


# ----------------------------
# Back-compat aliases (optional)
# ----------------------------

modulate = tx
modulate_bits = tx
demodulate = rx
demodulate_pcm = rx
Demodulator = RX
AFSKDemodulator = RX


# ----------------------------
# Optional WAV helpers (stage-driven)
# ----------------------------

def write_bits_wav(bits: np.ndarray, wav_path: str, cfg: Config) -> None:
    """
    Debug helper: bits -> PCM via AFSK TX, then WAV write via Containers stage (tx).
    Keeps this module reusable while respecting the container-stage boundary.
    """
    # Import locally to avoid hard dependency during pytest collection if containers isn’t wired yet.
    from rflink_modem_v2.pipeline.stages.containers.stage import tx as containers_tx
    from rflink_modem_v2.pipeline.stages.containers.config.stage_config import Config as ContainersCfg

    pcm = tx(bits, cfg)
    c_cfg = ContainersCfg(enabled=True, tx_path=wav_path, sample_rate=cfg.sample_rate)
    containers_tx(pcm, c_cfg)


def read_wav_bits(wav_path: str, cfg: Config) -> np.ndarray:
    """
    Debug helper: WAV read via Containers stage (rx), then PCM -> bits via AFSK RX.
    """
    from rflink_modem_v2.pipeline.stages.containers.stage import rx as containers_rx
    from rflink_modem_v2.pipeline.stages.containers.config.stage_config import Config as ContainersCfg

    c_cfg = ContainersCfg(enabled=True, rx_path=wav_path, sample_rate=cfg.sample_rate)
    pcm, fs = containers_rx(None, c_cfg)
    if int(fs) != int(cfg.sample_rate):
        raise ValueError(f"WAV fs={fs} does not match cfg.sample_rate={cfg.sample_rate}")
    return rx(pcm, cfg)
