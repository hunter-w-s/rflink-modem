from __future__ import annotations

import numpy as np

from rflink_modem.containers.wav import read_wav_mono
from rflink_modem.modem.audio.afsk_tx import AFSKTxConfig, samples_per_symbol


def _tone_refs(freq_hz: float, fs: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build cosine/sine reference vectors for quadrature correlation.
    Using both makes detection phase-insensitive.
    """
    t = np.arange(n, dtype=np.float32) / float(fs)
    w = 2.0 * np.pi * float(freq_hz)
    c = np.cos(w * t).astype(np.float32)
    s = np.sin(w * t).astype(np.float32)
    return c, s


def demod_pcm_to_bits(pcm: np.ndarray, cfg: AFSKTxConfig) -> np.ndarray:
    """
    Demodulate a BFSK/AFSK PCM stream into bits using per-symbol quadrature energy.
    Assumes perfect symbol timing and known lead/trail silence durations from cfg.
    Returns uint8 array of 0/1.
    """
    fs = cfg.sample_rate
    sps = samples_per_symbol(cfg)

    # Skip configured silences (v0.1: assumes TX inserted them and nothing trimmed)
    lead_n = int(round(cfg.lead_silence_s * fs))
    trail_n = int(round(cfg.trail_silence_s * fs))

    if lead_n + trail_n >= pcm.size:
        raise ValueError("PCM too short after removing lead/trail silence")

    body = pcm[lead_n : pcm.size - trail_n]

    # Truncate to an integer number of symbols
    n_sym = body.size // sps
    if n_sym == 0:
        raise ValueError("No full symbols to demodulate")
    body = body[: n_sym * sps]

    # Reshape to (symbols, samples_per_symbol)
    X = body.reshape(n_sym, sps)

    # Reference tones
    c_mark, s_mark = _tone_refs(cfg.mark_hz, fs, sps)
    c_space, s_space = _tone_refs(cfg.space_hz, fs, sps)

    # Quadrature correlation energies (vectorized)
    # I = dot(X, cos), Q = dot(X, sin), energy = I^2 + Q^2
    I_m = X @ c_mark
    Q_m = X @ s_mark
    E_m = I_m * I_m + Q_m * Q_m

    I_s = X @ c_space
    Q_s = X @ s_space
    E_s = I_s * I_s + Q_s * Q_s

    bits = (E_m > E_s).astype(np.uint8)
    return bits


def demod_wav_to_bits(wav_path: str, cfg: AFSKTxConfig) -> np.ndarray:
    pcm, fs = read_wav_mono(wav_path)
    if fs != cfg.sample_rate:
        raise ValueError(f"WAV fs={fs} does not match cfg.sample_rate={cfg.sample_rate}")
    return demod_pcm_to_bits(pcm, cfg)
