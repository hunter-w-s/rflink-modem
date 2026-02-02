from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

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
    n-MFSK modem config (coherent/quadrature energy detector).

    Design assumptions (same baseline as your AFSK module):
      - sample_rate must be divisible by symbol_rate (perfect timing).
      - Tones are known and fixed: cfg.tones_hz (length M).
      - M must be a power of two so symbols map cleanly to k=log2(M) bits.
      - Input bitstream length must be multiple of k (no implicit padding).
      - PCM is float32 mono in ~[-amplitude, +amplitude].

    Notes:
      - This is *orthogonal-ish* MFSK only if tone spacing is chosen sensibly
        (roughly Δf >= 1/Tsym for near-orthogonality), but we’re not enforcing it here.
    """
    sample_rate: int = 48_000
    symbol_rate: int = 1_200
    tones_hz: Tuple[float, ...] = (1200.0, 2200.0)  # M=2 default (BFSK-compatible)
    amplitude: float = 0.8
    lead_silence_s: float = 0.0
    trail_silence_s: float = 0.0


# ----------------------------
# Helpers
# ----------------------------

def samples_per_symbol(cfg: Config) -> int:
    if cfg.sample_rate % cfg.symbol_rate != 0:
        raise ValueError(
            "sample_rate must be divisible by symbol_rate for the current MSFK modem. "
            f"Got sample_rate={cfg.sample_rate}, symbol_rate={cfg.symbol_rate}."
        )
    return cfg.sample_rate // cfg.symbol_rate


def _validate_cfg(cfg: Config) -> Tuple[int, int]:
    tones = tuple(float(x) for x in cfg.tones_hz)
    if len(tones) < 2:
        raise ValueError("tones_hz must contain at least 2 tones")
    if len(set(tones)) != len(tones):
        raise ValueError("tones_hz must not contain duplicates")

    M = len(tones)

    # Require power of two so we can do k = log2(M) without fractional bits.
    if M & (M - 1) != 0:
        raise ValueError(f"M={M} tones is not a power of two")

    k = int(np.log2(M))
    if 2**k != M:
        raise ValueError("Internal error: log2(M) not integral")

    return M, k


def _validate_bits(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if b.size == 0:
        raise ValueError("bits is empty")
    if np.any((b != 0) & (b != 1)):
        raise ValueError("bits must contain only 0/1")
    return b


def _bits_to_symbols_msb(bits: np.ndarray, k: int) -> np.ndarray:
    """
    Pack bits (MSB-first within each k-bit symbol) -> uint16 symbol indices [0..M-1].
    Requires len(bits) % k == 0.
    """
    n = bits.size
    if n % k != 0:
        raise ValueError(f"bits length {n} is not a multiple of bits_per_symbol k={k}")

    bits = bits.reshape(-1, k).astype(np.uint8)
    # MSB-first: value = b0*2^(k-1) + ... + b(k-1)
    weights = (1 << np.arange(k - 1, -1, -1, dtype=np.uint16))
    sym = (bits.astype(np.uint16) * weights).sum(axis=1)
    return sym.astype(np.uint16)


def _symbols_to_bits_msb(sym: np.ndarray, k: int) -> np.ndarray:
    """
    Unpack uint16 symbol indices -> bits (MSB-first) length = len(sym)*k.
    """
    sym = np.asarray(sym, dtype=np.uint16).reshape(-1)
    out = np.empty(sym.size * k, dtype=np.uint8)
    o = 0
    for v in sym:
        for shift in range(k - 1, -1, -1):
            out[o] = (v >> shift) & 1
            o += 1
    return out


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


# ----------------------------
# TX (canonical entrypoint)
# ----------------------------

def tx(bits: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Vectorized, phase-continuous MFSK modulator.

    Input:  bits (0/1) as array-like
    Output: pcm float32 mono
    """
    M, k = _validate_cfg(cfg)
    b = _validate_bits(bits)

    sym = _bits_to_symbols_msb(b, k)  # [0..M-1]
    fs = int(cfg.sample_rate)
    sps = samples_per_symbol(cfg)

    tones = np.asarray(cfg.tones_hz, dtype=np.float64)
    freq_per_sym = tones[sym]                      # length n_sym
    inc_sym = (2.0 * np.pi * freq_per_sym) / fs    # phase increment per sample
    inc = np.repeat(inc_sym, sps)                  # per-sample increment

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

def _demod_body_to_symbols(
    body: np.ndarray,
    cfg: Config,
    c_refs: np.ndarray,
    s_refs: np.ndarray,
) -> np.ndarray:
    """
    body: float32, stripped of lead/trail, length N
    c_refs/s_refs: shape (M, sps)
    Returns: uint16 symbol indices length n_sym
    """
    sps = samples_per_symbol(cfg)

    n_sym = body.size // sps
    if n_sym == 0:
        raise ValueError("No full symbols to demodulate")

    body = body[: n_sym * sps]
    X = body.reshape(n_sym, sps)

    # Energy detector across M tones (vectorized).
    # I = X @ c.T  -> shape (n_sym, M) if c_refs is (M, sps)
    I = X @ c_refs.T
    Q = X @ s_refs.T
    E = I * I + Q * Q  # (n_sym, M)

    sym_hat = np.argmax(E, axis=1).astype(np.uint16)
    return sym_hat


# ----------------------------
# RX (canonical entrypoint)
# ----------------------------

def rx(pcm: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Demodulate MFSK PCM into bits using per-symbol quadrature energy across all tones.
    Assumes perfect symbol timing and known lead/trail silence durations from cfg.
    Returns uint8 array of 0/1.
    """
    M, k = _validate_cfg(cfg)
    pcm = np.asarray(pcm, dtype=np.float32)
    fs = int(cfg.sample_rate)
    sps = samples_per_symbol(cfg)

    body = _strip_silence(pcm, cfg)

    # Build tone references (M, sps)
    c_refs = np.empty((M, sps), dtype=np.float32)
    s_refs = np.empty((M, sps), dtype=np.float32)
    for i, f in enumerate(cfg.tones_hz):
        c, s = _tone_refs(f, fs, sps)
        c_refs[i] = c
        s_refs[i] = s

    sym_hat = _demod_body_to_symbols(body, cfg, c_refs, s_refs)
    return _symbols_to_bits_msb(sym_hat, k)


# ----------------------------
# Cached RX (optional)
# ----------------------------

class RX:
    """
    Cached receiver to avoid rebuilding tone references every call.
    Uniform pattern: RX(cfg).rx(pcm) -> bits
    """
    def __init__(self, cfg: Config):
        M, k = _validate_cfg(cfg)
        self.cfg = cfg
        self.M = M
        self.k = k
        self.fs = int(cfg.sample_rate)
        self.sps = samples_per_symbol(cfg)

        self.c_refs = np.empty((M, self.sps), dtype=np.float32)
        self.s_refs = np.empty((M, self.sps), dtype=np.float32)
        for i, f in enumerate(cfg.tones_hz):
            c, s = _tone_refs(f, self.fs, self.sps)
            self.c_refs[i] = c
            self.s_refs[i] = s

    def rx(self, pcm: np.ndarray) -> np.ndarray:
        pcm = np.asarray(pcm, dtype=np.float32)
        body = _strip_silence(pcm, self.cfg)
        sym_hat = _demod_body_to_symbols(body, self.cfg, self.c_refs, self.s_refs)
        return _symbols_to_bits_msb(sym_hat, self.k)


# ----------------------------
# Back-compat aliases (optional)
# ----------------------------

modulate = tx
demodulate = rx
Demodulator = RX
