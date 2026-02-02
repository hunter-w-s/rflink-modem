from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Config:
    """
    XOR whitening using a 16-bit LFSR. Symmetric:
      rx(tx(x)) == x

    This is NOT crypto. It's for spectral/run-length properties.

    seed: non-zero 16-bit LFSR seed.
    """
    seed: int = 0xACE1  # must be non-zero 16-bit


def tx(data: bytes, *, cfg: Any) -> bytes:
    """
    TX direction: apply whitening to bytes.
    Uniform module API: tx(bytes, *, cfg) -> bytes
    """
    return _xor_whiten(data, seed=_get_seed(cfg))


def rx(data: bytes, *, cfg: Any) -> bytes:
    """
    RX direction: remove whitening from bytes.
    Whitening is symmetric, so this is identical to tx().
    Uniform module API: rx(bytes, *, cfg) -> bytes
    """
    return _xor_whiten(data, seed=_get_seed(cfg))


# ----------------------------
# Internal
# ----------------------------

def _get_seed(cfg: Any) -> int:
    seed = getattr(cfg, "seed", None)
    if seed is None:
        raise AttributeError("cfg missing required int attribute: seed")
    if not isinstance(seed, int):
        raise TypeError("cfg.seed must be int")
    if not (0 <= seed <= 0xFFFF) or seed == 0:
        raise ValueError("cfg.seed must be a non-zero 16-bit value")
    return seed & 0xFFFF


def _xor_whiten(data: bytes, *, seed: int) -> bytes:
    """
    Reference-compatible with prior implementation:
    - 16-bit LFSR
    - generate 1 pseudo-random byte per input byte (8 LFSR steps)
    - XOR PRN byte with input byte

    Taps implemented as: new_bit = bit15 ^ bit11 ^ bit4 ^ bit0
    (matches your previous reference).
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes-like")

    lfsr = seed & 0xFFFF
    out = bytearray(len(data))

    for i, b in enumerate(data):
        prn = 0
        for _ in range(8):
            bit15 = (lfsr >> 15) & 1
            bit11 = (lfsr >> 11) & 1
            bit4 = (lfsr >> 4) & 1
            bit0 = lfsr & 1
            new_bit = bit15 ^ bit11 ^ bit4 ^ bit0

            lfsr = ((lfsr << 1) & 0xFFFF) | new_bit
            prn = ((prn << 1) | (lfsr & 1)) & 0xFF

        out[i] = (b ^ prn) & 0xFF

    return bytes(out)
