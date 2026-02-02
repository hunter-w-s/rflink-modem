from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Config:
    """
    Fixed-size blocking / rate matching.

    block_size: output length will be padded to a multiple of this.
    pad: byte used for padding.
    strict_rx: if True, rx() raises if len(data) not multiple of block_size.
    """
    block_size: int = 255  # sensible default for RS(255,k) ecosystems
    pad: int = 0x00
    strict_rx: bool = False


def tx(data: bytes, *, cfg: Any) -> bytes:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("tx: data must be bytes-like")

    block_size = _get_block_size(cfg)
    pad = _get_pad(cfg)

    b = bytes(data)
    if len(b) == 0:
        return b

    rem = len(b) % block_size
    if rem == 0:
        return b

    need = block_size - rem
    return b + bytes([pad]) * need


def rx(data: bytes, *, cfg: Any) -> bytes:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("rx: data must be bytes-like")

    block_size = _get_block_size(cfg)
    strict_rx = _get_strict_rx(cfg)

    b = bytes(data)
    if strict_rx and (len(b) % block_size != 0):
        raise ValueError(f"rx: length {len(b)} not multiple of block_size={block_size}")

    # Do NOT trim here (framing owns trimming).
    return b


def _get_block_size(cfg: Any) -> int:
    bs = getattr(cfg, "block_size", None)
    if bs is None:
        raise AttributeError("cfg missing required attribute: block_size")
    if not isinstance(bs, int):
        raise TypeError("cfg.block_size must be int")
    if bs <= 0:
        raise ValueError("cfg.block_size must be > 0")
    return bs


def _get_pad(cfg: Any) -> int:
    p = getattr(cfg, "pad", None)
    if p is None:
        raise AttributeError("cfg missing required attribute: pad")
    if not isinstance(p, int):
        raise TypeError("cfg.pad must be int")
    if not (0 <= p <= 255):
        raise ValueError("cfg.pad must be in [0,255]")
    return p


def _get_strict_rx(cfg: Any) -> bool:
    s = getattr(cfg, "strict_rx", None)
    if s is None:
        raise AttributeError("cfg missing required attribute: strict_rx")
    if not isinstance(s, bool):
        raise TypeError("cfg.strict_rx must be bool")
    return s
