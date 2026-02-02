from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math


@dataclass(frozen=True)
class Config:
    """
    Rectangular block interleaver.

    depth: number of rows in the matrix.
    pad: byte used to pad to a whole matrix.

    Behavior:
    - tx pads so total length is a multiple of depth
    - rx inverts, returning the padded length
    - trimming should be done by framing/stage using known payload length
    """
    depth: int = 8
    pad: int = 0x00


def tx(data: bytes, *, cfg: Any) -> bytes:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("tx: data must be bytes-like")

    depth = _get_depth(cfg)
    pad = _get_pad(cfg)

    b = bytes(data)
    if len(b) == 0:
        return b

    width = math.ceil(len(b) / depth)
    n = depth * width
    if n != len(b):
        b = b + bytes([pad]) * (n - len(b))

    # Write row-major into [depth][width], read column-major
    out = bytearray(n)
    k = 0
    for c in range(width):
        for r in range(depth):
            out[k] = b[r * width + c]
            k += 1
    return bytes(out)


def rx(data: bytes, *, cfg: Any) -> bytes:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("rx: data must be bytes-like")

    depth = _get_depth(cfg)

    b = bytes(data)
    if len(b) == 0:
        return b

    if len(b) % depth != 0:
        raise ValueError(f"rx: length {len(b)} not divisible by depth={depth}")

    width = len(b) // depth
    n = len(b)

    # Inverse: write column-major into [depth][width], read row-major
    out = bytearray(n)
    k = 0
    for c in range(width):
        for r in range(depth):
            out[r * width + c] = b[k]
            k += 1
    return bytes(out)


def _get_depth(cfg: Any) -> int:
    depth = getattr(cfg, "depth", None)
    if depth is None:
        raise AttributeError("cfg missing required int attribute: depth")
    if not isinstance(depth, int):
        raise TypeError("cfg.depth must be int")
    if depth <= 0:
        raise ValueError("cfg.depth must be > 0")
    return depth


def _get_pad(cfg: Any) -> int:
    pad = getattr(cfg, "pad", None)
    if pad is None:
        raise AttributeError("cfg missing required int attribute: pad")
    if not isinstance(pad, int):
        raise TypeError("cfg.pad must be int")
    if not (0 <= pad <= 255):
        raise ValueError("cfg.pad must be in [0,255]")
    return pad
