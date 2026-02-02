from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math


@dataclass(frozen=True)
class Config:
    """
    Tail-biting convolutional *block* interleaver (length-preserving permutation)
    with padding to guarantee invertibility under a simple condition.

    Mapping (after padding length to N multiple of depth):
      out[(k + (k % depth) * step) % N] = in[k]

    Invertibility condition:
      N % depth == 0 AND gcd(step + 1, depth) == 1

    Behavior:
    - tx pads to a multiple of depth using pad byte
    - rx inverts and returns padded length
    - trimming belongs in framing/stage (later), same as block interleaver
    """
    depth: int = 8
    step: int = 4
    pad: int = 0x00


def tx(data: bytes, *, cfg: Any) -> bytes:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("tx: data must be bytes-like")

    depth, step, pad = _get_params(cfg)
    _validate_invertible(depth, step)

    b = bytes(data)
    if len(b) == 0:
        return b

    # Pad to multiple of depth (guarantees N % depth == 0)
    n0 = len(b)
    n = ((n0 + depth - 1) // depth) * depth
    if n != n0:
        b = b + bytes([pad]) * (n - n0)

    out = bytearray(n)

    for k in range(n):
        branch = k % depth
        idx = (k + branch * step) % n
        out[idx] = b[k]

    return bytes(out)


def rx(data: bytes, *, cfg: Any) -> bytes:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("rx: data must be bytes-like")

    depth, step, _pad = _get_params(cfg)
    _validate_invertible(depth, step)

    b = bytes(data)
    n = len(b)
    if n == 0:
        return b

    if n % depth != 0:
        raise ValueError(f"rx: length {n} not divisible by depth={depth}")

    out = bytearray(n)

    # Inverse:
    # tx wrote out[idx(k)] = in[k]
    # so in[k] = out[idx(k)] => out[k] = b[idx(k)]
    for k in range(n):
        branch = k % depth
        idx = (k + branch * step) % n
        out[k] = b[idx]

    return bytes(out)


def _get_params(cfg: Any) -> tuple[int, int, int]:
    depth = getattr(cfg, "depth", None)
    step = getattr(cfg, "step", None)
    pad = getattr(cfg, "pad", None)

    for name, v in (("depth", depth), ("step", step), ("pad", pad)):
        if v is None:
            raise AttributeError(f"cfg missing required attribute: {name}")

    if not isinstance(depth, int) or depth <= 0:
        raise ValueError("cfg.depth must be int > 0")
    if not isinstance(step, int) or step < 0:
        raise ValueError("cfg.step must be int >= 0")
    if not isinstance(pad, int) or not (0 <= pad <= 255):
        raise ValueError("cfg.pad must be in [0,255]")

    return depth, step, pad


def _validate_invertible(depth: int, step: int) -> None:
    g = math.gcd(step + 1, depth)
    if g != 1:
        raise ValueError(
            f"non-invertible params: gcd(step+1, depth) must be 1, got {g} "
            f"for depth={depth}, step={step}"
        )
