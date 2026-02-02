from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import struct
import zlib


# ============================
# Config
# ============================

@dataclass(frozen=True)
class Config:
    """
    Minimal config for length-based framing.

    You can replace this with your project's config object later,
    as long as it provides the same attributes.
    """
    magic: bytes = b"\xA5\xC3"     # 2 bytes
    version: int = 1              # 1 byte
    flags: int = 0                # 1 byte
    max_payload: int = 65535      # 2-byte length field, so max 65535
    seq: int = 0                  # 4 bytes, supplied by caller for determinism


# ============================
# Framing format
# ============================
#
# Frame:
#   MAGIC(2) | VER(1) | FLAGS(1) | LEN(2) | SEQ(4) | HDR_CRC16(2) | PAYLOAD(LEN) | CRC32(4)
#
# Notes:
# - HDR_CRC16 is CRC16-CCITT over: MAGIC..SEQ (i.e., first 2+1+1+2+4 = 10 bytes)
# - CRC32 is over: (VER..SEQ + PAYLOAD) OR (MAGIC..SEQ + PAYLOAD) — choose one and be consistent.
#   Here we do CRC32 over: (MAGIC..SEQ + PAYLOAD), excluding HDR_CRC16 and excluding CRC32 itself.
#   That makes it maximally strict and simplifies validation.

_HDR_FMT_NOCRC = ">2sBBHI"  # magic(2), ver(u8), flags(u8), len(u16), seq(u32)
_HDR_NOCRC_LEN = struct.calcsize(_HDR_FMT_NOCRC)  # 10 bytes
_HDR_CRC_FMT = ">H"  # u16
_HDR_TOTAL_LEN = _HDR_NOCRC_LEN + 2  # 12 bytes

_CRC32_FMT = ">I"
_CRC32_LEN = 4


# ============================
# Public API (uniform)
# ============================

def tx(data: bytes, *, cfg: Any) -> bytes:
    """
    TX direction: payload bytes -> framed bytes
    Uniform module API: tx(bytes, *, cfg) -> bytes
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("tx: data must be bytes-like")

    magic = _getattr_bytes(cfg, "magic", length=2)
    version = _getattr_int(cfg, "version", 0, 255)
    flags = _getattr_int(cfg, "flags", 0, 255)
    seq = _getattr_int(cfg, "seq", 0, 0xFFFFFFFF)
    max_payload = _getattr_int(cfg, "max_payload", 0, 65535)

    payload = bytes(data)
    if len(payload) > max_payload:
        raise ValueError(f"tx: payload too large: {len(payload)} > max_payload={max_payload}")

    hdr_no_crc = struct.pack(_HDR_FMT_NOCRC, magic, version, flags, len(payload), seq)
    hdr_crc = _crc16_ccitt(hdr_no_crc)
    hdr = hdr_no_crc + struct.pack(_HDR_CRC_FMT, hdr_crc)

    crc32_val = zlib.crc32(hdr_no_crc + payload) & 0xFFFFFFFF
    trailer = struct.pack(_CRC32_FMT, crc32_val)

    return hdr + payload + trailer


def rx(data: bytes, *, cfg: Any) -> bytes:
    """
    RX direction: framed bytes (single complete frame) -> payload bytes
    Uniform module API: rx(bytes, *, cfg) -> bytes

    IMPORTANT: This expects exactly ONE complete frame in `data`.
    Any streaming/buffering / magic scanning belongs in stage.py later.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("rx: data must be bytes-like")

    b = bytes(data)

    if len(b) < _HDR_TOTAL_LEN + _CRC32_LEN:
        raise ValueError("rx: frame too short")

    expected_magic = _getattr_bytes(cfg, "magic", length=2)
    max_payload = _getattr_int(cfg, "max_payload", 0, 65535)

    # Parse header
    hdr_no_crc = b[:_HDR_NOCRC_LEN]
    magic, version, flags, payload_len, seq = struct.unpack(_HDR_FMT_NOCRC, hdr_no_crc)

    if magic != expected_magic:
        raise ValueError("rx: bad magic")

    if payload_len > max_payload:
        raise ValueError(f"rx: payload_len too large: {payload_len} > max_payload={max_payload}")

    hdr_crc_got = struct.unpack(_HDR_CRC_FMT, b[_HDR_NOCRC_LEN:_HDR_TOTAL_LEN])[0]
    hdr_crc_exp = _crc16_ccitt(hdr_no_crc)
    if hdr_crc_got != hdr_crc_exp:
        raise ValueError("rx: header CRC16 failed")

    total_len = _HDR_TOTAL_LEN + payload_len + _CRC32_LEN
    if len(b) != total_len:
        raise ValueError(f"rx: frame length mismatch: got {len(b)} expected {total_len}")

    payload = b[_HDR_TOTAL_LEN:_HDR_TOTAL_LEN + payload_len]
    crc32_got = struct.unpack(_CRC32_FMT, b[-_CRC32_LEN:])[0]
    crc32_exp = zlib.crc32(hdr_no_crc + payload) & 0xFFFFFFFF
    if crc32_got != crc32_exp:
        raise ValueError("rx: payload CRC32 failed")

    return payload


# ============================
# Helpers
# ============================

def _getattr_int(cfg: Any, name: str, lo: int, hi: int) -> int:
    v = getattr(cfg, name, None)
    if v is None:
        raise AttributeError(f"cfg missing required int attribute: {name}")
    if not isinstance(v, int):
        raise TypeError(f"cfg.{name} must be int")
    if not (lo <= v <= hi):
        raise ValueError(f"cfg.{name} out of range [{lo},{hi}]")
    return v


def _getattr_bytes(cfg: Any, name: str, *, length: int) -> bytes:
    v = getattr(cfg, name, None)
    if v is None:
        raise AttributeError(f"cfg missing required bytes attribute: {name}")
    if not isinstance(v, (bytes, bytearray)):
        raise TypeError(f"cfg.{name} must be bytes-like")
    v = bytes(v)
    if len(v) != length:
        raise ValueError(f"cfg.{name} must be exactly {length} bytes")
    return v


def _crc16_ccitt(data: bytes, poly: int = 0x1021, init: int = 0xFFFF) -> int:
    """
    CRC-16/CCITT-FALSE style:
      - poly 0x1021
      - init 0xFFFF
      - no reflection
      - xorout 0x0000
    """
    crc = init
    for byte in data:
        crc ^= (byte << 8) & 0xFFFF
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc
