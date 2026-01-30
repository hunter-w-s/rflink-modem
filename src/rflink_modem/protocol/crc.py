# protocol/crc.py
from __future__ import annotations

from typing import Optional


def _reflect_bits(x: int, width: int) -> int:
    r = 0
    for _ in range(width):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def crc16_ccitt_false(data: bytes, *, init: int = 0xFFFF) -> int:
    """
    CRC-16/CCITT-FALSE
      width=16 poly=0x1021 init=0xFFFF refin=false refout=false xorout=0x0000
      Check("123456789") = 0x29B1
    """
    crc = init & 0xFFFF
    poly = 0x1021

    for b in data:
        crc ^= (b & 0xFF) << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def crc32_ieee(data: bytes, *, init: int = 0xFFFFFFFF) -> int:
    """
    CRC-32/ISO-HDLC (aka "IEEE 802.3")
      width=32 poly=0x04C11DB7 init=0xFFFFFFFF refin=true refout=true xorout=0xFFFFFFFF
      Check("123456789") = 0xCBF43926

    Implementation: reflected, bitwise (no table) for clarity & lock-in.
    """
    # Reflected polynomial for 0x04C11DB7 is 0xEDB88320
    poly = 0xEDB88320
    crc = init & 0xFFFFFFFF

    for b in data:
        crc ^= (b & 0xFF)
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
            crc &= 0xFFFFFFFF

    return (crc ^ 0xFFFFFFFF) & 0xFFFFFFFF


def crc16(data: bytes) -> int:
    """Convenience alias (current project default): CRC-16/CCITT-FALSE."""
    return crc16_ccitt_false(data)


def crc32(data: bytes) -> int:
    """Convenience alias (current project default): CRC-32/ISO-HDLC (IEEE)."""
    return crc32_ieee(data)
