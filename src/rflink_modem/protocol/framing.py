# src/rflink_modem/protocol/framing.py
from __future__ import annotations

import struct
from dataclasses import dataclass

from .crc import crc16_ccitt_false, crc32_ieee
# --- add near top of src/rflink_modem/protocol/framing.py ---
from .whitening import whiten

FLAG_WHITENED = 0x01
SYNC_WORD = 0xA55A
VERSION = 1

# Big-endian:
# sync:u16, version:u8, flags:u8, seq:u16, payload_len:u16  => 8 bytes
_HEADER_CORE_STRUCT = struct.Struct(">HBBHH")
# hdr_crc16:u16 => 2 bytes
_HEADER_CRC_STRUCT = struct.Struct(">H")
# payload_crc32:u32 => 4 bytes
_PAYLOAD_CRC_STRUCT = struct.Struct(">I")

HEADER_LEN = _HEADER_CORE_STRUCT.size + _HEADER_CRC_STRUCT.size  # 10


@dataclass(frozen=True, slots=True)
class FrameHeader:
    sync: int = SYNC_WORD
    version: int = VERSION
    flags: int = 0
    seq: int = 0
    payload_len: int = 0

    def pack_core(self) -> bytes:
        if not (0 <= self.sync <= 0xFFFF):
            raise ValueError("sync out of range")
        if not (0 <= self.version <= 0xFF):
            raise ValueError("version out of range")
        if not (0 <= self.flags <= 0xFF):
            raise ValueError("flags out of range")
        if not (0 <= self.seq <= 0xFFFF):
            raise ValueError("seq out of range")
        if not (0 <= self.payload_len <= 0xFFFF):
            raise ValueError("payload_len out of range")

        return _HEADER_CORE_STRUCT.pack(
            self.sync,
            self.version,
            self.flags,
            self.seq,
            self.payload_len,
        )

    @staticmethod
    def unpack_core(b: bytes) -> "FrameHeader":
        if len(b) != _HEADER_CORE_STRUCT.size:
            raise ValueError("header core wrong length")
        sync, version, flags, seq, payload_len = _HEADER_CORE_STRUCT.unpack(b)
        return FrameHeader(sync=sync, version=version, flags=flags, seq=seq, payload_len=payload_len)


def pack_frame(
    payload: bytes,
    *,
    flags: int = 0,
    seq: int = 0,
    version: int = VERSION,
    sync: int = SYNC_WORD,
    whiten_payload: bool = False,
    whiten_seed: int = 0xACE1,
) -> bytes:
    payload_len = len(payload)
    if payload_len > 0xFFFF:
        raise ValueError("payload too large")

    tx_payload = payload
    tx_flags = flags & 0xFF
    if whiten_payload:
        tx_payload = whiten(payload, seed=whiten_seed)
        tx_flags |= FLAG_WHITENED

    hdr = FrameHeader(sync=sync, version=version, flags=tx_flags, seq=seq, payload_len=payload_len)
    core = hdr.pack_core()

    hdr_crc = crc16_ccitt_false(core)
    header_bytes = core + _HEADER_CRC_STRUCT.pack(hdr_crc)

    payload_crc = crc32_ieee(tx_payload)
    return header_bytes + tx_payload + _PAYLOAD_CRC_STRUCT.pack(payload_crc)



def unpack_frame(
    frame: bytes,
    *,
    expect_version: int = VERSION,
    expect_sync: int = SYNC_WORD,
) -> tuple[FrameHeader, bytes]:
    if len(frame) < HEADER_LEN + _PAYLOAD_CRC_STRUCT.size:
        raise ValueError("frame too short")

    core = frame[: _HEADER_CORE_STRUCT.size]
    (hdr_crc_recv,) = _HEADER_CRC_STRUCT.unpack(frame[_HEADER_CORE_STRUCT.size : HEADER_LEN])

    hdr_crc_calc = crc16_ccitt_false(core)
    if hdr_crc_calc != hdr_crc_recv:
        raise ValueError("bad header crc16")

    hdr = FrameHeader.unpack_core(core)

    if hdr.sync != expect_sync:
        raise ValueError("bad sync")
    if hdr.version != expect_version:
        raise ValueError("bad version")

    total_len = frame_total_len(hdr.payload_len)
    if len(frame) != total_len:
        raise ValueError("frame length mismatch")

    payload = frame[HEADER_LEN : HEADER_LEN + hdr.payload_len]
    (payload_crc_recv,) = _PAYLOAD_CRC_STRUCT.unpack(frame[HEADER_LEN + hdr.payload_len : total_len])
    payload_crc_calc = crc32_ieee(payload)
    if payload_crc_calc != payload_crc_recv:
        raise ValueError("bad payload crc32")

    # If whitened, dewhiten for the caller AFTER CRC passes
    if hdr.flags & FLAG_WHITENED:
        payload = whiten(payload)  # default seed matches pack_frame default

    return hdr, payload



def frame_total_len(payload_len: int) -> int:
    if not (0 <= payload_len <= 0xFFFF):
        raise ValueError("payload_len out of range")
    return HEADER_LEN + payload_len + _PAYLOAD_CRC_STRUCT.size


def header_struct() -> struct.Struct:
    """Expose the struct used for the header core (tooling/tests)."""
    return _HEADER_CORE_STRUCT
