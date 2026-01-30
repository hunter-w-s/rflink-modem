# src/rflink_modem/verify/stream_parser.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from rflink_modem.protocol.framing import (
    SYNC_WORD,
    VERSION,
    HEADER_LEN,
    FrameHeader,
    frame_total_len,
)
from rflink_modem.protocol.crc import crc16_ccitt_false, crc32_ieee


_SYNC_BYTES = SYNC_WORD.to_bytes(2, "big")


@dataclass
class StreamStats:
    bytes_in: int = 0
    frames_ok: int = 0
    frames_bad_header_crc: int = 0
    frames_bad_payload_crc: int = 0
    resync_skips: int = 0
    drops_too_short: int = 0


class FrameStreamParser:
    """
    Streaming parser for framed packets.

    Assumption: byte-aligned framing (sync is searched on byte boundaries).
    This matches typical modem pipelines where you recover bytes before framing.
    """

    def __init__(self, *, expect_sync: int = SYNC_WORD, expect_version: int = VERSION, max_payload: int = 0xFFFF):
        self._expect_sync = expect_sync
        self._expect_version = expect_version
        self._max_payload = max_payload
        self._buf = bytearray()
        self.stats = StreamStats()

    def reset(self) -> None:
        self._buf.clear()

    def feed(self, data: bytes) -> List[Tuple[FrameHeader, bytes]]:
        """
        Feed bytes into the parser. Returns list of decoded frames.
        """
        if not data:
            return []

        self.stats.bytes_in += len(data)
        self._buf.extend(data)

        out: List[Tuple[FrameHeader, bytes]] = []

        while True:
            # Need at least header to do anything meaningful
            if len(self._buf) < HEADER_LEN:
                break

            # Find sync (byte-aligned)
            idx = self._buf.find(_SYNC_BYTES)
            if idx < 0:
                # Keep last byte in case sync spans chunk boundary
                if len(self._buf) > 1:
                    self.stats.resync_skips += len(self._buf) - 1
                    self._buf[:] = self._buf[-1:]
                break

            if idx > 0:
                self.stats.resync_skips += idx
                del self._buf[:idx]

            if len(self._buf) < HEADER_LEN:
                break

            # Validate header CRC16 without fully unpacking via framing.unpack_frame
            core = bytes(self._buf[:8])  # header core is 8 bytes
            hdr_crc_recv = int.from_bytes(self._buf[8:10], "big")
            hdr_crc_calc = crc16_ccitt_false(core)
            if hdr_crc_calc != hdr_crc_recv:
                self.stats.frames_bad_header_crc += 1
                # resync by advancing one byte
                self.stats.resync_skips += 1
                del self._buf[0:1]
                continue

            # Parse header core
            # Same packing as framing: >HBBHH
            sync = int.from_bytes(core[0:2], "big")
            version = core[2]
            flags = core[3]
            seq = int.from_bytes(core[4:6], "big")
            payload_len = int.from_bytes(core[6:8], "big")

            if sync != self._expect_sync or version != self._expect_version:
                # shouldn't happen if sync matches, but keep it strict
                self.stats.frames_bad_header_crc += 1
                self.stats.resync_skips += 1
                del self._buf[0:1]
                continue

            if payload_len > self._max_payload:
                # bogus length; skip a byte and resync
                self.stats.resync_skips += 1
                del self._buf[0:1]
                continue

            need = frame_total_len(payload_len)
            if len(self._buf) < need:
                # Wait for more data
                break

            payload = bytes(self._buf[HEADER_LEN : HEADER_LEN + payload_len])
            payload_crc_recv = int.from_bytes(self._buf[HEADER_LEN + payload_len : need], "big")
            payload_crc_calc = crc32_ieee(payload)

            if payload_crc_calc != payload_crc_recv:
                self.stats.frames_bad_payload_crc += 1
                # resync by advancing one byte
                self.stats.resync_skips += 1
                del self._buf[0:1]
                continue

            hdr = FrameHeader(sync=sync, version=version, flags=flags, seq=seq, payload_len=payload_len)
            out.append((hdr, payload))
            self.stats.frames_ok += 1

            # consume this frame
            del self._buf[:need]

        return out
