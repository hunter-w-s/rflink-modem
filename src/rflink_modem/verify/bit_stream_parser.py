# src/rflink_modem/verify/bit_stream_parser.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from rflink_modem.protocol.framing import HEADER_LEN, FrameHeader, frame_total_len, unpack_frame
from rflink_modem.verify.sync_search import bytes_to_bits, bits_to_bytes, find_sync_bits


@dataclass
class BitStreamStats:
    bits_in: int = 0
    frames_ok: int = 0
    resync_bit_skips: int = 0
    candidates_tried: int = 0
    drops_buffer_trim: int = 0


class BitFrameStreamParser:
    """
    Bit-level streaming parser that finds sync at arbitrary bit offsets and yields valid frames.

    Strategy:
      - Maintain a growing bit buffer.
      - Scan for sync_word (via find_sync_bits).
      - For each candidate start:
          * Ensure we can reconstruct HEADER_LEN bytes
          * Reconstruct header bytes, parse payload_len from header core
          * Ensure we can reconstruct full frame bytes
          * Call unpack_frame() to validate header CRC + payload CRC
          * If success: consume bits for full frame; emit frame
          * If fail: advance 1 bit and retry (hard resync)
    """

    def __init__(self, *, max_buffer_bits: int = 1_000_000):
        self._bits: List[int] = []
        self._max_buffer_bits = max_buffer_bits
        self.stats = BitStreamStats()

    def reset(self) -> None:
        self._bits.clear()

    def feed_bytes(self, data: bytes, *, msb_first: bool = True) -> List[Tuple[FrameHeader, bytes]]:
        return self.feed_bits(bytes_to_bits(data, msb_first=msb_first), msb_first=msb_first)

    def feed_bits(self, bits: Sequence[int], *, msb_first: bool = True) -> List[Tuple[FrameHeader, bytes]]:
        if not bits:
            return []

        # normalize to 0/1 ints
        self._bits.extend((b & 1) for b in bits)
        self.stats.bits_in += len(bits)

        # Prevent unbounded growth in pathological garbage streams.
        # Keep the newest tail; sync search works forward so this is safe-ish.
        if len(self._bits) > self._max_buffer_bits:
            drop = len(self._bits) - self._max_buffer_bits
            self.stats.drops_buffer_trim += drop
            del self._bits[:drop]

        out: List[Tuple[FrameHeader, bytes]] = []

        while True:
            # Need at least enough bits to cover a header (10 bytes)
            if len(self._bits) < (HEADER_LEN * 8):
                break

            hits = find_sync_bits(self._bits, msb_first=msb_first)
            if not hits:
                # Keep a small tail so sync spanning future bits can still match.
                # Sync is 16 bits, so keep last 15 bits.
                keep = 15
                if len(self._bits) > keep:
                    self.stats.resync_bit_skips += len(self._bits) - keep
                    del self._bits[:-keep]
                break

            start = hits[0]

            # Drop pre-sync bits
            if start > 0:
                self.stats.resync_bit_skips += start
                del self._bits[:start]

            # Now sync should be at bit offset 0
            if len(self._bits) < (HEADER_LEN * 8):
                break

            # Reconstruct header bytes (byte-aligned from this bit offset)
            header = bits_to_bytes(self._bits, bit_offset=0, n_bytes=HEADER_LEN, msb_first=msb_first)
            if len(header) != HEADER_LEN:
                break

            # Parse payload_len from header core bytes.
            # Header core layout: >HBBHH, payload_len is last u16 in the 8-byte core.
            payload_len = int.from_bytes(header[6:8], "big")
            total_len = frame_total_len(payload_len)
            total_bits = total_len * 8

            if len(self._bits) < total_bits:
                # Wait for more bits
                break

            frame_bytes = bits_to_bytes(self._bits, bit_offset=0, n_bytes=total_len, msb_first=msb_first)

            self.stats.candidates_tried += 1
            try:
                hdr, payload = unpack_frame(frame_bytes)
            except ValueError:
                # Not a valid frame at this bit position; advance by 1 bit and rescan
                self.stats.resync_bit_skips += 1
                del self._bits[:1]
                continue

            out.append((hdr, payload))
            self.stats.frames_ok += 1

            # Consume full frame bits
            del self._bits[:total_bits]

        return out
