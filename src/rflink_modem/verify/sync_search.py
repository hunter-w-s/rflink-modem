# src/rflink_modem/verify/sync_search.py
from __future__ import annotations

from typing import List, Sequence

from rflink_modem.protocol.framing import SYNC_WORD


def bytes_to_bits(data: bytes, *, msb_first: bool = True) -> List[int]:
    bits: List[int] = []
    for b in data:
        if msb_first:
            for i in range(7, -1, -1):
                bits.append((b >> i) & 1)
        else:
            for i in range(8):
                bits.append((b >> i) & 1)
    return bits


def bits_to_bytes(
    bits: Sequence[int],
    *,
    bit_offset: int = 0,
    n_bytes: int | None = None,
    msb_first: bool = True,
) -> bytes:
    """
    Reassemble bytes from a bits sequence starting at an arbitrary bit offset.

    If n_bytes is None, consume as many full bytes as possible.
    """
    if bit_offset < 0:
        raise ValueError("bit_offset < 0")
    if bit_offset > len(bits):
        return b""

    remaining = len(bits) - bit_offset
    max_full_bytes = remaining // 8

    if n_bytes is None:
        n_bytes = max_full_bytes
    if n_bytes < 0:
        raise ValueError("n_bytes < 0")
    if n_bytes > max_full_bytes:
        raise ValueError("not enough bits for requested n_bytes")

    out = bytearray()
    idx = bit_offset

    for _ in range(n_bytes):
        v = 0
        if msb_first:
            for _i in range(8):
                v = (v << 1) | (bits[idx] & 1)
                idx += 1
        else:
            for shift in range(8):
                v |= (bits[idx] & 1) << shift
                idx += 1
        out.append(v & 0xFF)

    return bytes(out)


def _word_to_bits(word: int, width: int, *, msb_first: bool = True) -> List[int]:
    if width <= 0:
        raise ValueError("width must be positive")
    if not (0 <= word < (1 << width)):
        raise ValueError("word out of range for width")

    out: List[int] = []
    if msb_first:
        for i in range(width - 1, -1, -1):
            out.append((word >> i) & 1)
    else:
        for i in range(width):
            out.append((word >> i) & 1)
    return out


def find_sync_bits(
    bits: Sequence[int],
    *,
    sync_word: int = SYNC_WORD,
    sync_width: int = 16,
    msb_first: bool = True,
) -> List[int]:
    """
    Bit-level scan for sync_word. Returns all bit indices where sync starts.
    """
    pattern = _word_to_bits(sync_word, sync_width, msb_first=msb_first)

    n = len(bits)
    m = len(pattern)
    if n < m:
        return []

    hits: List[int] = []
    # O(n*m) scan (fine for typical modem buffers; optimize later if needed)
    for i in range(0, n - m + 1):
        ok = True
        for j in range(m):
            if (bits[i + j] & 1) != pattern[j]:
                ok = False
                break
        if ok:
            hits.append(i)

    return hits
