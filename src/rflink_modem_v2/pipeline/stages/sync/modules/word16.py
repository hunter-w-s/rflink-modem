from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Config:
    """
    Byte-aligned 16-bit sync word insert/search.

    sync_word: 16-bit value inserted as big-endian bytes on TX.
    keep_sync: if True, RX returns stream starting at the sync word;
               if False, RX returns bytes immediately after the sync word.
    """
    sync_word: int = 0x2DD4  # common 16-bit sync/unique word value
    keep_sync: bool = False


def tx(data: bytes, *, cfg: Any) -> bytes:
    """
    Prepend sync word bytes to a byte stream.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("tx: data must be bytes-like")

    sync = _get_sync_bytes(cfg)
    return sync + bytes(data)


def rx(data: bytes, *, cfg: Any) -> bytes:
    """
    Find first occurrence of sync word on byte boundaries and return the
    aligned stream (either at sync or after it).

    Raises ValueError if sync not found.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("rx: data must be bytes-like")

    b = bytes(data)
    sync = _get_sync_bytes(cfg)
    keep_sync = _get_keep_sync(cfg)

    i = b.find(sync)
    if i < 0:
        raise ValueError("rx: sync not found")

    return b[i:] if keep_sync else b[i + len(sync):]


def _get_sync_bytes(cfg: Any) -> bytes:
    w = getattr(cfg, "sync_word", None)
    if w is None:
        raise AttributeError("cfg missing required attribute: sync_word")
    if not isinstance(w, int):
        raise TypeError("cfg.sync_word must be int")
    if not (0 <= w <= 0xFFFF):
        raise ValueError("cfg.sync_word must be in [0, 65535]")
    return int(w).to_bytes(2, "big")


def _get_keep_sync(cfg: Any) -> bool:
    ks = getattr(cfg, "keep_sync", None)
    if ks is None:
        raise AttributeError("cfg missing required attribute: keep_sync")
    if not isinstance(ks, bool):
        raise TypeError("cfg.keep_sync must be bool")
    return ks
