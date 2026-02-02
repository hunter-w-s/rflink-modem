from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import os

try:
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
except ImportError as e:  # pragma: no cover
    ChaCha20Poly1305 = None  # type: ignore


@dataclass(frozen=True)
class Config:
    """
    AEAD crypto using ChaCha20-Poly1305.

    key: 32 bytes
    aad: authenticated associated data (optional; empty is fine)
    fixed_nonce: if set, tx uses this nonce instead of random (ONLY for tests)
    """
    key: bytes
    aad: bytes = b""
    fixed_nonce: Optional[bytes] = None  # 12 bytes


def tx(data: bytes, *, cfg: Any) -> bytes:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("tx: data must be bytes-like")

    aead, aad, nonce = _get_aead_aad_nonce(cfg, for_tx=True)
    ct = aead.encrypt(nonce, bytes(data), aad)
    return nonce + ct


def rx(data: bytes, *, cfg: Any) -> bytes:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("rx: data must be bytes-like")

    aead, aad, _ = _get_aead_aad_nonce(cfg, for_tx=False)
    b = bytes(data)
    if len(b) < 12 + 16:  # nonce + tag minimum
        raise ValueError("rx: ciphertext too short")

    nonce = b[:12]
    ct = b[12:]
    try:
        pt = aead.decrypt(nonce, ct, aad)
    except Exception as e:
        raise ValueError("rx: authentication failed") from e
    return pt


def _get_aead_aad_nonce(cfg: Any, *, for_tx: bool):
    if ChaCha20Poly1305 is None:
        raise ImportError("cryptography is required for chacha20poly1305 module")

    key = getattr(cfg, "key", None)
    aad = getattr(cfg, "aad", None)
    fixed_nonce = getattr(cfg, "fixed_nonce", None)

    if not isinstance(key, (bytes, bytearray)) or len(key) != 32:
        raise ValueError("cfg.key must be 32 bytes")
    if not isinstance(aad, (bytes, bytearray)):
        raise TypeError("cfg.aad must be bytes-like")

    aead = ChaCha20Poly1305(bytes(key))

    if for_tx:
        if fixed_nonce is not None:
            if not isinstance(fixed_nonce, (bytes, bytearray)) or len(fixed_nonce) != 12:
                raise ValueError("cfg.fixed_nonce must be 12 bytes when set")
            nonce = bytes(fixed_nonce)
        else:
            nonce = os.urandom(12)  # MUST be unique per message
    else:
        nonce = b""  # unused on RX

    return aead, bytes(aad), nonce
