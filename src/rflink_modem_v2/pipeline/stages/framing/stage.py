from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple
import importlib
import pkgutil


@dataclass(frozen=True)
class Config:
    """
    Framing stage config.

    module: framing module name (e.g. "length")
    module_cfg: instance of that module's Config (or None -> defaults)
    """
    module: str = "length"
    module_cfg: Any = None


def available_modules() -> list[str]:
    pkg = importlib.import_module(f"{__package__}.modules")
    names = [m.name for m in pkgutil.iter_modules(pkg.__path__)]
    return sorted([n for n in names if not n.startswith("_")])


def _import_framing_module(name: str):
    if not isinstance(name, str) or not name:
        raise ValueError("cfg.module must be a non-empty string")
    return importlib.import_module(f"{__package__}.modules.{name}")


def _resolve_module_and_cfg(cfg: Config):
    mod = _import_framing_module(cfg.module)
    if not hasattr(mod, "Config"):
        raise AttributeError(f"framing module '{cfg.module}' missing Config")
    if not hasattr(mod, "tx") or not hasattr(mod, "rx"):
        raise AttributeError(f"framing module '{cfg.module}' missing tx/rx")

    module_cfg = cfg.module_cfg if cfg.module_cfg is not None else mod.Config()
    return mod, module_cfg


# ----------------------------
# TX
# ----------------------------

def tx(data: bytes, *, cfg: Config) -> bytes:
    """
    Stage TX: payload bytes -> framed bytes (single frame).
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("tx: data must be bytes-like")
    mod, module_cfg = _resolve_module_and_cfg(cfg)
    return mod.tx(bytes(data), cfg=module_cfg)


# ----------------------------
# RX (stream-aware)
# ----------------------------

def rx(data: bytes, *, cfg: Config) -> Tuple[List[bytes], bytes]:
    """
    Stage RX: consume a byte stream, return (payloads, remainder).

    - Scans for MAGIC defined by module cfg (must expose .magic as bytes)
    - Extracts as many complete frames as possible
    - Leaves incomplete tail in remainder
    - Drops garbage before magic (resync behavior)

    This assumes the selected framing module's rx() expects exactly one full frame
    and validates it (CRC etc.) — like length.py does. :contentReference[oaicite:1]{index=1}
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("rx: data must be bytes-like")

    mod, module_cfg = _resolve_module_and_cfg(cfg)
    stream = bytes(data)

    magic = getattr(module_cfg, "magic", None)
    if not isinstance(magic, (bytes, bytearray)) or len(magic) == 0:
        raise AttributeError("module_cfg must define non-empty bytes attribute: magic")
    magic = bytes(magic)

    payloads: List[bytes] = []
    i = 0

    # We need minimal knowledge to know when a full frame might exist.
    # For length-based module, header structure is fixed and includes payload length.
    # We can either:
    #  (a) parse just enough header here (fast), or
    #  (b) try mod.rx() on candidates and catch ValueError (slower but generic).
    #
    # We'll do (a) for the current length-framing module, and fall back to (b) otherwise.
    if cfg.module == "length":
        payloads, remainder = _rx_length_stream(stream, mod=mod, module_cfg=module_cfg, magic=magic)
        return payloads, remainder

    # Generic fallback: scan magic, then try to validate frames by progressively growing.
    # NOTE: this is conservative and may be slow; you can add per-module fast paths later.
    while True:
        j = stream.find(magic, i)
        if j < 0:
            # no magic found; keep last len(magic)-1 bytes in case magic straddles boundary
            keep = max(0, len(magic) - 1)
            return payloads, stream[-keep:] if keep else b""

        # drop garbage before magic
        if j > 0:
            stream = stream[j:]
            i = 0
        else:
            i = 0

        # Try to find a frame boundary by attempting decode on prefixes.
        # We need at least some minimum; start at len(magic)+1 and grow.
        # In practice, you should add module-specific parsing like _rx_length_stream.
        decoded = False
        for end in range(len(magic) + 1, len(stream) + 1):
            chunk = stream[:end]
            try:
                payload = mod.rx(chunk, cfg=module_cfg)
            except ValueError:
                continue
            payloads.append(payload)
            stream = stream[end:]
            decoded = True
            break

        if not decoded:
            # no complete valid frame yet
            return payloads, stream


def _rx_length_stream(stream: bytes, *, mod, module_cfg, magic: bytes) -> Tuple[List[bytes], bytes]:
    """
    Fast-path for modules/length.py framing.

    Frame format (from module):
      MAGIC(2) | VER(1) | FLAGS(1) | LEN(2) | SEQ(4) | HDRCRC(2) | PAYLOAD | CRC32(4)
    => fixed header total 12 bytes, trailer 4 bytes. :contentReference[oaicite:2]{index=2}
    """
    payloads: List[bytes] = []
    i = 0

    HDR_TOTAL_LEN = 12
    CRC32_LEN = 4

    while True:
        j = stream.find(magic, i)
        if j < 0:
            keep = max(0, len(magic) - 1)
            return payloads, stream[-keep:] if keep else b""

        # drop garbage before magic
        if j > 0:
            stream = stream[j:]
        # now stream starts with magic
        if len(stream) < HDR_TOTAL_LEN + CRC32_LEN:
            return payloads, stream  # need more bytes

        # Parse payload length from header bytes:
        # magic(2) ver(1) flags(1) len(2) => len at offset 4..5 big-endian
        payload_len = int.from_bytes(stream[4:6], "big", signed=False)
        total_len = HDR_TOTAL_LEN + payload_len + CRC32_LEN

        if len(stream) < total_len:
            return payloads, stream  # need more bytes

        frame = stream[:total_len]
        try:
            payload = mod.rx(frame, cfg=module_cfg)
        except ValueError:
            # bad frame: resync by skipping first byte and searching again
            stream = stream[1:]
            continue

        payloads.append(payload)
        stream = stream[total_len:]
        i = 0
