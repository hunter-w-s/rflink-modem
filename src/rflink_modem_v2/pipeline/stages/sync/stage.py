from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple
import importlib
import pkgutil


@dataclass(frozen=True)
class Config:
    """
    Sync stage config.

    module: sync module name (e.g. "word16")
    module_cfg: instance of that module's Config (or None -> defaults)
    """
    module: str = "word16"
    module_cfg: Any = None


def available_modules() -> list[str]:
    """
    Enumerate available sync modules under pipeline/stages/sync/modules.
    """
    pkg = importlib.import_module(f"{__package__}.modules")
    names = [m.name for m in pkgutil.iter_modules(pkg.__path__)]
    return sorted([n for n in names if not n.startswith("_")])


def _import_sync_module(name: str):
    if not isinstance(name, str) or not name:
        raise ValueError("cfg.module must be a non-empty string")
    return importlib.import_module(f"{__package__}.modules.{name}")


def _resolve_module_and_cfg(cfg: Config):
    mod = _import_sync_module(cfg.module)

    if not hasattr(mod, "Config"):
        raise AttributeError(f"sync module '{cfg.module}' missing Config")
    if not hasattr(mod, "tx") or not hasattr(mod, "rx"):
        raise AttributeError(f"sync module '{cfg.module}' missing tx/rx")

    module_cfg = cfg.module_cfg if cfg.module_cfg is not None else mod.Config()
    return mod, module_cfg


def tx(data: bytes, *, cfg: Config) -> bytes:
    """
    Stage TX: add sync/preamble.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("tx: data must be bytes-like")
    mod, module_cfg = _resolve_module_and_cfg(cfg)
    return mod.tx(bytes(data), cfg=module_cfg)


def rx(data: bytes, *, cfg: Config) -> Tuple[bytes, bytes]:
    """
    Stage RX: stream-aware sync align.

    Returns (aligned, remainder).
      - aligned: bytes starting at (or after) the first sync marker, per module config
      - remainder: if no sync found, keep last (len(sync)-1) bytes as a tail to allow
                   sync spanning chunk boundaries on the next call.

    This is intentionally byte-aligned (stateless remainder is easy).
    Bit-offset sync is supported later via a different module type/policy. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("rx: data must be bytes-like")

    mod, module_cfg = _resolve_module_and_cfg(cfg)
    b = bytes(data)

    # For remainder logic we need sync length; require module_cfg exposes sync_word.
    sync_word = getattr(module_cfg, "sync_word", None)
    if not isinstance(sync_word, int) or not (0 <= sync_word <= 0xFFFF):
        raise AttributeError("module_cfg must expose int sync_word in [0,65535]")
    sync_len = 2

    try:
        aligned = mod.rx(b, cfg=module_cfg)
        return aligned, b""
    except ValueError:
        # keep tail so sync spanning boundary can be found next feed
        keep = max(0, sync_len - 1)
        return b"", b[-keep:] if keep else b""
