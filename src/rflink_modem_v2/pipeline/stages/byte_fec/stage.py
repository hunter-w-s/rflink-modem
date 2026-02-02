from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import importlib
import pkgutil


@dataclass(frozen=True)
class Config:
    """
    Byte FEC stage config.

    module: byte-fec module name (e.g. "rs255")
    module_cfg: instance of that module's Config (or None -> defaults)
    """
    module: str = "rs255"
    module_cfg: Any = None


def available_modules() -> list[str]:
    """
    Enumerate available byte_fec modules under pipeline/stages/byte_fec/modules.
    """
    pkg = importlib.import_module(f"{__package__}.modules")
    names = [m.name for m in pkgutil.iter_modules(pkg.__path__)]
    return sorted([n for n in names if not n.startswith("_")])


def _import_byte_fec_module(name: str):
    if not isinstance(name, str) or not name:
        raise ValueError("cfg.module must be a non-empty string")
    return importlib.import_module(f"{__package__}.modules.{name}")


def _resolve_module_and_cfg(cfg: Config):
    mod = _import_byte_fec_module(cfg.module)

    if not hasattr(mod, "Config"):
        raise AttributeError(f"byte_fec module '{cfg.module}' missing Config")
    if not hasattr(mod, "tx") or not hasattr(mod, "rx"):
        raise AttributeError(f"byte_fec module '{cfg.module}' missing tx/rx")

    module_cfg = cfg.module_cfg if cfg.module_cfg is not None else mod.Config()
    return mod, module_cfg


def tx(data: bytes, *, cfg: Config) -> bytes:
    """
    Stage TX: apply byte-level FEC (encode).
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("tx: data must be bytes-like")
    mod, module_cfg = _resolve_module_and_cfg(cfg)
    return mod.tx(bytes(data), cfg=module_cfg)


def rx(data: bytes, *, cfg: Config) -> bytes:
    """
    Stage RX: remove byte-level FEC (decode).
    May raise ValueError if uncorrectable.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("rx: data must be bytes-like")
    mod, module_cfg = _resolve_module_and_cfg(cfg)
    return mod.rx(bytes(data), cfg=module_cfg)
