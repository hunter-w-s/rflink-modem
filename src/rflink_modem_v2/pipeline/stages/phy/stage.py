from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any
import importlib
import pkgutil

import numpy as np


def available_modules() -> list[str]:
    """
    Enumerate available PHY modules under rflink_modem_v2.pipeline.stages.phy.modules
    (file names become module names).
    """
    pkg = importlib.import_module(f"{__package__}.modules")
    names = [m.name for m in pkgutil.iter_modules(pkg.__path__)]
    return sorted([n for n in names if not n.startswith("_")])


AVAILABLE_MODULES = available_modules()


# ============================
# Stage Config (uniform name)
# ============================

@dataclass(frozen=True)
class Config:
    """
    PHY stage config (module selector + module config payload).

    Compatibility goals:
      - Accept both `use_cached_demod` (legacy) and `use_cached_rx` (new)
      - Work with modules that expose either:
          tx/rx/RX (new)
        or modulate/demodulate/Demodulator (legacy)
        or mixed during migration

    Fields:
      - module: which PHY module to use (e.g. "afsk")
      - module_cfg: instance of that module's Config (or None -> defaults)
      - use_cached_demod: legacy name; still supported
      - use_cached_rx: new name; still supported
    """
    module: str = "afsk"
    module_cfg: Any = None

    # Keep BOTH. If both are specified, use_cached_rx wins.
    use_cached_demod: bool = True
    use_cached_rx: bool = True


# ============================
# Internal helpers
# ============================

def _import_phy_module(name: str):
    """
    Import rflink_modem_v2.pipeline.stages.phy.modules.<name>
    """
    if not isinstance(name, str) or not name:
        raise ValueError("cfg.module must be a non-empty string")
    return importlib.import_module(f"{__package__}.modules.{name}")


def _resolve_module_and_cfg(cfg: Config):
    mod = _import_phy_module(cfg.module)

    if not hasattr(mod, "Config"):
        raise AttributeError(f"PHY module '{cfg.module}' missing Config")

    module_cfg = cfg.module_cfg if cfg.module_cfg is not None else mod.Config()

    # TX function can be tx() or modulate()
    has_tx = callable(getattr(mod, "tx", None)) or callable(getattr(mod, "modulate", None))
    if not has_tx:
        raise AttributeError(
            f"PHY module '{cfg.module}' must provide tx(bits, cfg) or modulate(bits, cfg)"
        )

    # RX path can be:
    # - rx()
    # - demodulate()
    # - RX class
    # - Demodulator class
    has_rx_func = callable(getattr(mod, "rx", None)) or callable(getattr(mod, "demodulate", None))
    has_rx_class = hasattr(mod, "RX") or hasattr(mod, "Demodulator")
    if not (has_rx_func or has_rx_class):
        raise AttributeError(
            f"PHY module '{cfg.module}' must provide rx(pcm,cfg) or demodulate(pcm,cfg) "
            f"or RX(cfg).rx(pcm)/Demodulator(cfg).demodulate(pcm)"
        )

    return mod, module_cfg


def _cache_enabled(cfg: Config) -> bool:
    # If both exist and conflict, prefer the new flag. If only legacy is set, it still works.
    # This allows stage to work "regardless" of which kwarg the caller uses.
    return bool(cfg.use_cached_rx) if "use_cached_rx" in cfg.__dict__ else bool(cfg.use_cached_demod)


@lru_cache(maxsize=32)
def _cached_receiver(module_name: str, module_cfg: Any):
    """
    Cached receiver builder.

    Requires module_cfg to be hashable (frozen dataclass Configs are hashable).
    """
    mod = _import_phy_module(module_name)

    # Prefer new RX class if present
    if hasattr(mod, "RX"):
        return mod.RX(module_cfg)

    # Fall back to legacy Demodulator
    if hasattr(mod, "Demodulator"):
        return mod.Demodulator(module_cfg)

    raise AttributeError(f"PHY module '{module_name}' has no RX/Demodulator for caching")


def _call_tx(mod, bits: Any, module_cfg: Any) -> np.ndarray:
    fn = getattr(mod, "tx", None)
    if callable(fn):
        return fn(bits, module_cfg)

    fn = getattr(mod, "modulate", None)
    if callable(fn):
        return fn(bits, module_cfg)

    raise AttributeError("PHY module missing tx/modulate")  # should be unreachable


def _call_rx_func(mod, pcm: np.ndarray, module_cfg: Any) -> np.ndarray:
    fn = getattr(mod, "rx", None)
    if callable(fn):
        return fn(pcm, module_cfg)

    fn = getattr(mod, "demodulate", None)
    if callable(fn):
        return fn(pcm, module_cfg)

    raise AttributeError("PHY module missing rx/demodulate")  # maybe reachable if only class provided


def _call_receiver_obj(obj, pcm: np.ndarray) -> np.ndarray:
    # New style cached receiver
    fn = getattr(obj, "rx", None)
    if callable(fn):
        return fn(pcm)

    # Legacy cached demodulator
    fn = getattr(obj, "demodulate", None)
    if callable(fn):
        return fn(pcm)

    raise AttributeError("Cached receiver lacks rx() and demodulate()")


# ============================
# Public API (uniform names)
# ============================

def tx(data: Any, *, cfg: Config) -> np.ndarray:
    """
    TX direction for PHY stage:
      bits -> PCM

    `data` is expected to be array-like bits (0/1).
    Returns float32 PCM ndarray.
    """
    mod, module_cfg = _resolve_module_and_cfg(cfg)
    pcm = _call_tx(mod, data, module_cfg)
    return np.asarray(pcm, dtype=np.float32)


def rx(data: Any, *, cfg: Config) -> np.ndarray:
    """
    RX direction for PHY stage:
      PCM -> bits

    `data` is expected to be array-like PCM float.
    Returns uint8 bits ndarray (0/1).
    """
    mod, module_cfg = _resolve_module_and_cfg(cfg)
    pcm = np.asarray(data, dtype=np.float32)

    if _cache_enabled(cfg) and (hasattr(mod, "RX") or hasattr(mod, "Demodulator")):
        r = _cached_receiver(cfg.module, module_cfg)
        bits = _call_receiver_obj(r, pcm)
    else:
        # Prefer direct rx/demodulate if present; otherwise fall back to building a receiver each time.
        if callable(getattr(mod, "rx", None)) or callable(getattr(mod, "demodulate", None)):
            bits = _call_rx_func(mod, pcm, module_cfg)
        else:
            # Only class exists; instantiate non-cached
            if hasattr(mod, "RX"):
                bits = mod.RX(module_cfg).rx(pcm)
            elif hasattr(mod, "Demodulator"):
                bits = mod.Demodulator(module_cfg).demodulate(pcm)
            else:
                raise AttributeError(f"PHY module '{cfg.module}' has no rx path")

    return np.asarray(bits, dtype=np.uint8)
