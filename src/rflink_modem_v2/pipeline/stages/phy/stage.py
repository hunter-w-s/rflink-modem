from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any
import importlib

import numpy as np
import pkgutil

def available_modules() -> list[str]:
    """
    Enumerate available PHY modules under rflink_modem_v2.pipeline.stages.phy.modules
    (file names become module names).
    """
    pkg = importlib.import_module(f"{__package__}.modules")
    names = [m.name for m in pkgutil.iter_modules(pkg.__path__)]
    # filter private modules if you ever add any
    return sorted([n for n in names if not n.startswith("_")])

AVAILABLE_MODULES = available_modules()


# ============================
# Stage Config (uniform name)
# ============================

@dataclass(frozen=True)
class Config:
    """
    PHY stage config (module selector + module config payload).

    - module: which PHY module to use (e.g. "afsk")
    - module_cfg: instance of that module's Config (or None -> defaults)
    - use_cached_demod: if True and module provides Demodulator, cache it
    """
    module: str = "afsk"
    module_cfg: Any = None
    use_cached_demod: bool = True


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

    # Expect each module to have a Config class
    if not hasattr(mod, "Config"):
        raise AttributeError(f"PHY module '{cfg.module}' missing Config")

    module_cfg = cfg.module_cfg if cfg.module_cfg is not None else mod.Config()

    # Minimal API contract for PHY modules
    if not hasattr(mod, "modulate"):
        raise AttributeError(f"PHY module '{cfg.module}' missing modulate(bits, cfg)")
    if not hasattr(mod, "demodulate") and not hasattr(mod, "Demodulator"):
        raise AttributeError(
            f"PHY module '{cfg.module}' must provide demodulate(pcm, cfg) or Demodulator(cfg).demodulate(pcm)"
        )

    return mod, module_cfg


@lru_cache(maxsize=32)
def _cached_demodulator(module_name: str, module_cfg: Any):
    """
    Cached Demodulator builder.
    Requires module_cfg to be hashable (your module Configs are frozen dataclasses, so good).
    """
    mod = _import_phy_module(module_name)
    if not hasattr(mod, "Demodulator"):
        raise AttributeError(f"PHY module '{module_name}' has no Demodulator for caching")
    return mod.Demodulator(module_cfg)


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
    pcm = mod.modulate(data, module_cfg)
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

    if cfg.use_cached_demod and hasattr(mod, "Demodulator"):
        d = _cached_demodulator(cfg.module, module_cfg)
        bits = d.demodulate(pcm)
    else:
        if not hasattr(mod, "demodulate"):
            raise AttributeError(f"PHY module '{cfg.module}' missing demodulate(pcm, cfg)")
        bits = mod.demodulate(pcm, module_cfg)

    return np.asarray(bits, dtype=np.uint8)
