from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Config:
    """
    Containers stage config.

    Design:
      - tx(x, cfg): typically writes x (e.g., PCM) to cfg.tx_path if enabled
      - rx(x, cfg): typically reads from cfg.rx_path (ignores x) if enabled

    This keeps a consistent "tx/rx" interface like your PHY module API.
    """
    enabled: bool = True

    # Where to write/read
    tx_path: Optional[str] = None
    rx_path: Optional[str] = None

    # WAV parameters
    sample_rate: Optional[int] = None  # required for writing; can be validated on read

    # Safety knobs
    clip: bool = True  # if you later add non-[-1,1] float sources
