from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config.stage_config import Config
from .modules.wav import read_wav_mono, write_wav_mono


def tx(pcm: np.ndarray, cfg: Config) -> np.ndarray:
    """
    TX-side container operation.

    - If cfg.enabled and cfg.tx_path is set, write pcm to WAV.
    - Always returns pcm unchanged (containers stage is side-effect only).
    """
    x = np.asarray(pcm, dtype=np.float32)

    if not cfg.enabled:
        return x

    if cfg.tx_path is None:
        # Explicit is better than implicit: if enabled, require a destination
        raise ValueError("containers.tx: cfg.tx_path is None but stage is enabled")

    if cfg.sample_rate is None:
        raise ValueError("containers.tx: cfg.sample_rate is required to write WAV")

    # wav.py currently clips internally; cfg.clip reserved for future expansion
    write_wav_mono(cfg.tx_path, x, int(cfg.sample_rate))
    return x


def rx(_ignored: Optional[object], cfg: Config) -> Tuple[np.ndarray, int]:
    """
    RX-side container operation.

    - If cfg.enabled and cfg.rx_path is set, read PCM from WAV.
    - Returns (pcm, fs).
    """
    if not cfg.enabled:
        raise ValueError("containers.rx: disabled; nothing to read")

    if cfg.rx_path is None:
        raise ValueError("containers.rx: cfg.rx_path is None but stage is enabled")

    pcm, fs = read_wav_mono(cfg.rx_path)

    if cfg.sample_rate is not None and int(fs) != int(cfg.sample_rate):
        raise ValueError(f"containers.rx: WAV fs={fs} does not match cfg.sample_rate={cfg.sample_rate}")

    return pcm, fs


class Stage:
    """
    Stage wrapper to match your pipeline stage pattern (run_tx/run_rx).
    Internally delegates to tx()/rx().
    """
    name = "containers"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run_tx(self, data: bytes, ctx: dict) -> tuple[bytes, dict]:
        # Convention: expect PCM in ctx["tx_pcm"] (set by PHY TX stage)
        pcm = ctx.get("tx_pcm", None)
        if pcm is not None:
            ctx["tx_pcm"] = tx(pcm, self.cfg)
        return data, ctx

    def run_rx(self, data: bytes, ctx: dict) -> tuple[bytes, dict]:
        # Convention: if rx_path configured, load PCM into ctx["rx_pcm"]
        pcm, fs = rx(None, self.cfg)
        ctx["rx_pcm"] = pcm
        ctx["sample_rate"] = fs
        return data, ctx
