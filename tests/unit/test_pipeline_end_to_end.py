from __future__ import annotations

from pathlib import Path
import random
import shutil

import numpy as np
import pytest

from rflink_modem.modem.pipeline import tx_bytes_to_pcm, rx_pcm_to_bytes
from rflink_modem.modem.audio.afsk_tx import AFSKTxConfig


def _repo_root() -> Path:
    # tests/unit/test_*.py -> tests/unit -> tests -> repo root
    return Path(__file__).resolve().parents[2]


def _out_dir(case: str) -> Path:
    d = _repo_root() / "tests" / "output" / "pipeline_test" / case
    d.mkdir(parents=True, exist_ok=True)
    return d


def _payload_from_asset(max_len: int = 200) -> bytes:
    # Keep it under RS(255) limits: frame + nsym must be <= 255
    p = _repo_root() / "tests" / "assets" / "cwg.png"
    data = p.read_bytes()
    return data[:max_len]


def _payload_random(rng: random.Random, n: int = 200) -> bytes:
    return bytes(rng.randrange(256) for _ in range(n))


@pytest.mark.parametrize("case", ["asset_png", "random_bytes"])
def test_pipeline_verbose_artifacts(case):
    rng = random.Random(0xBEEF)

    if case == "asset_png":
        payload = _payload_from_asset(200)
    else:
        payload = _payload_random(rng, 200)

    cfg = AFSKTxConfig(
        sample_rate=48000,
        symbol_rate=1200,
        mark_hz=2200.0,
        space_hz=1200.0,
        amplitude=0.8,
        lead_silence_s=0.25,
        trail_silence_s=0.25,
    )

    # Clean old outputs for this case so you only see current run artifacts.
    base = _out_dir(case)
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    tx_dir = base / "tx"
    rx_dir = base / "rx"

    pcm = tx_bytes_to_pcm(
        payload,
        afsk_cfg=cfg,
        nsym=10,
        interleave_depth=8,
        debug_dir=tx_dir,
        verbose=True,
    )

    # Optional deterministic impairment (keep mild)
    pcm_corrupted = np.array(pcm, copy=True)
    # tiny dropout segment in the *middle* of payload region
    mid = pcm_corrupted.shape[0] // 2
    pcm_corrupted[mid : mid + 200] = 0.0

    out = rx_pcm_to_bytes(
        pcm_corrupted,
        afsk_cfg=cfg,
        nsym=10,
        interleave_depth=8,
        debug_dir=rx_dir,
        verbose=True,
    )

    assert out == payload
