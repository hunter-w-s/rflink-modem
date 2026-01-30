from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
import pytest

from rflink_modem.modem.audio.afsk_tx import AFSKTxConfig
from rflink_modem.modem.pipeline import tx_bytes_to_pcm, rx_pcm_to_bytes
from tests.conftest import sample_assets_dir, outputs_unrev_test_dir


def _out_dir(request, case: str) -> Path:
    base = outputs_unrev_test_dir(request) / case
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _payload_from_asset(n: int) -> bytes:
    data = (sample_assets_dir() / "cwg.png").read_bytes()
    return data[:n]


def _payload_random(rng: random.Random, n: int = 200) -> bytes:
    return bytes(rng.randrange(256) for _ in range(n))


@pytest.mark.parametrize("case", ["asset_png", "random_bytes"])
def test_pipeline_verbose_artifacts(case, request):
    rng = random.Random(0xBEEF)
    payload = _payload_from_asset(200) if case == "asset_png" else _payload_random(rng, 200)

    cfg = AFSKTxConfig(
        sample_rate=48000,
        symbol_rate=1200,
        mark_hz=2200.0,
        space_hz=1200.0,
        amplitude=0.8,
        lead_silence_s=0.25,
        trail_silence_s=0.25,
    )

    base = _out_dir(request, case)
    tx_dir = base / "tx"
    rx_dir = base / "rx"
    tx_dir.mkdir(parents=True, exist_ok=True)
    rx_dir.mkdir(parents=True, exist_ok=True)

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
