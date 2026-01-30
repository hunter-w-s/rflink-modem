from __future__ import annotations

import hashlib
import random
import shutil
from pathlib import Path

import numpy as np
import pytest

from rflink_modem.modem.audio.afsk_tx import AFSKTxConfig
from rflink_modem.modem.blocking import tx_bytes_to_pcm_blocks, rx_pcm_blocks_to_bytes
from tests.conftest import sample_assets_dir, outputs_unrev_test_dir


def _payload_from_asset() -> bytes:
    return (sample_assets_dir() / "cwg.png").read_bytes()


def _payload_random(rng: random.Random, n: int = 8000) -> bytes:
    return bytes(rng.randrange(256) for _ in range(n))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _out_dir(request, case: str) -> Path:
    """
    Per-test output directory + case subfolder.
    Also clears any previous contents so you only see current run artifacts.
    """
    base = outputs_unrev_test_dir(request) / case
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    return base


@pytest.mark.parametrize("case", ["asset_png_full", "random_bytes"])
def test_pipeline_block_mode_verbose(case, request):
    rng = random.Random(0xBEEF)
    payload = _payload_from_asset() if case == "asset_png_full" else _payload_random(rng, 8000)

    cfg = AFSKTxConfig(
        sample_rate=96000,
        symbol_rate=9600,
        mark_hz=20000.0,
        space_hz=12000.0,
        amplitude=0.8,
        lead_silence_s=0.0,
        trail_silence_s=0.0,
    )

    base = _out_dir(request, case)

    # --- Save original payload for reference ---
    if case == "asset_png_full":
        (base / "original.png").write_bytes(payload)
    else:
        (base / "original.bin").write_bytes(payload)

    (base / "original.sha256").write_text(_sha256(payload), encoding="utf-8")

    pcms = tx_bytes_to_pcm_blocks(
        payload,
        afsk_cfg=cfg,
        nsym=10,
        interleave_depth=8,
        debug_dir=base,
        verbose=True,
    )

    # Mild deterministic impairment per-block (keep stable)
    pcms_corrupt = []
    for pcm in pcms:
        x = np.array(pcm, copy=True)
        if x.size > 500:
            mid = x.size // 2
            x[mid : mid + 200] = 0.0
        pcms_corrupt.append(x)

    out = rx_pcm_blocks_to_bytes(
        pcms_corrupt,
        afsk_cfg=cfg,
        nsym=10,
        interleave_depth=8,
        debug_dir=base,
        verbose=True,
    )

    # --- Save reconstructed output ---
    if case == "asset_png_full":
        (base / "reconstructed.png").write_bytes(out)
        assert out[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic sanity check
    else:
        (base / "reconstructed.bin").write_bytes(out)

    (base / "reconstructed.sha256").write_text(_sha256(out), encoding="utf-8")

    assert out == payload
