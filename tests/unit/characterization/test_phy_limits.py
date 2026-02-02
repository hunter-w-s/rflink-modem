from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from rflink_modem_v2.pipeline.stages.phy import stage as phy_stage


def _snr_db_to_sigma(signal: np.ndarray, snr_db: float) -> float:
    """
    Compute AWGN sigma for target SNR (dB), using signal power estimate.
    SNR(dB) = 10*log10(Psignal / Pnoise)
    """
    x = np.asarray(signal, dtype=np.float32)
    p_sig = float(np.mean(x * x)) + 1e-12
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    return float(np.sqrt(p_noise))


def _awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    sigma = _snr_db_to_sigma(x, snr_db)
    n = rng.normal(loc=0.0, scale=sigma, size=x.shape).astype(np.float32)
    return x + n


def _serialize_cfg(cfg: Any) -> Any:
    if cfg is None:
        return None
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    return repr(cfg)


def _run_roundtrip_once(
    bits: np.ndarray,
    module_name: str,
    module_cfg: Any,
    snr_db: float,
    rng: np.random.Generator,
    *,
    use_cached: bool = True,
) -> bool:
    """
    bits -> PCM -> AWGN -> bits' ; return success (exact match).
    """
    cfg = phy_stage.Config(
        module=module_name,
        module_cfg=module_cfg,
        use_cached_demod=use_cached,  # stage supports regardless
    )

    pcm = phy_stage.tx(bits, cfg=cfg)
    noisy = _awgn(pcm, snr_db, rng)
    bits_out = phy_stage.rx(noisy, cfg=cfg)

    return bool(np.array_equal(bits_out, bits))


def _estimate_snr_threshold(
    bits: np.ndarray,
    module_name: str,
    module_cfg: Any,
    rng: np.random.Generator,
    *,
    snr_hi: float = 35.0,
    snr_lo: float = 0.0,
    trials_per_point: int = 3,
    max_iters: int = 10,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Binary-search-ish threshold estimate:
      - We assume success at snr_hi and (eventual) failure at snr_lo for most configs.
      - We run a small number of trials at each midpoint and majority-vote.

    Returns:
      (snr_min_success_estimate, trace)
    """
    trace: List[Dict[str, Any]] = []

    def majority_success(snr_db: float) -> bool:
        ok = 0
        for _ in range(trials_per_point):
            if _run_roundtrip_once(bits, module_name, module_cfg, snr_db, rng):
                ok += 1
        return ok >= (trials_per_point // 2 + 1)

    # Confirm upper bound is truly “good” (otherwise config is broken)
    if not majority_success(snr_hi):
        trace.append({"snr_db": snr_hi, "majority_success": False, "note": "FAILED_AT_HI"})
        return float("inf"), trace

    lo = snr_lo
    hi = snr_hi
    for _ in range(max_iters):
        mid = (lo + hi) / 2.0
        ok = majority_success(mid)
        trace.append({"snr_db": mid, "majority_success": ok})
        if ok:
            hi = mid  # can go lower (harder)
        else:
            lo = mid  # need higher (easier)

    return hi, trace


@pytest.mark.characterization
def test_phy_limit_finder(tmp_path: Path):
    """
    Characterization test: maps PHY limits rather than asserting a pass/fail envelope.

    Skipped unless explicitly enabled:
      - RUN_CHAR=1 pytest -m characterization

    Outputs:
      - JSON report with threshold estimates for various configs.
    """
    if os.getenv("RUN_CHAR", "0") != "1":
        pytest.skip("Set RUN_CHAR=1 to run characterization sweeps")

    rng = np.random.default_rng(1337)

    # Deterministic bit pattern that stresses transitions + runs.
    rng = np.random.default_rng(1337)

    n_bits = ((512 * 2) + (256 * 4)) * 16
    bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)

    module_name = "afsk"  # extend later: parameterize across modules if desired
    mod = phy_stage._import_phy_module(module_name)  # ok in characterization

    # Baseline config (known-good)
    base_cfg = mod.Config(
        sample_rate=96_000,
        symbol_rate=9_600,
        mark_hz=16_000.0,
        space_hz=8_000.0,
        amplitude=0.8,
        lead_silence_s=0.0,
        trail_silence_s=0.0,
    )

    # Sweep knobs (keep small so it finishes quickly)
    samples_per_symbol_candidates = [4, 6, 8, 10]  # derived by choosing symbol_rate
    tone_separation_hz = [2000.0, 4000.0, 8000.0, 12000.0]  # mark-space separation
    base_center_hz = 12_000.0

    results: List[Dict[str, Any]] = []

    for sps in samples_per_symbol_candidates:
        symbol_rate = int(base_cfg.sample_rate // sps)

        for sep in tone_separation_hz:
            mark = base_center_hz + sep / 2.0
            space = base_center_hz - sep / 2.0

            cfg = mod.Config(
                sample_rate=base_cfg.sample_rate,
                symbol_rate=symbol_rate,
                mark_hz=float(mark),
                space_hz=float(space),
                amplitude=base_cfg.amplitude,
                lead_silence_s=base_cfg.lead_silence_s,
                trail_silence_s=base_cfg.trail_silence_s,
            )

            thr, trace = _estimate_snr_threshold(
                bits,
                module_name,
                cfg,
                rng,
                snr_hi=35.0,
                snr_lo=0.0,
                trials_per_point=3,
                max_iters=9,
            )
            from rflink_modem_v2.pipeline.stages.containers.stage import tx as containers_tx
            from rflink_modem_v2.pipeline.stages.containers.config.stage_config import Config as ContainersCfg


            # --- OPTIONAL: dump one listenable WAV for intuition ---
            if sps == 8 and sep == 8000.0:
                listen_dir = Path("tests/output/listen")
                listen_dir.mkdir(parents=True, exist_ok=True)

                # Build a PCM for this cfg
                stage_cfg = phy_stage.Config(
                    module=module_name,
                    module_cfg=cfg,
                    use_cached_demod=False,
                )
                pcm = phy_stage.tx(bits, cfg=stage_cfg)

                # Make a noisy version slightly above threshold
                snr_listen = float(thr + 1.0)
                noisy = _awgn(pcm, snr_listen, rng)
                markHz = cfg.mark_hz
                spaceHz = cfg.space_hz

                wav_path = listen_dir / f"afsk_sps{sps}_sep{int(sep)}_snr{snr_listen:.1f}_{markHz}_{spaceHz}.wav"

                c_cfg = ContainersCfg(
                    enabled=True,
                    tx_path=str(wav_path),
                    sample_rate=int(cfg.sample_rate),
                )

                containers_tx(noisy, c_cfg)




            results.append(
                {
                    "module": module_name,
                    "cfg": _serialize_cfg(cfg),
                    "derived": {
                        "samples_per_symbol": sps,
                        "tone_separation_hz": sep,
                        "center_hz": base_center_hz,
                    },
                    "snr_threshold_db_est": thr,
                    "trace": trace,
                }
            )

    out_dir = tmp_path / "characterization"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "module": module_name,
        "bits_len": int(bits.size),
        "sweeps": {
            "samples_per_symbol": samples_per_symbol_candidates,
            "tone_separation_hz": tone_separation_hz,
            "center_hz": base_center_hz,
        },
        "results": results,
        "notes": {
            "meaning_of_threshold": "Estimated minimum SNR (dB) for majority-success roundtrip under AWGN.",
            "trials_per_point": 3,
            "binary_search_iters": 9,
            "deterministic_seed": 1337,
        },
    }

    (out_dir / "phy_limits_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nPHY characterization report written to: {out_dir / 'phy_limits_report.json'}\n")

    # Intentional: no assert on capability. This test defines limits; it is not a pass/fail gate.
    # If you later want a floor, add a separate test that asserts threshold <= X for a baseline config.
