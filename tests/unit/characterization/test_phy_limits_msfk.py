from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from rflink_modem_v2.pipeline.stages.phy import stage as phy_stage


def _debruijn_sequence(k: int, n: int) -> List[int]:
    """
    De Bruijn sequence for alphabet size k and subsequences of length n.
    Returns a cyclic sequence as a list of integers in [0..k-1].
    Standard "FKM" algorithm.
    """
    a = [0] * (k * n)
    seq: List[int] = []

    def db(t: int, p: int) -> None:
        if t > n:
            if n % p == 0:
                seq.extend(a[1 : p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return seq


def _symbols_to_bits_msb(sym: np.ndarray, k_bits: int) -> np.ndarray:
    sym = np.asarray(sym, dtype=np.uint16).reshape(-1)
    out = np.empty(sym.size * k_bits, dtype=np.uint8)
    o = 0
    for v in sym:
        for shift in range(k_bits - 1, -1, -1):
            out[o] = (v >> shift) & 1
            o += 1
    return out


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
            hi = mid
        else:
            lo = mid

    return hi, trace


def _make_tones(center_hz: float, spacing_hz: float, m: int) -> Tuple[float, ...]:
    """
    Symmetric tones around center: center +/- (i - (m-1)/2)*spacing
    Example M=4: [-1.5, -0.5, +0.5, +1.5]*spacing + center
    """
    idx = np.arange(m, dtype=np.float64)
    offsets = (idx - (m - 1) / 2.0) * float(spacing_hz)
    tones = (float(center_hz) + offsets).tolist()
    return tuple(float(x) for x in tones)


# Deterministic dataset designed for "hearing" all tones:
# De Bruijn over symbols (order=2 -> covers every 2-symbol transition once per cycle)
def make_bits_for_m(m: int) -> np.ndarray:
    k_bits = int(np.log2(m))
    sym_list = _debruijn_sequence(m, 2)  # length m^2
    sym = np.asarray(sym_list, dtype=np.uint16)

    bits = _symbols_to_bits_msb(sym, k_bits)

    # Repeat to make it long enough to listen to
    reps = 4096 if m <= 4 else 8
    bits = np.tile(bits, reps).astype(np.uint8)

    return bits


@pytest.mark.characterization
def test_phy_limit_finder_msfk(tmp_path: Path):
    """
    Characterization test: maps PHY limits for MSFK rather than asserting pass/fail.

    Skipped unless explicitly enabled:
      RUN_CHAR=1 pytest -m characterization

    Outputs:
      - JSON report with threshold estimates for various configs.
      - Saves ONE WAV: the config with the lowest required SNR that passes.
      - Uses deterministic (non-random) bit pattern.
    """
    if os.getenv("RUN_CHAR", "0") != "1":
        pytest.skip("Set RUN_CHAR=1 to run characterization sweeps")

    # Deterministic noise (for repeatability)
    rng = np.random.default_rng(1337)

    module_name = "msfk"
    mod = phy_stage._import_phy_module(module_name)  # ok in characterization

    # Sweep knobs
    samples_per_symbol_candidates = [4, 6, 8, 10]
    m_candidates = [2, 4, 16]  # k=1,2,4
    tone_spacing_hz = [500.0, 1000.0, 2000.0, 4000.0]
    center_hz = 12_000.0

    # Baseline sample rate
    sample_rate = 96_000

    results: List[Dict[str, Any]] = []

    # Track BEST (lowest threshold) successful config to write one WAV at end
    best: Dict[str, Any] = {
        "thr": float("inf"),
        "cfg": None,
        "derived": None,
        "trace": None,
        "bits": None,
    }

    for m in m_candidates:
        k = int(np.log2(m))
        if 2 ** k != m:
            raise AssertionError("m_candidates must be powers of two")

        # Use De Bruijn bits that exercise ALL tones/transitions for this M
        bits = make_bits_for_m(m)

        # Ensure multiple of k by trimming (no padding; keeps TX reversible)
        rem = bits.size % k
        if rem != 0:
            bits = bits[: bits.size - rem]

        for sps in samples_per_symbol_candidates:
            symbol_rate = int(sample_rate // sps)

            for spacing in tone_spacing_hz:
                tones = _make_tones(center_hz=center_hz, spacing_hz=spacing, m=m)

                cfg = mod.Config(
                    sample_rate=sample_rate,
                    symbol_rate=symbol_rate,
                    tones_hz=tones,
                    amplitude=0.8,
                    lead_silence_s=0.0,
                    trail_silence_s=0.0,
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

                derived = {
                    "M": m,
                    "bits_per_symbol_k": k,
                    "samples_per_symbol": sps,
                    "tone_spacing_hz": float(spacing),
                    "center_hz": float(center_hz),
                    "symbol_rate": int(symbol_rate),
                }

                # Track best successful threshold
                if np.isfinite(thr) and float(thr) < float(best["thr"]):
                    best["thr"] = float(thr)
                    best["cfg"] = cfg
                    best["derived"] = derived
                    best["trace"] = trace
                    best["bits"] = bits.copy()

                results.append(
                    {
                        "module": module_name,
                        "cfg": _serialize_cfg(cfg),
                        "derived": derived,
                        "snr_threshold_db_est": float(thr),
                        "trace": trace,
                    }
                )

    out_dir = tmp_path / "characterization"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "module": module_name,
        "sample_rate": int(sample_rate),
        "sweeps": {
            "samples_per_symbol": samples_per_symbol_candidates,
            "M": m_candidates,
            "tone_spacing_hz": tone_spacing_hz,
            "center_hz": float(center_hz),
        },
        "results": results,
        "notes": {
            "meaning_of_threshold": "Estimated minimum SNR (dB) for majority-success roundtrip under AWGN.",
            "trials_per_point": 3,
            "binary_search_iters": 9,
            "deterministic_seed": 1337,
            "msfk_constraint": "len(bits) must be a multiple of k=log2(M); bits are trimmed per-M to satisfy this.",
            "bits_pattern": "De Bruijn over symbols (order=2), then symbols->MSB bits, then tiled",
            "wav_policy": "One WAV is saved for the best (lowest threshold) successful configuration.",
        },
    }

    report_path = out_dir / "phy_limits_msfk_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nPHY MSFK characterization report written to: {report_path}\n")

    # ------------------------------------------------------------
    # Save ONE WAV: the config with the lowest required SNR that passed
    # ------------------------------------------------------------
    if best["cfg"] is not None:
        cfg_best = best["cfg"]
        bits_best = best["bits"]

        stage_cfg = phy_stage.Config(
            module=module_name,
            module_cfg=cfg_best,
            use_cached_demod=False,
        )

        pcm = phy_stage.tx(bits_best, cfg=stage_cfg)

        # Use slightly above threshold to avoid boundary flakiness
        snr_listen = float(best["thr"] + 0.5)
        # If you want it to be *clearly audible* rather than "just barely decodable", use:
        # snr_listen = float(max(best["thr"] + 0.5, 40.0))

        noisy = _awgn(pcm, snr_listen, np.random.default_rng(999))

        listen_dir = Path("tests/output/listen")
        listen_dir.mkdir(parents=True, exist_ok=True)

        wav_path = listen_dir / (
            f"msfk_best_thr{best['thr']:.2f}dB_"
            f"m{best['derived']['M']}_sps{best['derived']['samples_per_symbol']}_"
            f"df{int(best['derived']['tone_spacing_hz'])}_"
            f"sr{best['derived']['symbol_rate']}.wav"
        )

        try:
            from rflink_modem_v2.pipeline.stages.containers.stage import tx as containers_tx
            from rflink_modem_v2.pipeline.stages.containers.config.stage_config import Config as ContainersCfg

            c_cfg = ContainersCfg(
                enabled=True,
                tx_path=str(wav_path),
                sample_rate=int(cfg_best.sample_rate),
            )
            containers_tx(noisy, c_cfg)

        except Exception:
            import wave

            x = np.clip(noisy, -1.0, 1.0)
            i16 = (x * 32767.0).astype(np.int16)

            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(int(cfg_best.sample_rate))
                wf.writeframes(i16.tobytes())

        meta_path = wav_path.with_suffix(".json")
        meta = {
            "best_threshold_db": float(best["thr"]),
            "snr_written_db": float(snr_listen),
            "cfg": _serialize_cfg(cfg_best),
            "derived": best["derived"],
            "trace": best["trace"],
            "bits_pattern": "De Bruijn over symbols (order=2), then symbols->MSB bits, then tiled",
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"\nSaved BEST WAV: {wav_path}")
        print(f"Saved META:     {meta_path}\n")
    else:
        print("\nNo successful configuration found; no WAV written.\n")

    # Intentional: no assert on capability. This test defines limits; it is not a pass/fail gate.
