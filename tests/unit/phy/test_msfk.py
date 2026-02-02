import numpy as np
import pytest

from rflink_modem_v2.pipeline.stages.phy.modules import msfk


def test_msfk_roundtrip_m4():
    cfg = msfk.Config(
        sample_rate=48_000,
        symbol_rate=1_200,
        tones_hz=(1000.0, 1500.0, 2000.0, 2500.0),  # M=4 => k=2
        amplitude=0.8,
        lead_silence_s=0.0,
        trail_silence_s=0.0,
    )

    rng = np.random.default_rng(123)
    bits = rng.integers(0, 2, size=200, dtype=np.uint8)  # 200 % 2 == 0

    pcm = msfk.tx(bits, cfg)
    out = msfk.rx(pcm, cfg)

    assert out.dtype == np.uint8
    assert np.array_equal(bits, out)


def test_msfk_requires_power_of_two():
    cfg = msfk.Config(
        sample_rate=48_000, symbol_rate=1_200,
        tones_hz=(1000.0, 1500.0, 2000.0),  # M=3 not power of two
    )
    bits = np.array([0, 1, 0, 1], dtype=np.uint8)
    with pytest.raises(ValueError):
        _ = msfk.tx(bits, cfg)


def test_msfk_requires_multiple_of_k():
    cfg = msfk.Config(
        sample_rate=48_000, symbol_rate=1_200,
        tones_hz=(1000.0, 1500.0, 2000.0, 2500.0),  # k=2
    )
    bits = np.array([1, 0, 1], dtype=np.uint8)  # len=3 not multiple of 2
    with pytest.raises(ValueError):
        _ = msfk.tx(bits, cfg)


def test_msfk_cached_rx_matches_function_rx():
    cfg = msfk.Config(
        sample_rate=48_000,
        symbol_rate=1_200,
        tones_hz=(1000.0, 1500.0, 2000.0, 2500.0),
    )
    bits = np.array([0, 1, 1, 0, 0, 0, 1, 1], dtype=np.uint8)  # 8 bits ok for k=2
    pcm = msfk.tx(bits, cfg)

    rx_cached = msfk.RX(cfg).rx(pcm)
    rx_func = msfk.rx(pcm, cfg)

    assert np.array_equal(rx_cached, rx_func)
