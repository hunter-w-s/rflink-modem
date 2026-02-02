import numpy as np
import pytest


def _bytes_to_bits_msb(data: bytes) -> np.ndarray:
    bits = []
    for b in data:
        for i in range(8):
            bits.append((b >> (7 - i)) & 1)
    return np.asarray(bits, dtype=np.uint8)


def _bits_to_bytes_msb(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size % 8 != 0:
        raise ValueError(f"bits length must be multiple of 8, got {bits.size}")
    out = bytearray(bits.size // 8)
    k = 0
    for i in range(len(out)):
        v = 0
        for _ in range(8):
            v = (v << 1) | int(bits[k] & 1)
            k += 1
        out[i] = v
    return bytes(out)


def test_afsk_exports_uniform_api():
    import rflink_modem_v2.pipeline.stages.phy.modules.afsk as m

    assert hasattr(m, "Config")
    assert callable(getattr(m, "modulate", None))
    assert callable(getattr(m, "demodulate", None))
    assert callable(getattr(m, "Demodulator", None))


def test_afsk_bits_pcm_bits_roundtrip_known_good_profile():
    from rflink_modem_v2.pipeline.stages.phy.modules.afsk import Config, modulate, demodulate

    cfg = Config(
        sample_rate=96000,
        symbol_rate=9600,
        mark_hz=16000.0,
        space_hz=8000.0,
        amplitude=0.8,
        lead_silence_s=0.0,
        trail_silence_s=0.0,
    )

    payload = bytes(range(32))
    bits_in = _bytes_to_bits_msb(payload)

    pcm = modulate(bits_in, cfg)
    assert isinstance(pcm, np.ndarray)
    assert pcm.ndim == 1
    assert pcm.size > 0
    assert pcm.dtype == np.float32
    assert np.isfinite(pcm).all()

    bits_out = demodulate(pcm, cfg)
    assert bits_out.dtype == np.uint8
    assert bits_out.shape == bits_in.shape
    assert np.array_equal(bits_out, bits_in)


@pytest.mark.parametrize("n_bytes", [1, 2, 7, 8, 15, 16, 31, 32, 64, 127])
def test_afsk_roundtrip_various_lengths(n_bytes: int):
    from rflink_modem_v2.pipeline.stages.phy.modules.afsk import Config, modulate, demodulate

    cfg = Config(
        sample_rate=96000,
        symbol_rate=9600,
        mark_hz=16000.0,
        space_hz=8000.0,
        amplitude=0.8,
        lead_silence_s=0.0,
        trail_silence_s=0.0,
    )

    payload = bytes((i * 13) % 256 for i in range(n_bytes))
    bits_in = _bytes_to_bits_msb(payload)

    pcm = modulate(bits_in, cfg)
    bits_out = demodulate(pcm, cfg)

    assert np.array_equal(bits_out, bits_in)
    assert _bits_to_bytes_msb(bits_out) == payload


def test_afsk_cached_demodulator_matches_stateless():
    from rflink_modem_v2.pipeline.stages.phy.modules.afsk import Config, modulate, demodulate, Demodulator

    cfg = Config(
        sample_rate=96000,
        symbol_rate=9600,
        mark_hz=16000.0,
        space_hz=8000.0,
        amplitude=0.8,
        lead_silence_s=0.0,
        trail_silence_s=0.0,
    )

    bits_in = np.asarray(([0, 1] * 200) + ([1, 1, 0, 0] * 100), dtype=np.uint8)
    pcm = modulate(bits_in, cfg)

    bits_a = demodulate(pcm, cfg)
    bits_b = Demodulator(cfg).demodulate(pcm)

    assert np.array_equal(bits_a, bits_b)
    assert np.array_equal(bits_b, bits_in)


def test_afsk_empty_bits_raises():
    from rflink_modem_v2.pipeline.stages.phy.modules.afsk import Config, modulate

    cfg = Config(sample_rate=96000, symbol_rate=9600, mark_hz=16000.0, space_hz=8000.0)
    with pytest.raises(ValueError):
        modulate(np.asarray([], dtype=np.uint8), cfg)
