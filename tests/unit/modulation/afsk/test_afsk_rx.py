import numpy as np
from pathlib import Path

from rflink_modem.modem.audio.afsk_tx import AFSKTxConfig, bits_to_wav
from rflink_modem.modem.audio.afsk_rx import demod_wav_to_bits


def test_afsk_loopback_exact(tmp_path: Path):
    # Use the known-good “worked” regime: 1200 baud with 1200/2400 tones
    cfg = AFSKTxConfig(
        sample_rate=48000,
        symbol_rate=1200,
        mark_hz=2400.0,
        space_hz=1200.0,
        amplitude=0.8,
        lead_silence_s=0.25,
        trail_silence_s=0.25,
    )

    # Deterministic bit pattern
    bits_in = ([0, 1, 0, 1, 1, 0, 0, 1] * 200)  # 1600 bits

    wav_path = tmp_path / "loop.wav"
    bits_to_wav(bits_in, str(wav_path), cfg)

    bits_out = demod_wav_to_bits(str(wav_path), cfg).tolist()

    assert bits_out == bits_in


def test_afsk_loopback_random_deterministic(tmp_path: Path):
    cfg = AFSKTxConfig(
        sample_rate=48000,
        symbol_rate=1200,
        mark_hz=2400.0,
        space_hz=1200.0,
        amplitude=0.8,
        lead_silence_s=0.25,
        trail_silence_s=0.25,
    )

    rng = np.random.default_rng(12345)
    bits_in = rng.integers(0, 2, size=5000, dtype=np.uint8).tolist()

    wav_path = tmp_path / "rand.wav"
    bits_to_wav(bits_in, str(wav_path), cfg)

    bits_out = demod_wav_to_bits(str(wav_path), cfg).tolist()
    assert bits_out == bits_in
