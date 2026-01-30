import numpy as np
from pathlib import Path

from rflink_modem.containers.wav import write_wav_mono, read_wav_mono


def test_wav_roundtrip_int16(tmp_path: Path):
    fs = 48000
    pcm_i16 = (np.arange(0, 1000, dtype=np.int16) - 500).astype(np.int16)

    p = tmp_path / "t.wav"
    write_wav_mono(p, pcm_i16, fs)

    pcm_f, fs2 = read_wav_mono(p)
    assert fs2 == fs
    assert pcm_f.dtype == np.float32
    assert len(pcm_f) == len(pcm_i16)

    # Convert back to int16 scale and ensure exact equality for this simple ramp
    pcm_back = np.round(pcm_f * 32768.0).astype(np.int16)
    assert np.array_equal(pcm_back, pcm_i16)


def test_wav_roundtrip_float_clip(tmp_path: Path):
    fs = 48000
    # Values beyond [-1,1] should be clipped by writer conversion
    pcm = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)

    p = tmp_path / "clip.wav"
    write_wav_mono(p, pcm, fs)

    pcm_f, fs2 = read_wav_mono(p)
    assert fs2 == fs

    # Ensure the clipped endpoints are within [-1,1]
    assert float(pcm_f.min()) >= -1.0
    assert float(pcm_f.max()) <= 1.0
