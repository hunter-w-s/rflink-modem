from __future__ import annotations

import wave
from pathlib import Path
import numpy as np


def float_to_int16(pcm: np.ndarray) -> np.ndarray:
    if pcm.dtype != np.float32 and pcm.dtype != np.float64:
        pcm = pcm.astype(np.float32, copy=False)
    pcm = np.clip(pcm, -1.0, 1.0)
    return (pcm * 32767.0).astype(np.int16)


def write_wav_mono(path: str | Path, pcm: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if pcm.dtype != np.int16:
        pcm_i16 = float_to_int16(pcm)
    else:
        pcm_i16 = pcm

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm_i16.tobytes())


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """
    Read mono WAV into float32 PCM in [-1, 1], return (pcm, sample_rate).
    Supports int16 PCM only (v0.1).
    """
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        n_ch = wf.getnchannels()
        if n_ch != 1:
            raise ValueError(f"Expected mono WAV, got {n_ch} channels")
        sampwidth = wf.getsampwidth()
        if sampwidth != 2:
            raise ValueError(f"Expected 16-bit WAV (sampwidth=2), got {sampwidth}")
        fs = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    pcm_i16 = np.frombuffer(raw, dtype=np.int16)
    pcm = (pcm_i16.astype(np.float32) / 32768.0)
    return pcm, int(fs)
