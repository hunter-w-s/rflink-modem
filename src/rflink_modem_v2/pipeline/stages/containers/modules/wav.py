from __future__ import annotations

import wave
from pathlib import Path
from typing import Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def write_wav_mono(path: PathLike, pcm_f32: np.ndarray, sample_rate: int) -> None:
    """
    Write mono WAV from float32 PCM in [-1, 1] (clipped).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(pcm_f32, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    int16 = (x * 32767.0).astype(np.int16)

    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(int(sample_rate))
        wf.writeframes(int16.tobytes())


def read_wav_mono(path: PathLike) -> Tuple[np.ndarray, int]:
    """
    Read mono WAV and return (float32 PCM in [-1, 1], sample_rate).
    Supports 16-bit PCM mono only (keeps it simple + deterministic).
    """
    p = Path(path)
    with wave.open(str(p), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        fs = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)

    if nch != 1:
        raise ValueError(f"WAV must be mono; got nchannels={nch}")
    if sw != 2:
        raise ValueError(f"WAV must be 16-bit PCM; got sampwidth={sw}")

    x_i16 = np.frombuffer(raw, dtype=np.int16)
    x_f32 = (x_i16.astype(np.float32) / 32767.0).astype(np.float32)
    return x_f32, int(fs)
