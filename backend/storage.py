"""Compact local audio storage with reversible peak scaling for float stems.

FLAC stores integer PCM. A 24-bit conversion has at most one PCM step of
rounding; the saved gain restores model values above full scale on decode.
"""
from pathlib import Path
import os

import numpy as np
import soundfile as sf


def write_flac24(path: Path, audio: np.ndarray, rate: int) -> float:
    data = np.asarray(audio, dtype=np.float32)
    if not data.size or not np.isfinite(data).all():
        raise ValueError("Invalid audio for compact storage")
    peak = float(np.max(np.abs(data)))
    gain = max(1.0, peak * 1.000001)
    target = Path(path)
    temporary = target.with_name(target.name + ".part")
    try:
        sf.write(temporary, data / gain, rate, format="FLAC", subtype="PCM_24")
        info = sf.info(temporary)
        if info.frames != len(data) or info.samplerate != rate or info.channels != (data.shape[1] if data.ndim == 2 else 1):
            raise RuntimeError("Compact audio validation failed")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return gain


def read_stored(path: Path, gain: float = 1.0) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    return audio * gain, rate
