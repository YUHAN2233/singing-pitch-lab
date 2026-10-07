"""Numerical and size checks for new compact local audio results."""
from pathlib import Path
import shutil
import uuid

import numpy as np
import soundfile as sf

from backend.storage import read_stored, write_flac24


def test_flac24_retains_over_full_scale_stems_without_clipping():
    root = Path(__file__).resolve().parents[1] / ".test-runs" / f"storage-{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    try:
        rate = 44100
        t = np.arange(rate * 3, dtype=np.float32) / rate
        audio = np.stack((1.37 * np.sin(2 * np.pi * 440 * t),
                          .73 * np.sin(2 * np.pi * 659.25 * t)), axis=1).astype(np.float32)
        path = root / "vocals.flac"
        gain = write_flac24(path, audio, rate)
        restored, actual_rate = read_stored(path, gain)
        assert actual_rate == rate and restored.shape == audio.shape
        assert gain > 1.37
        assert np.max(np.abs(restored - audio)) < 2e-7
        assert path.stat().st_size < audio.nbytes * .65
        info = sf.info(path)
        assert info.format == "FLAC" and info.subtype == "PCM_24"
    finally:
        assert root.resolve().parent.name == ".test-runs"
        shutil.rmtree(root)
