r"""Create original, deterministic PCM WAV fixtures; no downloaded music.

The synthetic mix is ONLY a file/separation-pipeline smoke test. It does not
measure separation quality on actual vocals. The generated audio is CC0.
Run: .venv\Scripts\python.exe scripts\generate_fixtures.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import wave

import numpy as np


def save_wav(path: Path, data: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
    channels = 1 if pcm.ndim == 1 else pcm.shape[1]
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


def generate(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    sample_rate = 44100
    duration = 9.0
    count = round(duration * sample_rate)
    dry = np.zeros(count)
    sharp = np.zeros(count)
    accompaniment = np.zeros((count, 2))
    notes = []
    rng = np.random.default_rng(20261005)
    for index, midi in enumerate((60, 62, 64, 67, 69, 67, 64, 60)):
        start, end = 0.5 + index, 1.2 + index
        begin, finish = round(start * sample_rate), round(end * sample_rate)
        time = np.arange(finish - begin) / sample_rate
        frequency = 440 * 2 ** ((midi - 69) / 12)
        envelope = np.ones(len(time))
        ramp = round(0.025 * sample_rate)
        envelope[:ramp] = np.sin(np.linspace(0, np.pi / 2, ramp)) ** 2
        envelope[-ramp:] = envelope[:ramp][::-1]
        for harmonic, amplitude in ((1, 0.28), (2, 0.14), (3, 0.065), (4, 0.03)):
            dry[begin:finish] += amplitude * envelope * np.sin(2 * np.pi * harmonic * frequency * time)
            sharp[begin:finish] += amplitude * envelope * np.sin(2 * np.pi * harmonic * frequency * 2 ** (35 / 1200) * time)
        notes.append({"start": start, "end": end, "midi": midi, "frequency": frequency,
                      "sharp_frequency": frequency * 2 ** (35 / 1200)})
        # A simple original pad and percussive hit; not a commercial melody or
        # an imitation of any artist. Stereo sides intentionally differ.
        pad_end = min(count, begin + sample_rate)
        pad_time = np.arange(pad_end - begin) / sample_rate
        pad_envelope = np.sin(np.pi * np.minimum(pad_time, 1.0)) ** 2
        for semitones, pan in ((-12, 0.25), (-5, 0.75), (0, 0.5)):
            pad = 0.10 * pad_envelope * np.sin(2 * np.pi * frequency * 2 ** (semitones / 12) * pad_time)
            accompaniment[begin:pad_end, 0] += pad * (1 - pan)
            accompaniment[begin:pad_end, 1] += pad * pan
        drum_length = round(0.14 * sample_rate)
        drum_time = np.arange(drum_length) / sample_rate
        drum = 0.12 * rng.normal(size=drum_length) * np.exp(-drum_time * 35)
        accompaniment[begin:begin + drum_length] += drum[:, None]
    mix = dry[:, None] + accompaniment
    save_wav(output / "melody_dry.wav", dry, sample_rate)
    save_wav(output / "melody_sharp.wav", sharp, sample_rate)
    save_wav(output / "synthetic_mix.wav", mix, sample_rate)
    save_wav(output / "synthetic_bgm.wav", accompaniment, sample_rate)
    save_wav(output / "silence.wav", np.zeros(sample_rate), sample_rate)
    gold = {
        "generator": "scripts/generate_fixtures.py",
        "seed": 20261005,
        "license": "CC0-1.0: original audio synthesized for this project; no third-party recordings.",
        "sample_rate": sample_rate,
        "duration_seconds": duration,
        "sharp_cents": 35.0,
        "notes": notes,
        "evaluation": "Evaluate stable note interiors (start + 0.1 to end - 0.1). Silence and transitions are not correct notes.",
        "separation_warning": "synthetic_mix.wav tests pipeline execution only; it is not a real-vocal separation-quality benchmark.",
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.glob("*.wav"))},
    }
    (output / "gold.json").write_text(json.dumps(gold, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return gold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成不含任何商业录音的自制测试音频。")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "fixtures")
    args = parser.parse_args()
    result = generate(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "files": list(result["sha256"]), "license": "CC0-1.0"}, ensure_ascii=False))
