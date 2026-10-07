"""Compare old/new local processing on a caller-supplied WAV, without editing it.

Example: .venv/Scripts/python.exe scripts/benchmark_storage.py .data/<id>/original.wav
Only the first 12 seconds are processed. Scratch copies are removed after use.
"""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time
import uuid

import numpy as np
import onnxruntime as ort
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app import MODEL
from backend.audio import decode_audio
from backend.pitch import analyze_pitch
from backend.separation import COMPENSATION, _process_chunks, prepare_audio
from backend.storage import read_stored, write_flac24


def run(source: Path, seconds: int) -> dict:
    root = Path(__file__).resolve().parents[1] / ".test-runs"
    root.mkdir(exist_ok=True)
    scratch = root / f"benchmark-{uuid.uuid4().hex}"
    scratch.mkdir()
    try:
        raw, rate = sf.read(source, dtype="float32", always_2d=True, frames=44100 * seconds)
        input_file = scratch / "input.wav"
        sf.write(input_file, raw, rate, subtype="FLOAT")
        start = time.perf_counter()
        audio, rate = decode_audio(input_file, scratch / "decoded.wav", original_name="input.wav")
        decode_seconds = time.perf_counter() - start
        mix = prepare_audio(audio, rate)
        outputs = {}
        for threads in (4, 16):
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(str(MODEL), sess_options=options, providers=["CPUExecutionProvider"])
            name = session.get_inputs()[0].name
            start = time.perf_counter()
            vocals = _process_chunks(mix, lambda packed: session.run(None, {name: packed})[0])
            vocals *= COMPENSATION
            accompaniment = mix - vocals
            elapsed = time.perf_counter() - start
            outputs[threads] = (vocals, accompaniment, elapsed)
        baseline_vocals, baseline_accompaniment, baseline_seconds = outputs[4]
        vocals, accompaniment, optimized_seconds = outputs[16]
        start = time.perf_counter()
        old_frames = analyze_pitch(baseline_vocals, rate)
        old_pitch_seconds = time.perf_counter() - start
        start = time.perf_counter()
        frames = analyze_pitch(vocals, rate)
        pitch_seconds = time.perf_counter() - start
        start = time.perf_counter()
        for stem, data in (("original", audio), ("vocals", baseline_vocals), ("accompaniment", baseline_accompaniment)):
            sf.write(scratch / f"old-{stem}.wav", data, rate, subtype="FLOAT")
        old_write_seconds = time.perf_counter() - start
        start = time.perf_counter()
        gains = {}
        for stem, data in (("original", audio), ("vocals", vocals), ("accompaniment", accompaniment)):
            gains[stem] = write_flac24(scratch / f"new-{stem}.flac", data, rate)
        new_write_seconds = time.perf_counter() - start
        restored, _ = read_stored(scratch / "new-vocals.flac", gains["vocals"])
        old_bytes = sum((scratch / f"old-{stem}.wav").stat().st_size for stem in gains)
        new_bytes = sum((scratch / f"new-{stem}.flac").stat().st_size for stem in gains)
        old_pitch = [item["frequency"] for item in old_frames]
        new_pitch = [item["frequency"] for item in frames]
        return {
            "source": source.name, "sample_seconds": round(len(audio) / rate, 3),
            "provider": session.get_providers(), "decode_seconds": round(decode_seconds, 3),
            "separation_4_threads_seconds": round(baseline_seconds, 3),
            "separation_16_threads_seconds": round(optimized_seconds, 3),
            "pitch_4_threads_seconds": round(old_pitch_seconds, 3),
            "pitch_16_threads_seconds": round(pitch_seconds, 3),
            "write_three_wav_seconds": round(old_write_seconds, 3),
            "write_three_flac_seconds": round(new_write_seconds, 3),
            "old_three_track_bytes": old_bytes, "new_three_track_bytes": new_bytes,
            "new_fraction": round(new_bytes / old_bytes, 4),
            "model_vocal_max_abs_delta": float(np.max(np.abs(vocals - baseline_vocals))),
            "restored_vocal_max_abs_delta": float(np.max(np.abs(restored - vocals))),
            "pitch_frame_count_equal": len(old_frames) == len(frames),
            "pitch_frequency_equal": old_pitch == new_pitch,
        }
    finally:
        assert scratch.resolve().parent == root.resolve()
        shutil.rmtree(scratch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--seconds", type=int, default=12)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.source, arguments.seconds), ensure_ascii=False, indent=2))
