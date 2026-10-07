"""Local API integration tests using real FFmpeg decoding and pitch analysis.

Synthetic tones are reproducible fixtures, not evidence of real singing accuracy.
Neural separation execution is covered by test_separation.py; the missing-model
test here verifies that a mixed track cannot silently bypass separation.
"""
from concurrent.futures import ThreadPoolExecutor
import importlib
import io
import json
from pathlib import Path
import shutil
import time
from threading import Event
from urllib.parse import quote
import uuid

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

api = importlib.import_module("backend.app")


@pytest.fixture
def tmp_path():
    """Avoid Python 3.13 mode-0700 temp ACL incompatibility in this sandbox."""
    base = (Path(__file__).resolve().parents[1] / ".test-runs").resolve()
    base.mkdir(exist_ok=True)
    directory = base / f"api-{uuid.uuid4().hex}"
    directory.mkdir()
    yield directory
    assert directory.resolve().parent == base
    shutil.rmtree(directory)


@pytest.fixture
def client(tmp_path, monkeypatch):
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-audio")
    monkeypatch.setattr(api, "DATA", tmp_path / "audio-cache")
    monkeypatch.setattr(api, "JOBS", {})
    monkeypatch.setattr(api, "POOL", pool)
    with TestClient(api.app) as test_client:
        yield test_client
    pool.shutdown(wait=True, cancel_futures=False)


def tone_bytes(frequency=440.0, duration=0.6, amplitude=0.25):
    rate = 44100
    t = np.arange(round(rate * duration)) / rate
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    output = io.BytesIO()
    sf.write(output, signal, rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def upload(client, data, *, kind="reference", mode="dry", name="测试音频.wav"):
    response = client.post(
        f"/api/jobs?kind={kind}&mode={mode}", content=data,
        headers={"Content-Type": "application/octet-stream", "X-Filename": quote(name)},
    )
    assert response.status_code == 202, response.text
    return response.json()["id"]


def wait_terminal(client, identifier, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{identifier}")
        assert response.status_code == 200, response.text
        result = response.json()
        if result["status"] in {"done", "error", "cancelled"}:
            return result
        time.sleep(0.025)
    pytest.fail(f"Job {identifier} did not complete: {result}")


def test_real_demo_analysis_detuning_and_downloads(client):
    """Browser demo traverses upload, FFmpeg, YIN, cache and HTTP range paths."""
    from backend.pitch import compare_tracks

    analyses = []
    for kind in ("reference", "recording"):
        demo = client.get(f"/api/demo?kind={kind}")
        assert demo.headers["x-demo-description"] == "synthetic-tone-not-human-singing"
        identifier = upload(client, demo.content, kind=kind)
        final = wait_terminal(client, identifier)
        assert final["status"] == "done", final
        result = final["result"]
        assert 6.39 < result["duration"] < 6.41
        assert result["name"] == "测试音频.wav"
        assert result["separation"] is None
        assert len(result["frames"]) > 300
        assert {frame["note"] for frame in result["frames"] if frame["frequency"]} >= {"C4", "D4", "E4", "F4", "G4", "A4"}
        assert not (api.DATA / identifier / "upload.bin").exists()
        assert (api.DATA / identifier / "analysis.json").is_file()
        analysis = client.get(f"/api/analysis/{identifier}")
        assert analysis.status_code == 200
        assert analysis.json() == result
        full = client.get(result["audio"]["vocals"])
        assert full.status_code == 200
        assert full.content[:4] == b"fLaC"
        assert full.headers["content-type"] == "audio/flac"
        assert not (api.DATA / identifier / "original.wav").exists()
        assert (api.DATA / identifier / "original.flac").is_file()
        assert not (api.DATA / identifier / "vocals.flac").exists()  # dry track shares original
        partial = client.get(result["audio"]["vocals"], headers={"Range": "bytes=0-127"})
        assert partial.status_code == 206
        assert partial.content == full.content[:128]
        assert partial.headers["content-range"].startswith("bytes 0-127/")
        assert client.get(f"/api/audio/{identifier}/accompaniment").status_code == 404
        analyses.append(result)
    comparison = compare_tracks(analyses[0]["frames"], analyses[1]["frames"])
    assert comparison["summary"]["comparable_frames"] > 230
    assert abs(comparison["summary"]["median_signed_cents"] - 35) < 2


def test_silence_is_unscored_end_to_end(client):
    identifier = upload(client, tone_bytes(amplitude=0))
    final = wait_terminal(client, identifier)
    assert final["status"] == "done", final
    assert final["result"]["frames"]
    assert all(frame["frequency"] is None and frame["confidence"] == 0
               for frame in final["result"]["frames"])


def test_synthetic_ncm_import_runs_real_audio_and_pitch_pipeline(client):
    from tests.test_ncm import synthetic_ncm
    signal = .25 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)
    output = io.BytesIO()
    sf.write(output, signal, 44100, format="FLAC")
    identifier = upload(client, synthetic_ncm(output.getvalue()), name="synthetic.ncm")
    final = wait_terminal(client, identifier)
    assert final["status"] == "done", final
    assert any(frame["note"] == "A4" for frame in final["result"]["frames"])
    assert not (api.DATA / identifier / "ncm-payload.bin").exists()


@pytest.mark.parametrize("path", ["/api/health", "/api/jobs", "/api/demo"])
@pytest.mark.parametrize("headers", [
    {"Host": "attacker.example"},
    {"Origin": "https://attacker.example"},
    {"Sec-Fetch-Site": "cross-site"},
])
def test_cross_origin_and_untrusted_host_rejected(client, path, headers):
    assert client.get(path, headers=headers).status_code == 403


def test_same_origin_health_and_response_privacy_headers(client):
    response = client.get("/api/health", headers={"Origin": "http://testserver"})
    assert response.status_code == 200
    assert response.json()["privacy"] == "local-only"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "connect-src 'self'" in response.headers["content-security-policy"]


def test_mixed_input_refused_when_model_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL", tmp_path / "missing.onnx")
    assert client.get("/api/health").json()["model_ready"] is False
    response = client.post("/api/jobs?mode=mixed", content=tone_bytes())
    assert response.status_code == 503
    assert "模型" in response.json()["detail"]
    assert "setup.ps1" in response.json()["detail"]
    assert client.get("/api/jobs").json() == []
    assert list(api.DATA.iterdir()) == []


@pytest.mark.parametrize("content,headers,status", [
    (b"", {}, 400),
    (b"x", {"Content-Length": str(160 * 1024 * 1024 + 1)}, 413),
    (b"x", {"Content-Length": "invalid"}, 400),
])
def test_invalid_upload_rejected_without_orphan_files(client, content, headers, status):
    response = client.post("/api/jobs?mode=dry", content=content, headers=headers)
    assert response.status_code == status
    assert client.get("/api/jobs").json() == []
    assert list(api.DATA.iterdir()) == []


def test_invalid_audio_failure_delete_and_recovery(client):
    invalid_id = upload(client, b"this is not audio", name="invalid.wav")
    invalid = wait_terminal(client, invalid_id)
    assert invalid["status"] == "error"
    assert "无法解码" in invalid["message"]
    assert client.get(f"/api/audio/{invalid_id}/vocals").status_code == 409
    assert client.get(f"/api/analysis/{invalid_id}").status_code == 409
    assert client.delete(f"/api/jobs/{invalid_id}").status_code == 200
    assert not (api.DATA / invalid_id).exists()
    recovered_id = upload(client, tone_bytes())
    recovered = wait_terminal(client, recovered_id)
    assert recovered["status"] == "done", recovered
    assert any(frame["note"] == "A4" for frame in recovered["result"]["frames"])


def test_running_cancellation_cleans_all_files_then_recovers(client, monkeypatch):
    """Pause the real decoder return to exercise cancellation deterministically."""
    decoded = Event()
    continue_decode = Event()
    actual_decode = api.decode_audio

    def controlled_decode(*args, **kwargs):
        output = actual_decode(*args, **kwargs)
        decoded.set()
        assert continue_decode.wait(10), "Test failed to release decoder"
        return output

    monkeypatch.setattr(api, "decode_audio", controlled_decode)
    identifier = upload(client, tone_bytes(duration=1.0))
    try:
        assert decoded.wait(10)
        assert (api.DATA / identifier / "original.wav").is_file()
        response = client.delete(f"/api/jobs/{identifier}")
        assert response.status_code == 200
        assert client.get(f"/api/audio/{identifier}/original").status_code == 409
    finally:
        continue_decode.set()
    final = wait_terminal(client, identifier)
    assert final["status"] == "cancelled", final
    # Terminal state and physical cleanup are separate thread operations.
    deadline = time.monotonic() + 2
    while (api.DATA / identifier).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not (api.DATA / identifier).exists()
    monkeypatch.setattr(api, "decode_audio", actual_decode)
    assert wait_terminal(client, upload(client, tone_bytes()))["status"] == "done"


def test_queued_cancellation_does_not_run_analysis(client):
    release_worker = Event()
    occupied = api.POOL.submit(release_worker.wait, 10)
    try:
        identifier = upload(client, tone_bytes())
        assert client.get(f"/api/jobs/{identifier}").json()["status"] == "queued"
        client.delete(f"/api/jobs/{identifier}")
    finally:
        release_worker.set()
        occupied.result(10)
    assert wait_terminal(client, identifier)["status"] == "cancelled"
    deadline = time.monotonic() + 2
    while (api.DATA / identifier).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not (api.DATA / identifier).exists()


def test_switching_file_deletes_cache_and_old_tombstone(client):
    first = upload(client, tone_bytes())
    assert wait_terminal(client, first)["status"] == "done"
    assert client.delete(f"/api/jobs/{first}").json()["status"] == "cancelled"
    assert not (api.DATA / first).exists()
    second = upload(client, tone_bytes(frequency=523.251))
    assert wait_terminal(client, second)["status"] == "done"
    assert client.get(f"/api/jobs/{first}").status_code == 404
    assert client.get(f"/api/audio/{first}/vocals").status_code == 404
    assert len(client.get("/api/jobs").json()) == 1
    client.delete(f"/api/jobs/{second}")
    assert list(api.DATA.iterdir()) == []


def test_cached_analysis_restored_after_lifespan_restart(client):
    identifier = upload(client, tone_bytes())
    original = wait_terminal(client, identifier)["result"]
    with api.LOCK:
        api.JOBS.clear()
    with TestClient(api.app) as restarted:
        recovered = restarted.get(f"/api/jobs/{identifier}").json()
        assert recovered["status"] == "done"
        assert recovered["result"] == original
        assert restarted.get(original["audio"]["vocals"]).status_code == 200
    client.delete(f"/api/jobs/{identifier}")
    assert not (api.DATA / identifier).exists()


def test_identical_reference_upload_reuses_result_without_new_analysis(client):
    audio = tone_bytes()
    first = upload(client, audio)
    assert wait_terminal(client, first)["status"] == "done"
    response = client.post("/api/jobs?kind=reference&mode=dry", content=audio,
                           headers={"X-Filename": "same-content.wav"})
    assert response.status_code == 202
    assert response.json()["id"] == first
    assert response.json()["reused"] is True
    assert len(list(api.DATA.iterdir())) == 1
    summary = client.get("/api/jobs").json()[0]
    assert summary["source_bytes"] == len(audio)
    assert summary["duration"] > 0
    assert summary["disk_bytes"] > 0


def test_restart_restores_more_than_twenty_four_reference_results(client):
    for index in range(26):
        identifier = uuid.uuid4().hex
        folder = api.DATA / identifier
        folder.mkdir()
        result = {"id": identifier, "name": f"song-{index}.wav", "kind": "reference",
                  "mode": "dry", "duration": 1.0, "frames": [], "audio": {}}
        (folder / "analysis.json").write_text(json.dumps(result), encoding="utf-8")
        (folder / "original.wav").write_bytes(b"test")
        (folder / "vocals.wav").write_bytes(b"test")
    with api.LOCK:
        api.JOBS.clear()
    with TestClient(api.app) as restarted:
        recovered = [job for job in restarted.get("/api/jobs").json() if job["status"] == "done"]
        assert len(recovered) == 26


def test_legacy_float_wav_cache_remains_playable_after_restart(client):
    identifier = uuid.uuid4().hex
    folder = api.DATA / identifier
    folder.mkdir()
    old_result = {"id": identifier, "name": "old-song.wav", "kind": "reference", "mode": "dry",
                  "duration": .6, "frames": [], "audio": {"original": f"/api/audio/{identifier}/original",
                  "vocals": f"/api/audio/{identifier}/vocals"}}
    (folder / "analysis.json").write_text(json.dumps(old_result), encoding="utf-8")
    samples = np.sin(2 * np.pi * 440 * np.arange(26460) / 44100).astype(np.float32)
    sf.write(folder / "original.wav", samples, 44100, subtype="FLOAT")
    sf.write(folder / "vocals.wav", samples, 44100, subtype="FLOAT")
    with api.LOCK:
        api.JOBS.clear()
    with TestClient(api.app) as restarted:
        job = restarted.get(f"/api/jobs/{identifier}").json()
        assert job["status"] == "done" and job["result"] == old_result
        audio = restarted.get(f"/api/audio/{identifier}/vocals")
        assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
        assert audio.content[:4] == b"RIFF"


def test_invalid_enum_values_and_unknown_id(client):
    assert client.post("/api/jobs?mode=pretend", content=tone_bytes()).status_code == 422
    assert client.post("/api/jobs?kind=other&mode=dry", content=tone_bytes()).status_code == 422
    assert client.get("/api/jobs/not-a-job").status_code == 404
    assert client.delete("/api/jobs/not-a-job").status_code == 404


def test_delete_failure_keeps_retryable_job_and_blocks_audio(client, monkeypatch):
    identifier = upload(client, tone_bytes())
    assert wait_terminal(client, identifier)["status"] == "done"
    actual_remove = api.shutil.rmtree
    calls = []

    def file_in_use(path, *args, **kwargs):
        calls.append(str(path))
        raise PermissionError("Simulated Windows file in use")

    monkeypatch.setattr(api.shutil, "rmtree", file_in_use)
    rejected = client.delete(f"/api/jobs/{identifier}")
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "error"
    assert rejected.json()["result"] is None
    assert "未完全清除" in rejected.json()["message"]
    assert (api.DATA / identifier / "original.flac").exists()
    assert client.get(f"/api/audio/{identifier}/vocals").status_code == 409
    assert len(calls) == 1
    monkeypatch.setattr(api.shutil, "rmtree", actual_remove)
    retried = client.delete(f"/api/jobs/{identifier}")
    assert retried.json()["status"] == "cancelled"
    assert not (api.DATA / identifier).exists()


@pytest.mark.parametrize("bad_json", ["[]", "null", "{", '{"id": "wrong"}'])
def test_invalid_cache_does_not_crash_startup_and_is_preserved(client, bad_json):
    identifier = uuid.uuid4().hex
    folder = api.DATA / identifier
    folder.mkdir()
    (folder / "analysis.json").write_text(bad_json, encoding="utf-8")
    (folder / "vocals.wav").write_bytes(b"private-audio-fragment")
    with TestClient(api.app) as restarted:
        assert restarted.get("/api/health").status_code == 200
        assert restarted.get(f"/api/jobs/{identifier}").json()["status"] == "error"
        assert (folder / "vocals.wav").read_bytes() == b"private-audio-fragment"
        assert restarted.delete(f"/api/jobs/{identifier}").json()["status"] == "cancelled"
        assert not folder.exists()


def test_startup_removes_interrupted_uuid_cache_preserves_unrelated_folder(client):
    interrupted = api.DATA / uuid.uuid4().hex
    interrupted.mkdir()
    (interrupted / "upload.bin").write_bytes(b"private-interrupted-upload")
    unrelated = api.DATA / "user-notes"
    unrelated.mkdir()
    note = unrelated / "keep.txt"
    note.write_text("This is not an app job.", encoding="utf-8")
    with TestClient(api.app) as restarted:
        assert restarted.get("/api/health").status_code == 200
        assert not interrupted.exists()
        assert note.read_text(encoding="utf-8") == "This is not an app job."


def test_jobs_summary_avoids_duplicate_frame_payload(client):
    identifier = upload(client, tone_bytes())
    assert wait_terminal(client, identifier)["status"] == "done"
    summary = client.get("/api/jobs").json()[0]
    assert summary["id"] == identifier
    assert summary["name"] == "测试音频.wav"
    assert "result" not in summary
    full = client.get("/api/jobs?summary=false").json()[0]
    assert full["result"]["frames"]
