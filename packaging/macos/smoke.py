"""Exercise the frozen macOS launcher with synthetic audio and no model."""
from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
import wave


def get(url: str):
    with urlopen(url, timeout=5) as response:
        return response.status, response.read(), response.headers.get("Content-Type", "")


def synthetic_wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        frames = bytearray()
        for index in range(44100 * 2):
            left = int(9000 * math.sin(2 * math.pi * 220 * index / 44100))
            right = int(9000 * math.sin(2 * math.pi * 330 * index / 44100))
            frames += left.to_bytes(2, "little", signed=True)
            frames += right.to_bytes(2, "little", signed=True)
        wav.writeframes(frames)
    return output.getvalue()


def submit(base: str, mode: str, audio: bytes) -> None:
    request = Request(f"{base}/api/jobs?kind=reference&mode={mode}", data=audio,
                      headers={"Content-Type": "audio/wav", "X-Filename": "synthetic.wav"},
                      method="POST")
    with urlopen(request, timeout=15) as response:
        assert response.status == 202
        job = json.load(response)
    identifier = job["id"]
    for _ in range(180):
        status, body, _ = get(f"{base}/api/jobs/{identifier}")
        assert status == 200
        state = json.loads(body)
        if state["status"] == "done":
            result = state["result"]
            assert result is not None and result["duration"] > 1
            print(f"{mode}: done, {result['duration']:.2f}s", flush=True)
            return
        if state["status"] in ("error", "cancelled"):
            raise AssertionError(f"{mode}: {state}")
        time.sleep(1)
    raise AssertionError(f"{mode}: timeout")


def exercise(binary: Path, root: Path, env: dict[str, str]) -> None:
    log_path = root / 'smoke-process.log'
    log = log_path.open('wb')
    process = subprocess.Popen([str(binary)], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        base = None
        for _ in range(100):
            assert process.poll() is None, f"launcher exited: {process.returncode}"
            try:
                port = json.loads((root / "state.json").read_text(encoding="utf-8"))["port"]
                candidate = f"http://127.0.0.1:{port}"
                if get(candidate + "/api/health")[0] == 200:
                    base = candidate
                    break
            except (OSError, ValueError, KeyError, URLError):
                pass
            time.sleep(0.3)
        assert base is not None, "server did not become ready"
        health = json.loads(get(base + "/api/health")[1])
        assert health["privacy"] == "local-only"
        assert health["model_ready"] is False, "model must not be in public artifact"
        assert not (root / "models" / "UVR_MDXNET_Main.onnx").exists()
        for route in ("/", "/duet.html"):
            status, body, content_type = get(base + route)
            assert status == 200 and b"html" in body.lower() and "html" in content_type
        again = subprocess.run([str(binary)], env=env, timeout=20)
        assert again.returncode == 0, "second launch failed"
        audio = synthetic_wav()
        submit(base, "dry", audio)
        submit(base, "duet_channels", audio)
        quit_result = subprocess.run([str(binary), "--quit"], env=env, timeout=20)
        assert quit_result.returncode == 0
        assert process.wait(timeout=25) == 0
        assert not (root / "control.sock").exists()
    except Exception:
        print(log_path.read_text(errors='replace'), flush=True)
        error_log = root / 'launcher-error.log'
        if error_log.exists():
            print(error_log.read_text(errors='replace'), flush=True)
        raise
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        log.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="tj-smoke-") as temporary:
        root = Path(temporary)
        env = dict(os.environ, TINGJIAN_USER_ROOT=str(root), TINGJIAN_HEADLESS="1",
                   TINGJIAN_NO_BROWSER="1")
        exercise(args.binary.resolve(), root, env)
        exercise(args.binary.resolve(), root, env)
        env['TINGJIAN_HEADLESS'] = '0'
        exercise(args.binary.resolve(), root, env)
        print('macOS Tk window startup and quit passed', flush=True)
    print("macOS frozen launcher smoke passed", flush=True)


if __name__ == "__main__":
    main()
