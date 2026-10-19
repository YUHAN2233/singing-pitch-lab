"""Loopback-only FastAPI app. User audio never leaves this machine."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, RLock
from typing import Literal
from urllib.parse import unquote, urlparse
import io
import hashlib
import json
import os
import re
import shutil
import time
import uuid

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .audio import decode_audio
from .pitch import analyze_pitch
from .duet import analyze_two_voices
from .separation import model_info, separate
from .storage import write_flac24

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("SINGING_LAB_DATA", str(ROOT / ".data"))).resolve()
MODEL = Path(os.environ.get("SINGING_LAB_MODEL", str(ROOT / "models" / "UVR_MDXNET_Main.onnx"))).resolve()
MAX_BYTES = 160 * 1024 * 1024
LOCK = RLock()
JOBS: dict[str, "Job"] = {}
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-audio")


@dataclass
class Job:
    id: str
    name: str
    kind: str
    mode: str
    folder: Path
    status: str = "uploading"
    progress: float = 0.0
    message: str = "接收本机文件"
    result: dict | None = None
    cancel: Event = field(default_factory=Event)
    created: float = field(default_factory=time.time)
    source_sha256: str | None = None
    source_bytes: int | None = None

    def public(self):
        return {"id": self.id, "status": self.status, "progress": self.progress,
                "message": self.message, "result": self.result}


def update(job, status=None, progress=None, message=None):
    with LOCK:
        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = round(float(progress), 4)
        if message is not None:
            job.message = message


def remove_job_files(job):
    """Never claim privacy cleanup succeeded when Windows still holds a file."""
    if job.folder.parent.resolve() != DATA or not re.fullmatch(r"[0-9a-f]{32}", job.folder.name):
        raise RuntimeError("拒绝清理项目数据目录之外的路径")
    try:
        if job.folder.exists():
            shutil.rmtree(job.folder)
        return True
    except OSError:
        job.result = None
        update(job, "error", message="本机文件尚未完全清除，可能仍在播放。请暂停音频后再次点击取消 / 移除。")
        return False


def process_job(job):
    try:
        if job.cancel.is_set():
            return
        update(job, "running", 0.02, "正在本机解码音频")
        audio, rate = decode_audio(job.folder / "upload.bin", job.folder / "original.wav", job.cancel.is_set, job.name)
        if job.cancel.is_set():
            return
        if job.mode.startswith("duet_"):
            process_duet_job(job, audio, rate)
            return
        warnings = []
        stems = {"original": f"/api/audio/{job.id}/original"}
        storage_gains = {}
        if job.mode == "mixed":
            update(job, progress=0.08, message="正在用 MDX 神经网络分离人声与伴奏；CPU 离线处理")
            vocals, accompaniment, rate = separate(audio, rate, MODEL, cancel=job.cancel.is_set,
                progress=lambda p: update(job, progress=0.08 + 0.70 * p))
            storage_gains["vocals"] = write_flac24(job.folder / "vocals.flac", vocals, rate)
            storage_gains["accompaniment"] = write_flac24(job.folder / "accompaniment.flac", accompaniment, rate)
            stems["accompaniment"] = f"/api/audio/{job.id}/accompaniment"
            warnings.append("神经分离可能残留乐器、和声或混响。请试听人声；多人合唱和残留伴奏不适合可靠单声部评分。")
        else:
            vocals = audio
            # Dry vocals are exactly the original, so one compact file serves both routes.
            warnings.append("干声模式基于你确认文件没有伴奏；不会执行分离。若有背景音乐，请切换为混音模式。")
        if job.cancel.is_set():
            return
        stems["vocals"] = f"/api/audio/{job.id}/vocals"
        update(job, progress=0.80, message="分析单声部人声音高；忽略无声与低置信度帧")
        frames = analyze_pitch(vocals, rate, cancel=job.cancel.is_set,
            progress=lambda p: update(job, progress=0.80 + 0.19 * p))
        if job.cancel.is_set():
            return
        storage_gains["original"] = write_flac24(job.folder / "original.flac", audio, rate)
        if job.mode == "dry":
            storage_gains["vocals"] = storage_gains["original"]
        result = {"id": job.id, "name": job.name, "kind": job.kind, "mode": job.mode,
                  "duration": len(vocals) / rate, "frames": frames, "audio": stems,
                  "warnings": warnings, "sample_rate": rate,
                  "storage_format": "FLAC PCM_24 with peak gain", "storage_gains": storage_gains,
                  "algorithm": "YIN / monophonic", "separation": "UVR_MDXNET_Main" if job.mode == "mixed" else None,
                  "created_at": job.created, "completed_at": time.time(),
                  "source_sha256": job.source_sha256, "source_bytes": job.source_bytes}
        if job.cancel.is_set():
            return
        (job.folder / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        (job.folder / "upload.bin").unlink(missing_ok=True)
        (job.folder / "original.wav").unlink(missing_ok=True)
        with LOCK:
            job.result = result
            update(job, "done", 1.0, "分析完成，结果已缓存于本机")
    except Exception as exc:
        if not job.cancel.is_set():
            if remove_job_files(job):
                update(job, "error", message=str(exc)[:400] or "处理失败，请重试")
    finally:
        if job.cancel.is_set():
            job.result = None
            if remove_job_files(job):
                update(job, "cancelled", message="已取消并清除本机文件")


def _track_summary(frames):
    voiced = [frame for frame in frames if frame.get("frequency") is not None]
    return {"total_frames": len(frames), "voiced_frames": len(voiced),
            "voiced_ratio": round(len(voiced) / len(frames), 3) if frames else 0,
            "median_frequency": round(float(np.median([f["frequency"] for f in voiced])), 2) if voiced else None,
            "lowest_note": min(voiced, key=lambda f: f["midi"])["note"] if voiced else None,
            "highest_note": max(voiced, key=lambda f: f["midi"])["note"] if voiced else None}


def process_duet_job(job, audio, rate):
    """Analyze only explicitly selected channels or experimental vocal mixture."""
    warnings = []
    stems = {"original": f"/api/audio/{job.id}/original"}
    storage_gains = {}
    pairs = None
    if job.mode == "duet_channels":
        if audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError("左右声道模式要求双声道音频")
        if np.sqrt(np.mean((audio[:, 0] - audio[:, 1]) ** 2)) < max(.002, np.sqrt(np.mean(audio ** 2)) * .03):
            raise ValueError("左右声道几乎相同，不能作为两位歌手的独立输入")
        update(job, progress=.10, message="分别分析左、右声道")
        left = analyze_pitch(audio[:, 0], rate, cancel=job.cancel.is_set,
                             progress=lambda p: update(job, progress=.10+.42*p))
        right = analyze_pitch(audio[:, 1], rate, cancel=job.cancel.is_set,
                              progress=lambda p: update(job, progress=.52+.42*p))
        tracks = [{"label": "左声道", "frames": left, "summary": _track_summary(left)},
                  {"label": "右声道", "frames": right, "summary": _track_summary(right)}]
        algorithm = "YIN per selected stereo channel"
        warnings.append("左右声道必须确实分别录下两个人；普通立体声混音不能按声道当作两位歌手。")
    else:
        vocals = audio
        if job.mode == "duet_song":
            update(job, progress=.08, message="先用 UVR 分离人声与伴奏")
            vocals, accompaniment, rate = separate(audio, rate, MODEL, cancel=job.cancel.is_set,
                progress=lambda p: update(job, progress=.08+.68*p))
            storage_gains["vocals"] = write_flac24(job.folder / "vocals.flac", vocals, rate)
            storage_gains["accompaniment"] = write_flac24(job.folder / "accompaniment.flac", accompaniment, rate)
            stems.update(vocals=f"/api/audio/{job.id}/vocals",
                         accompaniment=f"/api/audio/{job.id}/accompaniment")
            warnings.append("UVR 只分人声与伴奏，并未分出两个歌手；伴奏残留会造成误检。")
        update(job, progress=.78 if job.mode == "duet_song" else .1,
               message="实验性估计同时出现的两个基频")
        pairs = analyze_two_voices(vocals, rate, cancel=job.cancel.is_set,
            progress=lambda p: update(job, progress=(.78+.17*p) if job.mode == "duet_song" else (.1+.85*p)))
        tracks = []
        for label, key in (("较低音高槽", "lower"), ("较高音高槽", "higher")):
            frames = [{"time": pair["time"], **pair[key], "rms": pair["rms"]} for pair in pairs]
            tracks.append({"label": label, "frames": frames, "summary": _track_summary(frames)})
        algorithm = "experimental harmonic multi-F0, two slots"
        warnings.append("实验性多音高估计：两个槽位按每帧高低排序，不代表固定歌手身份；同音、交叉、谐波重叠及弱声部可能无法区分。")
        if job.mode == "duet_mix":
            warnings.append("此模式要求已无伴奏的双人干声；普通歌曲请先选伴奏分离模式。")
    if job.cancel.is_set():
        return
    storage_gains["original"] = write_flac24(job.folder / "original.flac", audio, rate)
    result = {"id": job.id, "name": job.name, "kind": job.kind, "mode": job.mode,
              "duration": len(audio)/rate, "frames": [], "tracks": tracks,
              "paired_frames": pairs, "audio": stems, "warnings": warnings, "sample_rate": rate,
              "storage_format": "FLAC PCM_24 with peak gain", "storage_gains": storage_gains,
              "algorithm": algorithm, "separation": "UVR_MDXNET_Main" if job.mode == "duet_song" else None,
              "created_at": job.created, "completed_at": time.time(),
              "source_sha256": job.source_sha256, "source_bytes": job.source_bytes}
    if job.cancel.is_set():
        return
    (job.folder / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (job.folder / "upload.bin").unlink(missing_ok=True)
    (job.folder / "original.wav").unlink(missing_ok=True)
    with LOCK:
        job.result = result
        update(job, "done", 1., "双声部分析完成")


@asynccontextmanager
async def lifespan(app):
    DATA.mkdir(parents=True, exist_ok=True)
    # Restore completed analyses after restart, never re-run queued microphone/audio work.
    with LOCK:
        # Interrupted/failed uploads must not become inaccessible private files.
        # Only remove our known UUID job directories, never arbitrary user paths.
        for folder in DATA.iterdir():
            if (folder.is_dir() and re.fullmatch(r"[0-9a-f]{32}", folder.name)
                    and not (folder / "analysis.json").is_file() and folder.name not in JOBS):
                interrupted = Job(folder.name, "上次中断的任务", "reference", "mixed", folder,
                                  status="error", message="上次任务中断")
                if not remove_job_files(interrupted):
                    JOBS[interrupted.id] = interrupted
        for path in sorted(DATA.glob("*/analysis.json"), key=lambda p: p.stat().st_mtime):
            try:
                r = json.loads(path.read_text(encoding="utf-8"))
                if (not isinstance(r, dict) or path.parent.name != r.get("id")
                        or not re.fullmatch(r"[0-9a-f]{32}", path.parent.name)
                        or r.get("kind") not in ("reference", "recording")
                        or r.get("mode") not in ("dry", "mixed", "duet_channels", "duet_mix", "duet_song")
                        or not isinstance(r.get("name"), str)
                        or not isinstance(r.get("frames"), list)
                        or not isinstance(r.get("audio"), dict)
                        or (r.get("storage_format") is not None and
                            (not (path.parent / "original.flac").is_file() or
                             (r.get("mode") in ("mixed", "duet_song") and
                              (not (path.parent / "vocals.flac").is_file() or
                               not (path.parent / "accompaniment.flac").is_file()))))
                        or (r.get("storage_format") is None and
                            (not (path.parent / "original.wav").is_file() or
                             not (path.parent / "vocals.wav").is_file()))):
                    raise ValueError("损坏的任务缓存")
                JOBS[r["id"]] = Job(r["id"], r["name"], r["kind"], r["mode"], path.parent,
                                    "done", 1, "已恢复本机缓存", r,
                                    created=float(r.get("created_at") or path.stat().st_mtime),
                                    source_sha256=r.get("source_sha256"), source_bytes=r.get("source_bytes"))
            except (ValueError, TypeError, KeyError, OSError):
                if re.fullmatch(r"[0-9a-f]{32}", path.parent.name):
                    invalid = Job(path.parent.name, "Unreadable cache", "reference", "mixed", path.parent,
                                  status="error", message="Cache unreadable; files are preserved for inspection.")
                    JOBS[invalid.id] = invalid
    yield
    with LOCK:
        for job in JOBS.values():
            if job.status not in ("done", "error", "cancelled"):
                job.cancel.set()


app = FastAPI(title="音准实验室 · 本机服务", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.middleware("http")
async def local_only(request: Request, call_next):
    host = request.headers.get("host", "").split(":")[0].lower()
    if host not in ("127.0.0.1", "localhost", "testserver"):
        return JSONResponse({"detail": "仅允许本机访问"}, status_code=403)
    origin = request.headers.get("origin")
    if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
        return JSONResponse({"detail": "不允许跨站访问"}, status_code=403)
    if request.headers.get("sec-fetch-site") == "cross-site":
        return JSONResponse({"detail": "不允许跨站访问"}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Permissions-Policy"] = "microphone=(self), camera=(), geolocation=()"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; worker-src 'self' blob:; object-src 'none'; frame-ancestors 'none'"
    return response


@app.get("/api/health")
def health():
    try:
        info = model_info(MODEL)
        model_error = None
    except (OSError, ValueError) as exc:
        info, model_error = None, str(exc)
    return {"version": "1.0.0", "model_ready": info is not None,
            "model_name": "UVR MDX-Net Main (ONNX / CPU)", "privacy": "local-only",
            "model": info, "model_error": model_error,
            "max_bytes": MAX_BYTES, "max_duration_seconds": 600, "data_directory": str(DATA)}


@app.get("/api/jobs")
def list_jobs(summary: bool = True):
    with LOCK:
        if not summary:
            return [j.public() for j in JOBS.values()]
        return [{"id": j.id, "status": j.status, "progress": j.progress, "message": j.message,
                 "name": j.name, "kind": j.kind, "mode": j.mode, "created": j.created,
                 "completed_at": j.result.get("completed_at", j.created) if j.result else None,
                 "duration": j.result.get("duration") if j.result else None,
                 "source_bytes": j.source_bytes,
                 "disk_bytes": sum(p.stat().st_size for p in j.folder.iterdir() if p.is_file()) if j.folder.exists() else 0}
                for j in JOBS.values()]


@app.post("/api/jobs", status_code=202)
async def create_job(request: Request, kind: Literal["reference", "recording"] = "reference",
                     mode: Literal["mixed", "dry", "duet_channels", "duet_mix", "duet_song"] = "mixed"):
    if mode in ("mixed", "duet_song") and not health()["model_ready"]:
        raise HTTPException(503, "缺少人声分离模型。请按安装说明导入匹配的模型；不可把混合音乐当干声评分。")
    length = request.headers.get("content-length")
    if length:
        try:
            if int(length) > MAX_BYTES:
                raise HTTPException(413, "文件超过 160 MiB，请裁剪后重试。")
        except ValueError:
            raise HTTPException(400, "文件长度无效")
    with LOCK:
        # Prune cancelled tombstones so normal file switching never exhausts quota.
        for key in [k for k, j in JOBS.items() if j.status == "cancelled"]:
            JOBS.pop(key, None)
        if sum(j.status in ("uploading", "queued", "running") for j in JOBS.values()) >= 3:
            raise HTTPException(429, "已有 3 个任务，请等待或取消后再试。")
        identifier = uuid.uuid4().hex
        folder = DATA / identifier
        folder.mkdir(parents=True)
        name = unquote(request.headers.get("x-filename", "音频文件"))[:200]
        job = Job(identifier, name, kind, mode, folder)
        JOBS[identifier] = job
    try:
        total = 0
        digest = hashlib.sha256()
        with (folder / "upload.bin").open("wb") as target:
            async for chunk in request.stream():
                total += len(chunk)
                if total > MAX_BYTES:
                    raise HTTPException(413, "文件超过 160 MiB，请裁剪后重试。")
                target.write(chunk)
                digest.update(chunk)
        if total == 0:
            raise HTTPException(400, "文件为空，请重新选择。")
        job.source_sha256, job.source_bytes = digest.hexdigest(), total
        with LOCK:
            duplicate = next((item for item in JOBS.values() if item.id != identifier and item.status == "done"
                              and item.kind == kind and item.mode == mode
                              and item.source_sha256 == job.source_sha256), None)
            if duplicate and remove_job_files(job):
                JOBS.pop(identifier, None)
                return {**duplicate.public(), "reused": True}
        update(job, "queued", 0.01, "等待本机处理")
        POOL.submit(process_job, job)
        return job.public()
    except BaseException:
        with LOCK:
            if remove_job_files(job):
                JOBS.pop(identifier, None)
        raise


def get_job(identifier):
    with LOCK:
        job = JOBS.get(identifier)
        if not job:
            raise HTTPException(404, "任务不存在，可能已清除。")
        return job


@app.get("/api/jobs/{identifier}")
def job_status(identifier: str):
    with LOCK:
        return get_job(identifier).public()


@app.delete("/api/jobs/{identifier}")
def delete_job(identifier: str):
    with LOCK:
        job = get_job(identifier)
        job.cancel.set()
        if job.status in ("done", "error", "cancelled"):
            job.result = None
            if remove_job_files(job):
                update(job, "cancelled", message="已清除本机文件")
        else:
            job.message = "正在取消，当前推理块结束后清除文件"
        return job.public()


@app.get("/api/audio/{identifier}/{stem}")
def get_audio(identifier: str, stem: Literal["original", "vocals", "accompaniment"]):
    job = get_job(identifier)
    if job.status != "done" or job.cancel.is_set():
        raise HTTPException(409, "音频尚未就绪或已取消")
    compact = bool(job.result and job.result.get("storage_format"))
    if compact:
        selected = "original" if stem == "vocals" and job.mode in ("dry", "duet_channels", "duet_mix") else stem
        path = job.folder / f"{selected}.flac"
        media_type, extension = "audio/flac", "flac"
    else:
        path = job.folder / f"{stem}.wav"
        media_type, extension = "audio/wav", "wav"
    if not path.is_file():
        raise HTTPException(404, "此模式没有该音轨")
    return FileResponse(path, media_type=media_type, filename=f"{stem}.{extension}", content_disposition_type="inline")


@app.get("/api/analysis/{identifier}")
def analysis_download(identifier: str):
    job = get_job(identifier)
    if job.status != "done":
        raise HTTPException(409, "分析尚未完成")
    return FileResponse(job.folder / "analysis.json", media_type="application/json", filename="analysis.json")


@app.get("/api/demo")
def demo(kind: Literal["reference", "recording"] = "reference"):
    rate = 44100
    notes = [60, 62, 64, 65, 67, 69, 67, 64]
    sections = []
    for note in notes:
        t = np.arange(int(rate * .7)) / rate
        f = 440 * 2 ** ((note - 69) / 12 + (35 / 1200 if kind == "recording" else 0))
        env = np.minimum(1, t / .025) * np.minimum(1, (.7 - t) / .035)
        sections.append((.28 * np.sin(2 * np.pi * f * t) + .12 * np.sin(4 * np.pi * f * t)) * env)
        sections.append(np.zeros(int(rate * .1)))
    output = io.BytesIO()
    sf.write(output, np.concatenate(sections), rate, format="WAV", subtype="PCM_16")
    return Response(output.getvalue(), media_type="audio/wav", headers={"X-Demo-Description": "synthetic-tone-not-human-singing"})


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
