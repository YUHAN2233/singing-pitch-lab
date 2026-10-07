"""Bounded, local-only audio decoding; no shell or network protocols."""
from pathlib import Path
import subprocess
import time
import numpy as np
import soundfile as sf
import imageio_ffmpeg
from .ncm import MAGIC, extract_ncm

MAX_SECONDS = 600


def decode_audio(source: Path, target: Path, cancel=lambda: False, original_name=""):
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    with source.open("rb") as uploaded:
        signature = uploaded.read(len(MAGIC))
    prepared = target.with_name("ncm-payload.bin") if signature == MAGIC else None
    if prepared:
        extract_ncm(source, prepared, cancel)
    elif original_name.lower().endswith(".ncm"):
        raise ValueError("此文件虽名为 NCM，但文件头不是受支持的 NCM 容器；请检查原件。")
    # A tiny compressed file must not expand into unbounded memory or storage.
    command = [executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
               "-protocol_whitelist", "file,pipe", "-i", str(prepared or source), "-vn",
               "-t", str(MAX_SECONDS + 1), "-ar", "44100", "-ac", "2",
               "-c:a", "pcm_f32le", str(target)]
    log_path = target.with_suffix(".decode.log")
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        started = time.monotonic()
        try:
            while process.poll() is None:
                if cancel():
                    raise InterruptedError("已取消")
                if time.monotonic() - started > 120:
                    raise ValueError("解码超过 120 秒，请转换为 WAV 后重试。")
                time.sleep(0.05)
            if process.returncode != 0:
                raise ValueError("无法解码此文件。请选择有效的 WAV、MP3、M4A、FLAC、OGG 或 WebM 音频。")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            if prepared:
                prepared.unlink(missing_ok=True)
    if cancel():
        raise InterruptedError("已取消")
    info = sf.info(target)
    if info.duration > MAX_SECONDS + 0.001:
        raise ValueError("单个文件最多 10 分钟，请先裁剪后再试。")
    if info.duration < 0.15:
        raise ValueError("音频太短，至少需要 0.15 秒。")
    data, rate = sf.read(target, dtype="float32", always_2d=True)
    if not np.isfinite(data).all():
        raise ValueError("音频包含无效采样值。")
    return data, rate
