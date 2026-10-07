"""Local CPU inference for the UVR MDX-Net Main vocal model.

This module does not download models, contact a service, or use stereo centre
subtraction. The ONNX network estimates the complex vocal spectrogram; the
accompaniment is the mixture minus that estimate. Both stems retain the input
timeline. Separation artifacts remain possible and are not a quality guarantee.

Protocol references (checked 2026-10-04):
https://github.com/Anjok07/ultimatevocalremovergui/blob/master/separate.py
https://github.com/Anjok07/ultimatevocalremovergui/blob/master/lib_v5/tfc_tdf_v3.py
https://github.com/Anjok07/ultimatevocalremovergui/blob/master/UVR.py
https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_data_new.json

The implementation below is independent NumPy/SciPy code for the published
inference protocol. Inference intermediates are one fixed-size chunk at a time;
input and the two returned audio arrays necessarily scale with track duration.
"""

from __future__ import annotations

import hashlib
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import fft, signal

SAMPLE_RATE = 44100
N_FFT = 7680
HOP_LENGTH = 1024
DIM_F = 3072
DIM_T = 256
COMPENSATION = 1.043
MODEL_NAME = "UVR_MDXNET_Main.onnx"
MODEL_BYTES = 66759214
# Actual local model checked against UVR's official hash-to-configuration table.
# Main's current released bytes use 7680/3072/1.043, not the common 6144 config.
MODEL_UVR_HASH = "53c4baf4d12c3e6c3831bb8f5b532b93"
MODEL_SHA256 = "8289784cda38543ff431add4070662813311a8cccfc0112ca82f76d9dba2b4ca"
UVR_HASH_TAIL_BYTES = 10000 * 1024  # UVR uses 10000 KiB, NOT 10 MiB.


class CancelledError(Exception):
    """The caller cancelled processing; no partial stems are returned."""


def _check_cancel(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise CancelledError("已取消人声分离")


def uvr_model_hash(path: Path) -> str:
    """Reproduce the official UVR configuration fingerprint."""
    with Path(path).open("rb") as handle:
        handle.seek(max(0, os.fstat(handle.fileno()).st_size - UVR_HASH_TAIL_BYTES))
        return hashlib.md5(handle.read()).hexdigest()


@lru_cache(maxsize=2)
def _verified_model(path_string: str, size: int, mtime_ns: int) -> dict:
    del mtime_ns  # Included in cache key so replacing a model invalidates it.
    path = Path(path_string)
    if size != MODEL_BYTES:
        raise ValueError(f"分离模型不完整：应为 {MODEL_BYTES:,} 字节，实际 {size:,} 字节")
    fingerprint = uvr_model_hash(path)
    if fingerprint != MODEL_UVR_HASH:
        raise ValueError("分离模型指纹不匹配，请使用此项目指定的 UVR_MDXNET_Main.onnx")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != MODEL_SHA256:
        raise ValueError("分离模型完整性校验失败，请重新下载指定模型")
    return {
        "name": MODEL_NAME,
        "bytes": size,
        "sha256": digest.hexdigest(),
        "uvr_hash": fingerprint,
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "dim_f": DIM_F,
        "dim_t": DIM_T,
        "compensation": COMPENSATION,
        "primary_stem": "Vocals",
        "provider": "CPUExecutionProvider",
    }


def model_info(model_path: Path) -> dict:
    """Validate the local model without creating an inference session."""
    path = Path(model_path).resolve()
    if not path.is_file():
        raise FileNotFoundError("未找到分离模型，请先按本机安装说明下载模型")
    stat = path.stat()
    return dict(_verified_model(str(path), stat.st_size, stat.st_mtime_ns))


class STFT:
    """UVR-compatible periodic-Hann, centered, unnormalized complex STFT.

    Forward: channels × samples -> 1 × 2*channels × frequencies × frames.
    The four model channels are left-real, left-imag, right-real, right-imag.
    Inverse restores the unscaled signal with weighted overlap-add.
    """

    def __init__(self, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH,
                 dim_f: int = DIM_F):
        if n_fft < 2 or n_fft % 2 or not 0 < hop_length <= n_fft // 2:
            raise ValueError("invalid STFT FFT/hop settings")
        if not 1 <= dim_f <= n_fft // 2 + 1:
            raise ValueError("invalid STFT frequency count")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.dim_f = dim_f
        self.window = signal.windows.hann(n_fft, sym=False).astype(np.float32)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2 or audio.shape[-1] <= self.n_fft // 2:
            raise ValueError("STFT requires channels × samples longer than half FFT")
        padded = np.pad(audio, ((0, 0), (self.n_fft // 2, self.n_fft // 2)),
                        mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(
            padded, self.n_fft, axis=-1)[:, ::self.hop_length, :]
        spectrum = fft.rfft(frames * self.window, axis=-1, workers=1)
        spectrum = spectrum.transpose(0, 2, 1)[:, :self.dim_f, :]
        packed = np.stack((spectrum.real, spectrum.imag), axis=1)
        return np.ascontiguousarray(packed.reshape(
            1, audio.shape[0] * 2, self.dim_f, spectrum.shape[-1]), dtype=np.float32)

    def inverse(self, packed: np.ndarray) -> np.ndarray:
        packed = np.asarray(packed, dtype=np.float32)
        if packed.ndim != 4 or packed.shape[0] != 1 or packed.shape[1] % 2:
            raise ValueError("invalid packed STFT shape")
        channels, frequencies, frames = packed.shape[1] // 2, packed.shape[2], packed.shape[3]
        if frequencies > self.n_fft // 2 + 1 or frames < 2:
            raise ValueError("invalid packed STFT dimensions")
        parts = packed.reshape(channels, 2, frequencies, frames)
        spectrum = np.zeros((channels, self.n_fft // 2 + 1, frames), np.complex64)
        spectrum[:, :frequencies] = parts[:, 0] + 1j * parts[:, 1]
        waves = fft.irfft(spectrum, n=self.n_fft, axis=1, workers=1)
        total = self.n_fft + self.hop_length * (frames - 1)
        output = np.zeros((channels, total), dtype=np.float32)
        weight = np.zeros(total, dtype=np.float32)
        squared_window = self.window * self.window
        for frame in range(frames):
            offset = frame * self.hop_length
            output[:, offset:offset + self.n_fft] += waves[:, :, frame] * self.window
            weight[offset:offset + self.n_fft] += squared_window
        half = self.n_fft // 2
        output = output[:, half:-half]
        weight = weight[half:-half]
        if np.any(weight < 1e-8):
            raise RuntimeError("STFT inverse has uncovered samples")
        output /= weight
        return output


def prepare_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Validate mono/stereo floating-point audio and resample to 44.1 kHz."""
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise ValueError("采样率必须是整数")
    if not 8000 <= sample_rate <= 192000:
        raise ValueError("支持 8000–192000 Hz 采样率")
    audio = np.asarray(audio)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or audio.shape[1] not in (1, 2) or not len(audio):
        raise ValueError("音频必须是非空单声道或双声道（samples × channels）")
    if audio.dtype.kind not in "fc" or np.iscomplexobj(audio):
        raise ValueError("音频必须是浮点 PCM；整数 PCM 请先按位深归一化")
    if not np.all(np.isfinite(audio)):
        raise ValueError("音频包含无效数值 NaN/Inf")
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        gcd = math.gcd(sample_rate, SAMPLE_RATE)
        audio = signal.resample_poly(audio, SAMPLE_RATE // gcd,
                                     sample_rate // gcd, axis=0).astype(np.float32)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return np.ascontiguousarray(audio)


@lru_cache(maxsize=1)
def _session(path_string: str, size: int, mtime_ns: int):
    _verified_model(path_string, size, mtime_ns)
    import onnxruntime as ort

    options = ort.SessionOptions()
    # Measured on this 22-logical-CPU machine: 16 workers shortened one MDX
    # inference from ~5.7 s at 4 workers to ~2.8-3.3 s, without changing model.
    options.intra_op_num_threads = min(16, os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level = 3
    session = ort.InferenceSession(path_string, sess_options=options,
                                   providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    if len(session.get_inputs()) != 1 or model_input.type != "tensor(float)":
        raise ValueError("分离模型输入类型不符合预期")
    expected = (1, 4, DIM_F, DIM_T)
    if len(model_input.shape) != 4 or any(
            isinstance(got, int) and got != want
            for got, want in zip(model_input.shape, expected)):
        raise ValueError(f"分离模型输入尺寸不符合预期：{model_input.shape}")
    return session, model_input.name


def _process_chunks(audio: np.ndarray, predict: Callable[[np.ndarray], np.ndarray],
                    *, stft: STFT | None = None, dim_t: int = DIM_T,
                    remove_low_bins: bool = True,
                    cancel: Callable[[], bool] | None = None,
                    progress: Callable[[float], None] | None = None) -> np.ndarray:
    """One-batch overlap-add inference; `audio` is samples × 2.

    Private prediction injection exists to test transform/timeline independently
    of neural estimates, never as a production separation fallback.
    """
    transform = stft or STFT()
    size = transform.hop_length * (dim_t - 1)
    trim = transform.n_fft // 2
    stride = size - 2 * trim
    if stride <= 0:
        raise ValueError("STFT chunk must exceed FFT length")
    sample_count = len(audio)
    output = np.zeros_like(audio, dtype=np.float32)
    weights = np.zeros(sample_count, dtype=np.float32)
    # As in UVR's default overlap, FFT-sized overlap reduces chunk edge artifacts.
    window = np.hanning(size).astype(np.float32)
    starts = range(-trim, sample_count, stride)
    count = len(starts)
    if progress:
        progress(0.0)
    for index, start in enumerate(starts):
        _check_cancel(cancel)
        left, right = max(0, start), min(sample_count, start + size)
        block = np.zeros((2, size), dtype=np.float32)
        block[:, left - start:right - start] = audio[left:right].T
        if np.any(block):
            packed = transform(block)
            if remove_low_bins:
                packed[:, :, :3, :] = 0  # Official MDX input preconditioning.
            _check_cancel(cancel)
            prediction = np.asarray(predict(packed), dtype=np.float32)
            _check_cancel(cancel)  # Cancellation occurs no later than this block.
            if prediction.shape != packed.shape or not np.all(np.isfinite(prediction)):
                raise RuntimeError("分离模型返回了无效数据")
            estimate = transform.inverse(prediction)
        else:
            estimate = block  # Exact digital silence cannot contain a vocal.
        local = slice(left - start, right - start)
        blend = window[local]
        output[left:right] += estimate[:, local].T * blend[:, None]
        weights[left:right] += blend
        if progress:
            progress((index + 1) / count)
    _check_cancel(cancel)
    if np.any(weights <= 0):
        raise RuntimeError("人声分离分块覆盖不完整")
    output /= weights[:, None]
    return output


def separate(audio: np.ndarray, sample_rate: int, model_path: Path,
             cancel: Callable[[], bool] | None = None,
             progress: Callable[[float], None] | None = None
             ) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (vocals, accompaniment, 44100), each samples × 2 float32.

    `progress(fraction)` is called once per chunk. `cancel()` is checked before
    and after each ONNX inference; an in-flight ONNX block is allowed to finish.
    A cancellation raises CancelledError and never returns partial output.
    """
    _check_cancel(cancel)
    mix = prepare_audio(audio, sample_rate)
    _check_cancel(cancel)
    path = Path(model_path).resolve()
    model_info(path)
    stat = path.stat()
    session, input_name = _session(str(path), stat.st_size, stat.st_mtime_ns)
    _check_cancel(cancel)

    def predict(packed: np.ndarray) -> np.ndarray:
        return session.run(None, {input_name: packed})[0]

    vocals = _process_chunks(mix, predict, cancel=cancel, progress=progress)
    vocals *= COMPENSATION
    accompaniment = mix - vocals
    if not np.all(np.isfinite(vocals)) or not np.all(np.isfinite(accompaniment)):
        raise RuntimeError("分离结果包含无效数值")
    _check_cancel(cancel)
    return vocals, accompaniment, SAMPLE_RATE
