"""Synthetic NCM fixture: no downloaded music or network dependency."""
import io
import math
from pathlib import Path
import struct
import unittest
import uuid
import shutil
from contextlib import contextmanager

import numpy as np
import soundfile as sf

from backend.audio import decode_audio
from backend.ncm import CORE_KEY, MAGIC, SBOX, _aes128_decrypt, _audio_kind, _multiply, extract_ncm

TEST_RUNS = Path(__file__).resolve().parents[1] / ".test-runs"


@contextmanager
def temporary_folder():
    TEST_RUNS.mkdir(exist_ok=True)
    path = TEST_RUNS / f"ncm-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        assert path.resolve().parent == TEST_RUNS.resolve()
        shutil.rmtree(path)


def encrypt_block(block, key):
    expanded = list(key)
    rcon = 1
    while len(expanded) < 176:
        temp = expanded[-4:]
        if len(expanded) % 16 == 0:
            temp = [SBOX[item] for item in temp[1:] + temp[:1]]
            temp[0] ^= rcon
            rcon = _multiply(rcon, 2)
        start = len(expanded)
        expanded.extend(expanded[start - 16 + i] ^ temp[i] for i in range(4))
    state = list(block)
    def add_key(round_number):
        for i in range(16):
            state[i] ^= expanded[round_number * 16 + i]
    add_key(0)
    for round_number in range(1, 11):
        shifted = state[:]
        for row in range(4):
            for column in range(4):
                shifted[row + 4 * column] = SBOX[state[row + 4 * ((column + row) % 4)]]
        state = shifted
        if round_number < 10:
            for column in range(4):
                at = 4 * column
                a, b, c, d = state[at:at + 4]
                state[at:at + 4] = [
                    _multiply(a, 2) ^ _multiply(b, 3) ^ c ^ d,
                    a ^ _multiply(b, 2) ^ _multiply(c, 3) ^ d,
                    a ^ b ^ _multiply(c, 2) ^ _multiply(d, 3),
                    _multiply(a, 3) ^ b ^ c ^ _multiply(d, 2),
                ]
        add_key(round_number)
    return bytes(state)


def synthetic_ncm(payload):
    key = b"test-local-pitch-key"
    plain = b"neteasecloudmusic" + key
    plain += bytes([16 - len(plain) % 16]) * (16 - len(plain) % 16)
    encrypted_key = b"".join(encrypt_block(plain[i:i + 16], CORE_KEY) for i in range(0, len(plain), 16))
    box = list(range(256)); last = 0
    for i in range(256):
        last = (box[i] + last + key[i % len(key)]) & 255
        box[i], box[last] = box[last], box[i]
    output = bytearray(payload)
    for position in range(len(output)):
        i = (position + 1) & 255
        output[position] ^= box[(box[i] + box[(box[i] + i) & 255]) & 255]
    return (MAGIC + b"\x00\x00" + struct.pack("<I", len(encrypted_key))
            + bytes(value ^ 0x64 for value in encrypted_key)
            + struct.pack("<I", 0) + bytes(9) + struct.pack("<I", 0) + output)


class NcmTests(unittest.TestCase):
    def test_aes_standard_vector(self):
        key = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
        plain = bytes.fromhex("00112233445566778899aabbccddeeff")
        encrypted = bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a")
        self.assertEqual(encrypt_block(plain, key), encrypted)
        self.assertEqual(_aes128_decrypt(encrypted, key), plain)

    def test_synthetic_ncm_flac_decodes_locally_to_pitch_audio(self):
        signal = .2 * np.sin(2 * math.pi * 440 * np.arange(22050) / 44100)
        flac = io.BytesIO()
        sf.write(flac, signal, 44100, format="FLAC")
        source_bytes = synthetic_ncm(flac.getvalue())
        source_bytes = source_bytes[:8] + b"\x01\x70" + source_bytes[10:]
        with temporary_folder() as folder:
            source = folder / "upload.bin"
            target = folder / "original.wav"
            source.write_bytes(source_bytes)
            audio, rate = decode_audio(source, target, original_name="user-owned.ncm")
            self.assertEqual(rate, 44100)
            self.assertEqual(len(audio), 22050)
            self.assertGreater(float(np.sqrt(np.mean(audio ** 2))), .09)
            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertFalse((folder / "ncm-payload.bin").exists())

    def test_rejects_renamed_non_ncm_and_malformed_container(self):
        with temporary_folder() as folder:
            source = folder / "upload.bin"
            output = folder / "original.wav"
            source.write_bytes(b"not an ncm")
            with self.assertRaisesRegex(ValueError, "文件头不是"):
                decode_audio(source, output, original_name="fake.ncm")
            source.write_bytes(MAGIC + b"\x00\x00" + struct.pack("<I", 10000))
            with self.assertRaisesRegex(ValueError, "密钥长度无效"):
                extract_ncm(source, output)
            self.assertFalse(output.exists())

    def test_codec_signature_is_checked(self):
        self.assertEqual(_audio_kind(b"fLaC\x00"), "flac")
        self.assertEqual(_audio_kind(b"ID3\x03\x00"), "mp3")
        self.assertEqual(_audio_kind(b"\xff\xfb\x90\x64"), "mp3")
        with self.assertRaisesRegex(ValueError, "MP3 或 FLAC"):
            _audio_kind(b"RIFF")
